import numpy as np
import pytest

torch = pytest.importorskip("torch")

from segbench.datasets.blocks import RandomBlocks, predict_room
from segbench.datasets.s3dis import save_scene
from segbench.models.pointnet2 import (
    FeaturePropagation,
    PointNet2SemSeg,
    farthest_point_sample,
    query_ball_point,
)

SMALL = dict(npoints=(64, 32, 16, 8), radii=(0.1, 0.2, 0.4, 0.8), nsample=8)


def test_fps_indices_distincts_et_bien_repartis():
    torch.manual_seed(0)
    # deux amas éloignés : FPS doit choisir des points dans les deux
    xyz = torch.cat([torch.rand(1, 100, 3) * 0.1, torch.rand(1, 100, 3) * 0.1 + 5], dim=1)
    idx = farthest_point_sample(xyz, 8)
    assert idx.shape == (1, 8)
    assert len(set(idx[0].tolist())) == 8
    assert (idx[0] < 100).any() and (idx[0] >= 100).any()


def test_ball_query_dans_le_rayon():
    torch.manual_seed(0)
    xyz = torch.rand(2, 200, 3)
    centers = xyz[:, :10]
    idx = query_ball_point(0.2, 16, xyz, centers)
    assert idx.shape == (2, 10, 16)
    grouped = torch.stack([xyz[b, idx[b]] for b in range(2)])
    dist = (grouped - centers.unsqueeze(2)).norm(dim=-1)
    assert (dist <= 0.2 + 1e-5).all()


def test_feature_propagation_retrouve_les_valeurs_aux_points_connus():
    fp = FeaturePropagation(4, [4])
    fp.mlp = torch.nn.Identity()  # on ne teste que l'interpolation
    sparse = torch.rand(1, 10, 3)
    feat = torch.rand(1, 4, 10)
    out = fp(sparse, sparse, None, feat)
    torch.testing.assert_close(out, feat, atol=1e-4, rtol=1e-4)


def test_forme_de_la_sortie():
    model = PointNet2SemSeg(num_classes=13, **SMALL)
    out = model(torch.rand(2, 256, 3), torch.rand(2, 6, 256))
    assert out.shape == (2, 13, 256)


def test_apprend_un_petit_exemple():
    """Sur-apprentissage d'un seul lot : la perte doit nettement baisser.

    La courbe est bruitée (FPS part d'un point aléatoire à chaque passe),
    d'où la moyenne sur les dernières itérations.
    """
    torch.manual_seed(0)
    xyz = torch.rand(2, 256, 3)
    target = (xyz[:, :, 2] > 0.5).long()        # classe = au-dessus / en dessous
    features = torch.rand(2, 6, 256)
    model = PointNet2SemSeg(num_classes=2, dropout=0.0, **SMALL)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(60):
        loss = torch.nn.functional.cross_entropy(model(xyz, features), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < 0.6 * losses[0]


@pytest.fixture
def rooms(tmp_path):
    rng = np.random.default_rng(0)
    dirs = []
    for name, n in (("office_1", 3000), ("hallway_1", 1500)):
        coord = rng.uniform([0, 0, 0], [3, 2, 2.5], size=(n, 3)).astype(np.float32)
        scene = {
            "coord": coord,
            "color": rng.integers(0, 256, size=(n, 3)).astype(np.uint8),
            "segment": (coord[:, 2] > 1).astype(np.int16),
        }
        save_scene(scene, tmp_path / "Area_1" / name)
        dirs.append(tmp_path / "Area_1" / name)
    return dirs


def test_random_blocks(rooms):
    torch.manual_seed(0)
    ds = RandomBlocks(rooms, num_point=512, samples_per_epoch=4, min_points=100)
    xyz, features, target = ds[0]
    assert xyz.shape == (512, 3) and features.shape == (6, 512) and target.shape == (512,)
    assert features[:3].max() <= 1.0 and features[3:].min() >= 0.0
    assert set(target.unique().tolist()) <= {0, 1}


def test_predict_room_couvre_tous_les_points(rooms):
    from segbench.datasets.s3dis import load_scene

    scene = load_scene(rooms[0])
    model = PointNet2SemSeg(num_classes=2, **SMALL)
    pred = predict_room(model, scene, num_classes=2, num_point=256, block_size=1.0, stride=0.5, batch_size=8)
    assert pred.shape == (len(scene["coord"]),)
    assert set(np.unique(pred)) <= {0, 1}


@pytest.mark.parametrize("low", [0.008, 0.036, 0.001, 0.006])
def test_predict_room_points_sur_les_bords(low):
    """Points pile sur le bord, pas = taille du bloc : aucun ne doit être oublié.

    Valeurs tirées d'un cas réel : en float32, le centre du premier bloc
    (low + 0.5) est arrondi et le point x = low tombait juste en dehors.
    """
    rng = np.random.default_rng(0)
    coord = rng.uniform([low, low, 0], [5, 4, 2.5], size=(500, 3)).astype(np.float32)
    coord[:2, :2] = [[low, 1.3], [2.0, low]]
    scene = {"coord": coord, "color": np.zeros((500, 3), np.uint8), "segment": np.zeros(500, np.int16)}
    model = PointNet2SemSeg(num_classes=2, **SMALL)
    pred = predict_room(model, scene, num_classes=2, num_point=128, block_size=1.0, stride=1.0)
    assert pred.shape == (500,)


def test_sans_couleur(rooms):
    """Mode LiDAR sans RGB : 3 caractéristiques au lieu de 6, de bout en bout."""
    from segbench.datasets.s3dis import load_scene

    torch.manual_seed(0)
    ds = RandomBlocks(rooms, num_point=256, samples_per_epoch=2, min_points=50, use_color=False)
    _, features, _ = ds[0]
    assert features.shape == (3, 256)
    model = PointNet2SemSeg(num_classes=2, in_channels=3, **SMALL)
    scene = load_scene(rooms[0])
    del scene["color"]
    pred = predict_room(model, scene, num_classes=2, num_point=256, use_color=False)
    assert pred.shape == (len(scene["coord"]),)
    with pytest.raises(ValueError):
        predict_room(model, scene, num_classes=2, num_point=256, use_color=True)


def test_load_pointnet2(tmp_path):
    from segbench.models.pointnet2 import load_pointnet2

    for use_color, channels in ((True, 6), (False, 3)):
        model = PointNet2SemSeg(num_classes=13, in_channels=channels)
        torch.save({"model": model.state_dict(), "use_color": use_color}, tmp_path / "m.pth")
        loaded, flag = load_pointnet2(tmp_path / "m.pth", 13)
        assert flag is use_color and not loaded.training
    # ancien checkpoint, sans la clé use_color : couleur utilisée
    torch.save({"model": PointNet2SemSeg(num_classes=13).state_dict()}, tmp_path / "old.pth")
    assert load_pointnet2(tmp_path / "old.pth", 13)[1] is True


def test_checkpoint_rechargeable_en_mode_securise(tmp_path):
    model = PointNet2SemSeg(num_classes=2, **SMALL)
    torch.save({"model": model.state_dict(), "epoch": 1, "miou": float(np.float64(0.5))}, tmp_path / "best.pth")
    state = torch.load(tmp_path / "best.pth", weights_only=True)
    PointNet2SemSeg(num_classes=2, **SMALL).load_state_dict(state["model"])
