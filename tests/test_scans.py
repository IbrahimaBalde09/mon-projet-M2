import numpy as np
import pytest

from segbench.scan_io import read_scan
from segbench.scan_stats import compare_to, describe, largest_plane, plane_density, plane_noise

o3d = pytest.importorskip("open3d")


def room(spacing=0.02, sigma=0.0, seed=0, size=(4.0, 3.0, 2.5), grid=True):
    """Sol et deux murs, avec bruit gaussien ; grille régulière ou tirage uniforme de même densité."""
    rng = np.random.default_rng(seed)
    lx, ly, lz = size

    def plane(a, b):
        if grid:
            ga, gb = np.meshgrid(np.arange(0, a, spacing), np.arange(0, b, spacing))
            return ga.ravel(), gb.ravel()
        n = int(a * b / spacing**2)
        return rng.uniform(0, a, n), rng.uniform(0, b, n)

    x, y = plane(lx, ly)
    floor = np.stack([x, y, np.zeros_like(x)], axis=1)
    x, z = plane(lx, lz)
    wall1 = np.stack([x, np.zeros_like(x), z], axis=1)
    y, z = plane(ly, lz)
    wall2 = np.stack([np.zeros_like(y), y, z], axis=1)
    coord = np.concatenate([floor, wall1, wall2])
    return coord + rng.normal(0, sigma, size=coord.shape)


@pytest.mark.parametrize("grid", [True, False])
@pytest.mark.parametrize("sigma", [0.0, 0.01])
def test_densite_independante_du_bruit_et_du_motif(grid, sigma):
    coord = room(spacing=0.04, sigma=sigma, grid=grid)   # 625 points par m²
    normal, dist = largest_plane(coord)
    assert plane_density(coord, normal, dist) == pytest.approx(625, rel=0.1)


@pytest.mark.parametrize("sigma", [0.002, 0.01, 0.02])
def test_bruit_estime(sigma):
    _, dist = largest_plane(room(spacing=0.03, sigma=sigma))
    est, share = plane_noise(dist)
    assert est == pytest.approx(sigma, rel=0.25)
    assert share > 0.3


def test_comparaison_densite_et_bruit():
    ref = describe({"coord": room(spacing=0.02, sigma=0.002), "color": None})
    scan = describe({"coord": room(spacing=0.04, sigma=0.0102, seed=1, grid=False), "color": None})
    c = compare_to(scan, ref)
    assert c["densite_relative"] == pytest.approx(0.25, rel=0.15)   # 4x moins de points par m²
    assert c["niveau_densite"] == 3                                  # random_subsample à 25 %
    assert c["bruit_supplementaire_m"] == pytest.approx(0.01, rel=0.3)
    assert c["niveau_bruit"] == 2                                    # gaussian_noise sigma = 1 cm


def test_axe_vertical_detecte():
    d = describe({"coord": room(spacing=0.05, size=(6.0, 5.0, 2.5)), "color": None})
    assert d["plus_grand_plan"]["axe"] == "z"   # le sol est le plus grand plan


def test_lecture_ply_avec_echelle_et_axe(tmp_path):
    coord = room(spacing=0.1)
    yup = np.stack([coord[:, 0], coord[:, 2], -coord[:, 1]], axis=1) * 1000  # mm, y vers le haut
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(yup))
    o3d.io.write_point_cloud(str(tmp_path / "scan.ply"), pcd)
    scan = read_scan(tmp_path / "scan.ply", scale=0.001, up="y")
    np.testing.assert_allclose(scan["coord"], coord, atol=1e-4)
    assert scan["color"] is None and scan["meta"]["format"] == "ply"


def test_lecture_las_couleur_16_bits_intensite_et_metadonnees(tmp_path):
    laspy = pytest.importorskip("laspy")
    coord = room(spacing=0.1)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001] * 3
    header.offsets = [0.0] * 3
    header.system_identifier = "Scanner de test"
    las = laspy.LasData(header)
    las.x, las.y, las.z = coord[:, 0], coord[:, 1], coord[:, 2]
    las.red = las.green = las.blue = np.full(len(coord), 65535, dtype=np.uint16)
    las.intensity = np.arange(len(coord), dtype=np.uint16)
    las.write(tmp_path / "scan.las")

    scan = read_scan(tmp_path / "scan.las")
    np.testing.assert_allclose(scan["coord"], coord, atol=1e-3)
    assert (scan["color"] == 255).all()                 # 16 bits ramenés sur 8 bits
    assert scan["intensity"][5] == 5
    assert scan["meta"]["systeme"] == "Scanner de test"


def test_format_inconnu(tmp_path):
    (tmp_path / "scan.obj").write_text("")
    with pytest.raises(ValueError):
        read_scan(tmp_path / "scan.obj")
