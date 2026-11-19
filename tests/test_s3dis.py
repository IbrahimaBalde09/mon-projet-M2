import numpy as np
import pytest

from segbench.datasets.s3dis import (
    CLASS_TO_ID,
    CLASSES,
    class_of,
    iter_rooms,
    load_room,
    load_scene,
    read_points,
    room_type,
    save_scene,
)
from segbench.stats import room_stats


def write_points(path, pts):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, pts, fmt="%.3f %.3f %.3f %d %d %d")


@pytest.fixture
def fake_s3dis(tmp_path):
    """Mini S3DIS : Area_1 avec un bureau et un couloir."""
    rng = np.random.default_rng(0)

    def pts(n, z):
        xyz = rng.uniform(0, 4, size=(n, 3))
        xyz[:, 2] = z
        return np.hstack([xyz, rng.integers(0, 256, size=(n, 3))])

    office = tmp_path / "Area_1" / "office_1" / "Annotations"
    write_points(office / "floor_1.txt", pts(100, 0.0))
    write_points(office / "wall_1.txt", pts(50, 1.0))
    write_points(office / "chair_1.txt", pts(20, 0.5))
    write_points(office / "chair_2.txt", pts(30, 0.5))
    write_points(office / "stairs_1.txt", pts(10, 0.2))

    hallway = tmp_path / "Area_1" / "hallway_1" / "Annotations"
    write_points(hallway / "floor_1.txt", pts(40, 0.0))
    # ligne corrompue, comme dans certains fichiers de l'Area 5
    with open(hallway / "floor_1.txt", "ab") as f:
        f.write(b"1.000 2.000 0.000 10 20 3\x100\n")

    (tmp_path / "Area_1" / "notes").mkdir()  # dossier sans Annotations : ignoré
    return tmp_path


def test_read_points_ignore_les_caracteres_parasites(fake_s3dis):
    pts = read_points(fake_s3dis / "Area_1" / "hallway_1" / "Annotations" / "floor_1.txt")
    assert pts.shape == (41, 6)
    np.testing.assert_array_equal(pts[-1], [1, 2, 0, 10, 20, 30])


def test_noms_de_classes_et_de_pieces():
    assert class_of("Annotations/bookcase_12.txt") == "bookcase"
    assert class_of("stairs_1.txt") == "clutter"
    assert room_type("conferenceRoom_2") == "conferenceRoom"
    with pytest.raises(ValueError):
        class_of("piano_1.txt")


def test_iter_rooms(fake_s3dis):
    rooms = [(a, r) for a, r, _ in iter_rooms(fake_s3dis, ["Area_1"])]
    assert rooms == [("Area_1", "hallway_1"), ("Area_1", "office_1")]
    with pytest.raises(FileNotFoundError):
        list(iter_rooms(fake_s3dis, ["Area_9"]))


def test_load_room(fake_s3dis):
    scene = load_room(fake_s3dis / "Area_1" / "office_1")
    assert scene["coord"].shape == (210, 3) and scene["coord"].dtype == np.float32
    assert scene["color"].dtype == np.uint8
    counts = np.bincount(scene["segment"], minlength=len(CLASSES))
    assert counts[CLASS_TO_ID["floor"]] == 100
    assert counts[CLASS_TO_ID["chair"]] == 50
    assert counts[CLASS_TO_ID["clutter"]] == 10
    assert len(np.unique(scene["instance"])) == 5


def test_sauvegarde_et_rechargement(fake_s3dis, tmp_path):
    scene = load_room(fake_s3dis / "Area_1" / "office_1")
    save_scene(scene, tmp_path / "out" / "office_1")
    back = load_scene(tmp_path / "out" / "office_1")
    assert set(back) == {"coord", "color", "segment", "instance"}
    for k in scene:
        np.testing.assert_array_equal(back[k], scene[k])


def test_room_stats(fake_s3dis):
    stats = room_stats(load_room(fake_s3dis / "Area_1" / "office_1"), CLASSES)
    assert stats["n_points"] == 210
    assert stats["n_classes"] == 4          # floor, wall, chair, clutter
    assert stats["n_objects"] == 3          # 2 chaises + 1 escalier (clutter)
    assert stats["frac_floor"] == pytest.approx(100 / 210)
    assert 0 < stats["class_entropy"] <= np.log2(4)
