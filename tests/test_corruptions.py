import numpy as np
import pytest

from segbench.corruptions import (
    CORRUPTIONS,
    IGNORE_INDEX,
    SEVERITIES,
    add_outliers,
    corrupt,
    local_dropout,
    voxel_subsample,
)


@pytest.fixture
def scene():
    rng = np.random.default_rng(0)
    n = 5000
    return {
        "coord": rng.uniform(0, 5, size=(n, 3)),
        "color": rng.uniform(0, 255, size=(n, 3)),
        "segment": rng.integers(0, 13, size=n),
    }


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_tous_niveaux_coherents(scene, name):
    for severity in range(1, len(SEVERITIES[name]) + 1):
        out = corrupt(scene, name, severity)
        n = len(out["coord"])
        assert out["coord"].shape == (n, 3)
        assert out["color"].shape == (n, 3)
        assert out["segment"].shape == (n,)


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_entree_non_modifiee(scene, name):
    before = {k: v.copy() for k, v in scene.items()}
    corrupt(scene, name, len(SEVERITIES[name]))
    for k in scene:
        np.testing.assert_array_equal(scene[k], before[k])


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_reproductible(scene, name):
    a = corrupt(scene, name, 1, seed=42)
    b = corrupt(scene, name, 1, seed=42)
    np.testing.assert_array_equal(a["coord"], b["coord"])


def test_densite_decroit_avec_la_severite(scene):
    for name in ("random_subsample", "voxel_subsample"):
        sizes = [len(corrupt(scene, name, s)["coord"]) for s in range(1, 6)]
        assert sizes == sorted(sizes, reverse=True)


def test_voxel_un_point_par_voxel(scene):
    out = voxel_subsample(scene, 0.5, np.random.default_rng(0))
    keys = np.floor(out["coord"] / 0.5).astype(int)
    assert len(np.unique(keys, axis=0)) == len(keys)


def test_outliers_etiquetes_ignore(scene):
    out = add_outliers(scene, 0.1, np.random.default_rng(0))
    assert (out["segment"] == IGNORE_INDEX).sum() == 500
    np.testing.assert_array_equal(out["segment"][:5000], scene["segment"])


def test_local_dropout_vide_les_spheres(scene):
    rng = np.random.default_rng(0)
    out = local_dropout(scene, n_holes=3, radius=0.5, rng=rng)
    centers = scene["coord"][np.random.default_rng(0).choice(5000, size=3, replace=False)]
    for c in centers:
        assert (np.linalg.norm(out["coord"] - c, axis=1) > 0.5).all()
