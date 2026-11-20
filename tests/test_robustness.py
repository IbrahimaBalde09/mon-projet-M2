import numpy as np
import pytest

from segbench.corruptions import SEVERITIES
from segbench.robustness import CLEAN, conditions, evaluate_condition, relative_robustness


@pytest.fixture
def scenes():
    rng = np.random.default_rng(0)
    out = []
    for n in (2000, 1000):
        coord = rng.uniform(0, 4, size=(n, 3))
        out.append({
            "coord": coord,
            "color": rng.uniform(0, 255, size=(n, 3)),
            "segment": (coord[:, 2] > 2).astype(np.int64),
            "instance": np.zeros(n, dtype=np.int32),
        })
    return out


def oracle(scene):
    """Prédit la vérité terrain (les outliers, ignorés, reçoivent 0)."""
    return np.clip(scene["segment"], 0, None)


def by_height(scene):
    """« Modèle » qui ne regarde que la hauteur : sensible au bruit."""
    return (scene["coord"][:, 2] > 2).astype(np.int64)


def test_conditions():
    all_conditions = conditions()
    assert all_conditions[0] == (CLEAN, 0)
    assert len(all_conditions) == 1 + sum(len(v) for v in SEVERITIES.values())
    assert conditions(["gaussian_noise"], [1, 5]) == [(CLEAN, 0), ("gaussian_noise", 1), ("gaussian_noise", 5)]


@pytest.mark.parametrize("name", list(SEVERITIES))
def test_oracle_parfait_sous_toute_perturbation(scenes, name):
    row = evaluate_condition(oracle, scenes, 2, name, len(SEVERITIES[name]))
    assert row["miou"] == 1.0


def test_ratio_de_points(scenes):
    row = evaluate_condition(oracle, scenes, 2, "random_subsample", 2)  # 50 %
    assert row["point_ratio"] == pytest.approx(0.5, abs=0.01)
    row = evaluate_condition(oracle, scenes, 2, "add_outliers", 5)      # +20 %
    assert row["point_ratio"] == pytest.approx(1.2, abs=0.01)


def test_le_bruit_degrade_un_modele_sensible(scenes):
    clean = evaluate_condition(by_height, scenes, 2)
    noisy = evaluate_condition(by_height, scenes, 2, "gaussian_noise", 5)
    assert clean["miou"] == 1.0
    assert noisy["miou"] < clean["miou"]


def test_memes_perturbations_pour_tous_les_modeles(scenes):
    seen = []
    evaluate_condition(lambda s: seen.append(s["coord"].copy()) or oracle(s), scenes, 2, "gaussian_noise", 3, seed=7)
    again = []
    evaluate_condition(lambda s: again.append(s["coord"].copy()) or oracle(s), scenes, 2, "gaussian_noise", 3, seed=7)
    for a, b in zip(seen, again):
        np.testing.assert_array_equal(a, b)


def test_relative_robustness():
    rows = [
        {"corruption": CLEAN, "miou": 0.6},
        {"corruption": "gaussian_noise", "miou": 0.3},
        {"corruption": "gaussian_noise", "miou": 0.6},
    ]
    assert relative_robustness(rows) == {"gaussian_noise": pytest.approx(0.75)}
