"""Perturbations de nuages de points pour l'étude de robustesse.

Une scène est un dictionnaire au format Pointcept :
    coord   (N, 3) float  coordonnées en mètres
    color   (N, 3) float  couleurs RGB (optionnel)
    segment (N,)   int    étiquettes sémantiques

Chaque perturbation renvoie une nouvelle scène (l'entrée n'est jamais
modifiée) et prend un ``np.random.Generator`` pour être reproductible.
Les points ajoutés (outliers) reçoivent ``IGNORE_INDEX`` : ils n'ont pas de
vérité terrain et sont exclus des métriques.

Les niveaux de sévérité (1 à 5) suivent l'idée de ModelNet-C / Robo3D ; les
valeurs sont un point de départ à ajuster après les premières expériences.
"""

import numpy as np

IGNORE_INDEX = -1


def _select(scene, idx):
    """Sous-ensemble de points d'une scène (toutes les clés par point)."""
    n = len(scene["coord"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in scene.items()}


def random_subsample(scene, ratio, rng):
    """Garde une fraction ``ratio`` des points, tirés uniformément."""
    n = len(scene["coord"])
    k = max(1, int(round(n * ratio)))
    idx = np.sort(rng.choice(n, size=k, replace=False))
    return _select(scene, idx)


def voxel_subsample(scene, voxel_size, rng):
    """Garde un point tiré au hasard dans chaque voxel de côté ``voxel_size``.

    Contrairement à une moyenne par voxel, on garde des points réels, donc
    des étiquettes exactes.
    """
    keys = np.floor(scene["coord"] / voxel_size).astype(np.int64)
    order = rng.permutation(len(keys))
    _, first = np.unique(keys[order], axis=0, return_index=True)
    return _select(scene, np.sort(order[first]))


def gaussian_noise(scene, sigma, rng, clip=None):
    """Ajoute un bruit gaussien d'écart-type ``sigma`` (m) aux coordonnées."""
    noise = rng.normal(0.0, sigma, size=scene["coord"].shape)
    if clip is not None:
        noise = np.clip(noise, -clip, clip)
    out = dict(scene)
    out["coord"] = scene["coord"] + noise
    return out


def add_outliers(scene, ratio, rng):
    """Ajoute ``ratio * N`` points aléatoires dans la boîte englobante."""
    coord = scene["coord"]
    n_new = int(round(len(coord) * ratio))
    lo, hi = coord.min(axis=0), coord.max(axis=0)
    out = dict(scene)
    out["coord"] = np.concatenate([coord, rng.uniform(lo, hi, size=(n_new, 3))])
    out["segment"] = np.concatenate([scene["segment"], np.full(n_new, IGNORE_INDEX, dtype=scene["segment"].dtype)])
    if "color" in scene:
        out["color"] = np.concatenate([scene["color"], rng.uniform(0, 255, size=(n_new, 3))])
    if "instance" in scene:
        out["instance"] = np.concatenate([scene["instance"], np.full(n_new, -1, dtype=scene["instance"].dtype)])
    return out


def local_dropout(scene, n_holes, radius, rng):
    """Simule des occlusions : retire les points dans ``n_holes`` sphères.

    Les centres sont des points de la scène tirés au hasard.
    """
    coord = scene["coord"]
    centers = coord[rng.choice(len(coord), size=n_holes, replace=False)]
    keep = np.ones(len(coord), dtype=bool)
    for c in centers:
        keep &= np.linalg.norm(coord - c, axis=1) > radius
    return _select(scene, np.flatnonzero(keep))


def drop_color(scene, rng=None):
    """Remplace les couleurs par un gris uniforme (perte de l'information RGB)."""
    out = dict(scene)
    if "color" in scene:
        out["color"] = np.full_like(scene["color"], 127.5)
    return out


SEVERITIES = {
    "random_subsample": [{"ratio": r} for r in (0.75, 0.5, 0.25, 0.1, 0.05)],
    "voxel_subsample": [{"voxel_size": s} for s in (0.02, 0.04, 0.08, 0.12, 0.16)],
    "gaussian_noise": [{"sigma": s} for s in (0.005, 0.01, 0.02, 0.03, 0.05)],
    "add_outliers": [{"ratio": r} for r in (0.01, 0.02, 0.05, 0.1, 0.2)],
    "local_dropout": [{"n_holes": n, "radius": 0.5} for n in (1, 2, 4, 6, 8)],
    "drop_color": [{}],
}

CORRUPTIONS = {
    "random_subsample": random_subsample,
    "voxel_subsample": voxel_subsample,
    "gaussian_noise": gaussian_noise,
    "add_outliers": add_outliers,
    "local_dropout": local_dropout,
    "drop_color": drop_color,
}


def corrupt(scene, name, severity, seed=0):
    """Applique la perturbation ``name`` au niveau ``severity`` (à partir de 1)."""
    params = SEVERITIES[name][severity - 1]
    return CORRUPTIONS[name](scene, rng=np.random.default_rng(seed), **params)
