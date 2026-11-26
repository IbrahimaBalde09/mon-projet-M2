"""Caractéristiques d'un scan, pour le comparer à S3DIS et aux niveaux de perturbation.

    largest_plane   plus grand plan du scan (mur, sol), par RANSAC
    plane_noise     bruit de mesure estimé sur ce plan
    plane_density   densité de surface (points par m²) mesurée sur ce plan
    describe        résumé d'un scan
    compare_to      positionne le scan par rapport à une scène de référence
                    (densité relative, bruit) et aux niveaux de corruptions.py

Ces mesures disent à quoi « ressemble » notre LiDAR vu depuis S3DIS : plus
clairsemé ? plus bruité ? sans couleur ? C'est ce qui guide le choix du jeu
d'entraînement et l'interprétation du test sur nos propres scans.

La densité est mesurée en comptant les points par carreau sur le plan, et non
par la distance au plus proche voisin : cette distance dépend de la façon dont
le scanner répartit ses points (grille, motif aléatoire) et diminue quand le
bruit augmente, alors que le nombre de points par m² n'en dépend pas.
"""

import numpy as np
import open3d as o3d

from segbench.corruptions import SEVERITIES


def _pcd(coord):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(coord, dtype=np.float64)))


def largest_plane(coord, threshold=0.05, seed=0):
    """Plus grand plan (RANSAC) : renvoie (normale unitaire, distance signée de chaque point)."""
    coord = np.asarray(coord, dtype=np.float64)
    fit = coord
    if len(coord) > 2_000_000:  # RANSAC sur un sous-ensemble, distances sur tous les points
        fit = coord[np.random.default_rng(seed).choice(len(coord), 2_000_000, replace=False)]
    o3d.utility.random.seed(seed)
    (a, b, c, d), _ = _pcd(fit).segment_plane(distance_threshold=threshold / 2, ransac_n=3, num_iterations=1000)
    norm = np.linalg.norm([a, b, c])
    normal = np.array([a, b, c]) / norm
    return normal, (coord @ normal + d / norm)


def plane_noise(dist, threshold=0.05):
    """Bruit : écart-type robuste des distances au plan des points proches du plan.

    Renvoie (sigma en m, part des points sur ce plan). Un vrai mur n'est jamais
    parfaitement plat : c'est une borne haute du bruit du capteur.
    """
    band = dist[np.abs(dist) <= threshold]
    residual = band - np.median(band)
    return 1.4826 * float(np.median(np.abs(residual))), len(band) / len(dist)  # MAD -> écart-type


def plane_density(coord, normal, dist, threshold=0.05, cell=0.1):
    """Points par m² sur le plan : médiane, sur les carreaux occupés, de (nombre de points / surface).

    La médiane écarte les carreaux du bord, à moitié remplis.
    """
    near = np.abs(dist) <= threshold
    projected = coord[near] - np.outer(dist[near], normal)
    # base orthonormée (u, v) du plan
    u = np.cross(normal, [1.0, 0, 0] if abs(normal[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    cells = np.floor(np.stack([projected @ u, projected @ v], axis=1) / cell).astype(np.int64)
    _, counts = np.unique(cells, axis=0, return_counts=True)
    return float(np.median(counts)) / cell**2


def describe(scan):
    """Résumé d'un scan lu par scan_io.read_scan (ou d'une scène S3DIS préparée)."""
    coord = np.asarray(scan["coord"], dtype=np.float64)
    extent = coord.max(axis=0) - coord.min(axis=0)
    normal, dist = largest_plane(coord)
    sigma, share = plane_noise(dist)
    density = plane_density(coord, normal, dist)
    return {
        "n_points": int(len(coord)),
        "dimensions_m": [round(float(v), 2) for v in extent],
        "couleur": scan.get("color") is not None,
        "intensite": scan.get("intensity") is not None,
        "densite_pts_m2": density,
        "espacement_m": 1 / np.sqrt(density),  # équivalent sur une grille régulière
        "bruit_m": sigma,
        "plus_grand_plan": {"normale": [round(float(x), 3) for x in normal], "axe": "xyz"[int(np.argmax(np.abs(normal)))],
                            "part_points": round(share, 3)},
    }


def compare_to(desc, ref):
    """Densité relative et bruit en plus d'un scan par rapport à une scène de référence."""
    density_ratio = desc["densite_pts_m2"] / ref["densite_pts_m2"]
    noise_levels = [p["sigma"] for p in SEVERITIES["gaussian_noise"]]
    density_levels = [p["ratio"] for p in SEVERITIES["random_subsample"]]
    extra_noise = max(0.0, desc["bruit_m"] ** 2 - ref["bruit_m"] ** 2) ** 0.5  # les variances s'ajoutent
    return {
        "densite_relative": density_ratio,
        # niveau de random_subsample le plus proche quand le scan est moins dense
        "niveau_densite": (1 + int(np.argmin([abs(np.log(r / density_ratio)) for r in density_levels]))
                           if density_ratio < 0.875 else 0),
        "bruit_supplementaire_m": extra_noise,
        # niveau de gaussian_noise le plus proche du bruit en plus de celui de la référence
        "niveau_bruit": (1 + int(np.argmin([abs(s - extra_noise) for s in noise_levels]))
                         if extra_noise >= noise_levels[0] / 2 else 0),
    }
