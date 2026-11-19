"""Visualise une perturbation sur une pièce synthétique (sans jeu de données).

Exemples :
    python scripts/demo_corruptions.py --corruption gaussian_noise --severity 3
    python scripts/demo_corruptions.py --corruption local_dropout --severity 5

À gauche la scène d'origine, à droite la scène perturbée, colorées par classe.
"""

import argparse

import numpy as np
import open3d as o3d

from segbench.corruptions import CORRUPTIONS, IGNORE_INDEX, SEVERITIES, corrupt

# sol, plafond, mur, table
CLASS_COLORS = np.array([[0.6, 0.5, 0.4], [0.8, 0.8, 0.8], [0.4, 0.6, 0.8], [0.9, 0.5, 0.2]])
OUTLIER_COLOR = np.array([1.0, 0.0, 0.0])


def synthetic_room(n_per_m2=400, size=(5.0, 4.0, 2.8), seed=0):
    """Pièce rectangulaire avec une table, échantillonnée sur ses surfaces."""
    rng = np.random.default_rng(seed)
    lx, ly, lz = size
    parts = []

    def plane(origin, u, v, label):
        area = np.linalg.norm(u) * np.linalg.norm(v)
        a, b = rng.uniform(size=(2, int(area * n_per_m2), 1))
        parts.append((origin + a * u + b * v, label))

    o = np.zeros(3)
    ex, ey, ez = np.eye(3)
    plane(o, lx * ex, ly * ey, 0)                     # sol
    plane(lz * ez, lx * ex, ly * ey, 1)               # plafond
    plane(o, lx * ex, lz * ez, 2)                     # murs
    plane(ly * ey, lx * ex, lz * ez, 2)
    plane(o, ly * ey, lz * ez, 2)
    plane(lx * ex, ly * ey, lz * ez, 2)
    plane(np.array([1.5, 1.5, 0.75]), 2.0 * ex, 1.0 * ey, 3)  # plateau de table

    coord = np.concatenate([p for p, _ in parts])
    segment = np.concatenate([np.full(len(p), lab) for p, lab in parts])
    return {"coord": coord, "color": CLASS_COLORS[segment] * 255, "segment": segment}


def to_o3d(scene, offset=0.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene["coord"] + [offset, 0, 0])
    seg = scene["segment"]
    colors = np.where((seg == IGNORE_INDEX)[:, None], OUTLIER_COLOR, CLASS_COLORS[np.clip(seg, 0, None)])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corruption", choices=sorted(CORRUPTIONS), default="gaussian_noise")
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n_levels = len(SEVERITIES[args.corruption])
    if not 1 <= args.severity <= n_levels:
        parser.error(f"--severity doit être entre 1 et {n_levels} pour {args.corruption}")

    scene = synthetic_room()
    corrupted = corrupt(scene, args.corruption, args.severity, seed=args.seed)
    print(f"{args.corruption} niveau {args.severity} : {len(scene['coord'])} -> {len(corrupted['coord'])} points")
    o3d.visualization.draw_geometries([to_o3d(scene), to_o3d(corrupted, offset=6.0)])


if __name__ == "__main__":
    main()
