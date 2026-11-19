"""Statistiques descriptives de S3DIS préparé (voir prepare_s3dis.py).

Exemple :
    python scripts/s3dis_stats.py --data data/processed/s3dis --out outputs/s3dis_stats.csv

Écrit une ligne par pièce dans le CSV, puis affiche :
    - la répartition des pièces par zone et par type,
    - la répartition des points par classe (sur tout le jeu et sur l'Area 5),
    - la complexité moyenne par type de pièce.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from segbench.datasets.s3dis import CLASSES, TEST_AREA, load_scene, room_type
from segbench.stats import room_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="dossier produit par prepare_s3dis.py")
    parser.add_argument("--out", type=Path, default=Path("outputs/s3dis_stats.csv"))
    args = parser.parse_args()

    rows = []
    for room_dir in sorted(args.data.glob("Area_*/*")):
        if not (room_dir / "segment.npy").exists():
            continue
        area, room = room_dir.parent.name, room_dir.name
        stats = room_stats(load_scene(room_dir), CLASSES)
        rows.append({"area": area, "room": room, "room_type": room_type(room), **stats})
    if not rows:
        parser.error(f"aucune pièce trouvée dans {args.data}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} pièces -> {args.out}\n")

    print("Pièces par zone :", dict(Counter(r["area"] for r in rows)))
    print("Pièces par type :", dict(Counter(r["room_type"] for r in rows).most_common()), "\n")

    for label, subset in (("tout le jeu", rows), (TEST_AREA, [r for r in rows if r["area"] == TEST_AREA])):
        if not subset:
            continue
        points = np.array([[r[f"frac_{c}"] * r["n_points"] for c in CLASSES] for r in subset]).sum(axis=0)
        print(f"Répartition des points par classe ({label}) :")
        for name, p in sorted(zip(CLASSES, points / points.sum()), key=lambda x: -x[1]):
            print(f"  {name:<10} {100 * p:5.1f} %")
        print()

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["room_type"]].append(r)
    print(f"{'type':<16}{'pièces':>7}{'points':>12}{'surface':>9}{'objets':>8}{'classes':>9}{'entropie':>10}")
    for t, rs in sorted(by_type.items(), key=lambda x: -len(x[1])):
        mean = lambda k: np.mean([r[k] for r in rs])
        print(f"{t:<16}{len(rs):>7}{mean('n_points'):>12,.0f}{mean('area_m2'):>9.1f}"
              f"{mean('n_objects'):>8.1f}{mean('n_classes'):>9.1f}{mean('class_entropy'):>10.2f}")


if __name__ == "__main__":
    main()
