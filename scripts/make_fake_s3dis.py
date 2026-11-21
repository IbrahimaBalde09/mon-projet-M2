"""Génère un faux S3DIS synthétique, au même format que l'original.

Sert à tester toute la chaîne sur un PC sans les vraies données :
    python scripts/make_fake_s3dis.py --out data/raw/fake_s3dis
    python scripts/prepare_s3dis.py --raw data/raw/fake_s3dis --out data/processed/fake_s3dis
    python scripts/s3dis_stats.py --data data/processed/fake_s3dis --out outputs/fake/stats.csv
    python scripts/train_pointnet2.py --data data/processed/fake_s3dis --out outputs/fake/pointnet2 \
        --epochs 2 --samples-per-epoch 32 --batch-size 4 --eval-every 1 --max-test-rooms 1 --stride 1.0

Chaque pièce est une boîte (sol, plafond, 4 murs) meublée selon son type
(bureau, salle de réunion, couloir) : tables, chaises, porte, fenêtre,
tableau, bibliothèque, objets divers. Les résultats obtenus dessus n'ont
aucune valeur scientifique : ils servent seulement à vérifier que tout tourne.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

COLORS = {
    "ceiling": (230, 230, 225), "floor": (150, 120, 90), "wall": (200, 200, 190),
    "beam": (180, 180, 170), "column": (190, 185, 180), "window": (140, 180, 220),
    "door": (120, 80, 50), "table": (170, 130, 80), "chair": (40, 40, 45),
    "sofa": (120, 40, 40), "bookcase": (100, 70, 40), "board": (245, 245, 245),
    "clutter": (90, 140, 90),
}


def rect(rng, origin, u, v, density):
    """Points tirés uniformément sur le parallélogramme origin + a*u + b*v."""
    origin, u, v = (np.asarray(x, dtype=float) for x in (origin, u, v))
    area = np.linalg.norm(np.cross(u, v))
    n = max(10, int(area * density))
    a, b = rng.uniform(size=(2, n, 1))
    return origin + a * u + b * v


def box(rng, lo, hi, density):
    """Points sur les 6 faces d'une boîte alignée sur les axes."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    d = hi - lo
    ex, ey, ez = np.diag(d)
    faces = [(lo, ex, ey), (lo + ez, ex, ey), (lo, ex, ez), (lo + ey, ex, ez), (lo, ey, ez), (lo + ex, ey, ez)]
    return np.concatenate([rect(rng, o, u, v, density) for o, u, v in faces])


def make_room(rng, kind, density):
    """Renvoie {nom_classe: [nuage objet 1, nuage objet 2, ...]}."""
    if kind == "hallway":
        lx, ly = rng.uniform(8, 14), rng.uniform(1.8, 2.6)
    elif kind == "conferenceRoom":
        lx, ly = rng.uniform(6, 9), rng.uniform(4, 6)
    else:
        lx, ly = rng.uniform(4, 6), rng.uniform(3, 5)
    lz = rng.uniform(2.6, 3.2)
    objs = defaultdict(list)

    objs["floor"].append(rect(rng, (0, 0, 0), (lx, 0, 0), (0, ly, 0), density))
    objs["ceiling"].append(rect(rng, (0, 0, lz), (lx, 0, 0), (0, ly, 0), density))
    for origin, u in (((0, 0, 0), (lx, 0, 0)), ((0, ly, 0), (lx, 0, 0)), ((0, 0, 0), (0, ly, 0)), ((lx, 0, 0), (0, ly, 0))):
        objs["wall"].append(rect(rng, origin, u, (0, 0, lz), density))

    # porte et fenêtre : légèrement décollées du mur, comme dans S3DIS
    x = rng.uniform(0.3, lx - 1.3)
    objs["door"].append(rect(rng, (x, 0.02, 0), (0.9, 0, 0), (0, 0, 2.1), density))
    if kind != "hallway":
        x = rng.uniform(0.3, lx - 1.8)
        objs["window"].append(rect(rng, (x, ly - 0.02, 0.9), (1.5, 0, 0), (0, 0, 1.2), density))
        y = rng.uniform(0.3, ly - 2.3)
        objs["board"].append(rect(rng, (0.02, y, 0.9), (0, 2.0, 0), (0, 0, 1.2), density))

    def table_with_chairs(cx, cy, tx, ty, n_chairs):
        objs["table"].append(box(rng, (cx - tx / 2, cy - ty / 2, 0.72), (cx + tx / 2, cy + ty / 2, 0.76), density))
        for k in range(n_chairs):
            side = 1 if k % 2 == 0 else -1
            px = cx - tx / 2 + (k // 2 + 0.5) * tx / max(1, (n_chairs + 1) // 2)
            py = cy + side * (ty / 2 + 0.35)
            seat = box(rng, (px - 0.22, py - 0.22, 0.42), (px + 0.22, py + 0.22, 0.46), density * 3)
            back_y = py + side * 0.22
            back = box(rng, (px - 0.22, back_y - 0.02, 0.46), (px + 0.22, back_y + 0.02, 0.9), density * 3)
            objs["chair"].append(np.concatenate([seat, back]))
        for _ in range(rng.integers(0, 3)):  # objets posés sur la table
            ox, oy = cx + rng.uniform(-tx / 3, tx / 3), cy + rng.uniform(-ty / 3, ty / 3)
            objs["clutter"].append(box(rng, (ox, oy, 0.76), (ox + 0.25, oy + 0.2, 0.76 + rng.uniform(0.05, 0.3)), density * 3))

    if kind == "office":
        for k in range(rng.integers(1, 3)):
            table_with_chairs(lx * (0.35 + 0.3 * k), ly / 2, 1.4, 0.8, 2)
        objs["bookcase"].append(box(rng, (lx - 0.45, 0.4, 0), (lx - 0.05, 1.6, 2.0), density))
    elif kind == "conferenceRoom":
        table_with_chairs(lx / 2, ly / 2, lx * 0.55, 1.4, 8)
        if rng.random() < 0.5:
            objs["sofa"].append(box(rng, (0.5, 0.3, 0), (2.3, 1.1, 0.8), density))
    else:
        for _ in range(rng.integers(1, 4)):
            x = rng.uniform(0.5, lx - 0.5)
            objs["clutter"].append(box(rng, (x, 0.05, 0), (x + 0.4, 0.45, rng.uniform(0.3, 1.2)), density))
    return objs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rooms-per-area", type=int, default=3)
    parser.add_argument("--density", type=float, default=300, help="points par m² de surface")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    kinds = ["office", "conferenceRoom", "hallway"]
    total = 0
    for area in range(1, 7):
        counters = defaultdict(int)
        for r in range(args.rooms_per_area):
            kind = kinds[r % len(kinds)]
            counters[kind] += 1
            room_dir = args.out / f"Area_{area}" / f"{kind}_{counters[kind]}" / "Annotations"
            room_dir.mkdir(parents=True, exist_ok=True)
            for name, clouds in make_room(rng, kind, args.density).items():
                for i, xyz in enumerate(clouds, start=1):
                    rgb = np.clip(np.array(COLORS[name]) + rng.normal(0, 12, size=(len(xyz), 3)), 0, 255)
                    np.savetxt(room_dir / f"{name}_{i}.txt", np.hstack([xyz, rgb]), fmt="%.4f %.4f %.4f %d %d %d")
                    total += len(xyz)
    print(f"{6 * args.rooms_per_area} pièces, {total:,} points -> {args.out}")


if __name__ == "__main__":
    main()
