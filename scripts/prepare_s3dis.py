"""Convertit S3DIS brut en scènes au format Pointcept (un dossier par pièce).

Exemple :
    python scripts/prepare_s3dis.py \
        --raw data/raw/s3dis/Stanford3dDataset_v1.2_Aligned_Version \
        --out data/processed/s3dis

Résultat : data/processed/s3dis/Area_1/office_1/{coord,color,segment,instance}.npy
Les pièces déjà converties sont sautées (relancer reprend où on s'est arrêté).
"""

import argparse
import time
from pathlib import Path

from segbench.datasets.s3dis import AREAS, iter_rooms, load_room, save_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True, help="dossier Stanford3dDataset_v1.2_Aligned_Version")
    parser.add_argument("--out", type=Path, required=True, help="dossier de sortie")
    parser.add_argument("--areas", nargs="+", default=AREAS, help="zones à convertir (défaut : toutes)")
    parser.add_argument("--overwrite", action="store_true", help="reconvertir les pièces existantes")
    args = parser.parse_args()

    start = time.time()
    n_done = n_skipped = 0
    for area, room, room_dir in iter_rooms(args.raw, args.areas):
        out_dir = args.out / area / room
        if (out_dir / "segment.npy").exists() and not args.overwrite:
            n_skipped += 1
            continue
        scene = load_room(room_dir)
        save_scene(scene, out_dir)
        n_done += 1
        print(f"{area}/{room} : {len(scene['coord']):,} points")

    print(f"\n{n_done} pièces converties, {n_skipped} déjà présentes, en {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
