"""Évalue un modèle entraîné sur l'Area 5 propre puis perturbée.

Exemple :
    python scripts/eval_robustness.py --data data/processed/s3dis \
        --checkpoint outputs/pointnet2/best.pth --name pointnet2 \
        --out outputs/robustesse/pointnet2.csv

    # sous-ensemble pour un essai rapide
    python scripts/eval_robustness.py ... --corruptions gaussian_noise --severities 1 3 5 --max-test-rooms 2

Le CSV contient une ligne par (perturbation, niveau, graine) avec OA, mAcc,
mIoU et l'IoU de chaque classe. Les CSV de plusieurs modèles se comparent
ensuite avec plot_robustness.py.
"""

import argparse
import csv
import time
from pathlib import Path

import torch

from segbench.corruptions import SEVERITIES
from segbench.datasets.blocks import predict_room
from segbench.datasets.s3dis import CLASSES, load_scene
from segbench.models.pointnet2 import load_pointnet2
from segbench.robustness import conditions, evaluate_condition, relative_robustness


def pointnet2_predictor(checkpoint, args):
    model, use_color = load_pointnet2(checkpoint, len(CLASSES), args.device)

    def predict(scene):
        return predict_room(model, scene, len(CLASSES), num_point=args.num_point, block_size=args.block_size,
                            stride=args.stride, batch_size=args.batch_size, device=args.device,
                            use_color=use_color)

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True, help="nom du modèle dans le CSV (ex. pointnet2)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-area", default="Area_5")
    parser.add_argument("--corruptions", nargs="+", choices=list(SEVERITIES), default=None)
    parser.add_argument("--severities", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-test-rooms", type=int, default=None)
    parser.add_argument("--num-point", type=int, default=4096)
    parser.add_argument("--block-size", type=float, default=1.0)
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    room_dirs = sorted(d for d in (args.data / args.test_area).glob("*") if (d / "segment.npy").exists())
    room_dirs = room_dirs[: args.max_test_rooms]
    if not room_dirs:
        parser.error(f"aucune pièce dans {args.data / args.test_area}")
    scenes = [load_scene(d) for d in room_dirs]
    predict = pointnet2_predictor(args.checkpoint, args)

    todo = conditions(args.corruptions, args.severities)
    print(f"{len(scenes)} pièces, {len(todo)} conditions x {len(args.seeds)} graine(s)")
    rows = []
    for seed in args.seeds:
        for corruption, severity in todo:
            start = time.time()
            row = {"model": args.name, **evaluate_condition(predict, scenes, len(CLASSES), corruption, severity,
                                                             seed, class_names=CLASSES)}
            rows.append(row)
            print(f"  {corruption:<17} niveau {severity}  graine {seed}  mIoU {100 * row['miou']:5.1f}  "
                  f"({100 * row['point_ratio']:.0f} % des points, {time.time() - start:.0f} s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n-> {args.out}\nRobustesse relative (mIoU perturbé / mIoU propre) :")
    for name, value in relative_robustness(rows).items():
        print(f"  {name:<17} {100 * value:5.1f} %")


if __name__ == "__main__":
    main()
