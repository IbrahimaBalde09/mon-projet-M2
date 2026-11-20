"""Entraîne et évalue PointNet++ sur S3DIS préparé (voir prepare_s3dis.py).

Exemples :
    # entraînement complet (GPU recommandé)
    python scripts/train_pointnet2.py --data data/processed/s3dis --out outputs/pointnet2

    # essai rapide sur CPU
    python scripts/train_pointnet2.py --data data/processed/s3dis --out outputs/essai \
        --epochs 1 --samples-per-epoch 64 --batch-size 4 --max-test-rooms 2

Protocole : entraînement sur toutes les zones sauf --test-area (Area_5 par
défaut), évaluation sur --test-area toutes les --eval-every époques.
Sorties dans --out : config.json, log.csv, best.pth (meilleur mIoU), last.pth.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from segbench.datasets.blocks import RandomBlocks, predict_room
from segbench.datasets.s3dis import CLASSES, load_scene
from segbench.metrics import SegmentationMetrics
from segbench.models.pointnet2 import PointNet2SemSeg


def class_weights(room_dirs, num_classes):
    """Poids (max_freq / freq)^(1/3) : favorise un peu les classes rares."""
    counts = np.zeros(num_classes)
    for d in room_dirs:
        counts += np.bincount(np.load(d / "segment.npy"), minlength=num_classes)
    freq = counts / counts.sum()
    with np.errstate(divide="ignore"):
        w = np.where(freq > 0, (freq.max() / freq) ** (1 / 3), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, weights, device):
    model.train()
    total_loss = correct = seen = 0
    for xyz, features, target in loader:
        xyz, features, target = xyz.to(device), features.to(device), target.to(device)
        logits = model(xyz, features)
        loss = F.cross_entropy(logits, target, weight=weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(target)
        correct += (logits.argmax(1) == target).sum().item()
        seen += target.numel()
    return total_loss / len(loader.dataset), correct / seen


def evaluate(model, room_dirs, args, device):
    metrics = SegmentationMetrics(len(CLASSES), class_names=CLASSES)
    for d in room_dirs:
        scene = load_scene(d)
        pred = predict_room(model, scene, len(CLASSES), num_point=args.num_point, block_size=args.block_size,
                            stride=args.stride, batch_size=args.batch_size, device=device)
        metrics.update(pred, scene["segment"])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-area", default="Area_5")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-point", type=int, default=4096)
    parser.add_argument("--block-size", type=float, default=1.0)
    parser.add_argument("--stride", type=float, default=0.5, help="pas de la grille de test")
    parser.add_argument("--samples-per-epoch", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--max-test-rooms", type=int, default=None, help="limiter le test (essais rapides)")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    rooms = sorted(d for d in args.data.glob("Area_*/*") if (d / "segment.npy").exists())
    train_rooms = [d for d in rooms if d.parent.name != args.test_area]
    test_rooms = [d for d in rooms if d.parent.name == args.test_area][: args.max_test_rooms]
    if not train_rooms or not test_rooms:
        parser.error(f"pièces d'entraînement : {len(train_rooms)}, de test : {len(test_rooms)} (vérifier --data)")
    print(f"{len(train_rooms)} pièces d'entraînement, {len(test_rooms)} de test ({args.test_area}), device={args.device}")

    dataset = RandomBlocks(train_rooms, num_point=args.num_point, block_size=args.block_size,
                           samples_per_epoch=args.samples_per_epoch)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
    if len(loader) == 0:
        parser.error("--samples-per-epoch doit être >= --batch-size")

    model = PointNet2SemSeg(len(CLASSES)).to(args.device)
    print(f"{sum(p.numel() for p in model.parameters()):,} paramètres")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    weights = None if args.no_class_weights else class_weights(train_rooms, len(CLASSES)).to(args.device)

    log_path = args.out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "oa", "macc", "miou", "seconds"])

    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss, acc = train_one_epoch(model, loader, optimizer, weights, args.device)
        scheduler.step()
        row = [epoch, f"{loss:.4f}", f"{acc:.4f}", "", "", ""]
        print(f"époque {epoch}/{args.epochs} : perte {loss:.4f}, précision {100 * acc:.1f} %")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, test_rooms, args, args.device)
            m = metrics.compute()
            row[3:6] = [f"{m['oa']:.4f}", f"{m['macc']:.4f}", f"{m['miou']:.4f}"]
            print(metrics.summary())
            if m["miou"] > best_miou:
                best_miou = float(m["miou"])  # float Python : rechargeable avec weights_only=True
                torch.save({"model": model.state_dict(), "epoch": epoch, "miou": best_miou}, args.out / "best.pth")

        torch.save({"model": model.state_dict(), "epoch": epoch}, args.out / "last.pth")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row + [f"{time.time() - start:.0f}"])

    print(f"\nMeilleur mIoU sur {args.test_area} : {100 * best_miou:.2f} %")


if __name__ == "__main__":
    main()
