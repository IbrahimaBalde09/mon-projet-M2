"""Métriques de segmentation sémantique : OA, mAcc, IoU par classe et mIoU.

Tout repose sur une matrice de confusion accumulée sur l'ensemble du jeu de
test (et non une moyenne de mIoU par scène), comme dans les protocoles
standards de S3DIS et ScanNet.
"""

import numpy as np


def confusion_matrix(pred, target, num_classes, ignore_index=-1):
    """Matrice de confusion C, où C[i, j] = nb de points de classe i prédits j."""
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    if pred.shape != target.shape:
        raise ValueError(f"tailles différentes : pred {pred.shape}, target {target.shape}")
    keep = target != ignore_index
    pred, target = pred[keep], target[keep]
    if target.size and (target.min() < 0 or target.max() >= num_classes):
        raise ValueError("étiquette de vérité terrain hors de [0, num_classes)")
    if pred.size and (pred.min() < 0 or pred.max() >= num_classes):
        raise ValueError("prédiction hors de [0, num_classes)")
    flat = target.astype(np.int64) * num_classes + pred.astype(np.int64)
    return np.bincount(flat, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(cm):
    """Calcule OA, mAcc, IoU et mIoU à partir d'une matrice de confusion.

    Une classe absente de la vérité terrain et jamais prédite a un IoU
    indéfini (NaN) et n'entre pas dans les moyennes.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    gt_count = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    union = gt_count + pred_count - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(gt_count > 0, tp / gt_count, np.nan)
        iou = np.where(union > 0, tp / union, np.nan)

    total = cm.sum()
    return {
        "oa": tp.sum() / total if total > 0 else np.nan,
        "macc": np.nanmean(acc) if np.any(~np.isnan(acc)) else np.nan,
        "miou": np.nanmean(iou) if np.any(~np.isnan(iou)) else np.nan,
        "acc": acc,
        "iou": iou,
    }


class SegmentationMetrics:
    """Accumule les prédictions scène par scène, puis calcule les métriques."""

    def __init__(self, num_classes, ignore_index=-1, class_names=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target):
        self.cm += confusion_matrix(pred, target, self.num_classes, self.ignore_index)

    def compute(self):
        return metrics_from_confusion(self.cm)

    def summary(self):
        """Résumé texte : métriques globales puis IoU par classe (en %)."""
        m = self.compute()
        lines = [f"OA {100 * m['oa']:.2f} | mAcc {100 * m['macc']:.2f} | mIoU {100 * m['miou']:.2f}"]
        for name, iou in zip(self.class_names, m["iou"]):
            lines.append(f"  {name:<12} IoU {100 * iou:6.2f}")
        return "\n".join(lines)
