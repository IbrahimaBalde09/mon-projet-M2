"""Banc d'évaluation de robustesse : mêmes scènes, mêmes perturbations, pour tous les modèles.

Un modèle est vu comme une fonction ``predict(scene) -> classes par point``.
On peut ainsi comparer PointNet++ (code maison) et les modèles de Pointcept
avec exactement le même protocole.

Protocole :
  - chaque pièce de test est perturbée avant d'être découpée en blocs ;
  - la graine dépend de (graine, numéro de pièce) : toutes les méthodes
    voient exactement les mêmes nuages perturbés ;
  - les métriques sont calculées sur les points de la scène perturbée ; les
    points ajoutés (outliers) n'ont pas de vérité terrain et sont ignorés.

Indicateur de synthèse, inspiré de Robo3D : la robustesse relative
mIoU(perturbé) / mIoU(propre), moyennée sur les niveaux de sévérité.
"""

import numpy as np

from segbench.corruptions import IGNORE_INDEX, SEVERITIES, corrupt
from segbench.metrics import SegmentationMetrics

CLEAN = "clean"


def evaluate_condition(predict, scenes, num_classes, corruption=CLEAN, severity=0, seed=0, class_names=None):
    """Évalue ``predict`` sur toutes les scènes, pour une perturbation et un niveau."""
    metrics = SegmentationMetrics(num_classes, ignore_index=IGNORE_INDEX, class_names=class_names)
    n_after = 0
    for i, scene in enumerate(scenes):
        if corruption != CLEAN:
            scene = corrupt(scene, corruption, severity, seed=seed * 100_003 + i)
        n_after += len(scene["coord"])
        metrics.update(predict(scene), scene["segment"])
    n_before = sum(len(s["coord"]) for s in scenes)
    m = metrics.compute()
    row = {
        "corruption": corruption,
        "severity": severity,
        "seed": seed,
        "oa": m["oa"],
        "macc": m["macc"],
        "miou": m["miou"],
        "point_ratio": n_after / n_before,
    }
    names = class_names or [str(c) for c in range(num_classes)]
    row.update({f"iou_{n}": v for n, v in zip(names, m["iou"])})
    return row


def conditions(corruptions=None, severities=None):
    """Liste des conditions (perturbation, niveau), en commençant par la scène propre."""
    out = [(CLEAN, 0)]
    for name in corruptions or SEVERITIES:
        levels = range(1, len(SEVERITIES[name]) + 1)
        out += [(name, s) for s in levels if severities is None or s in severities]
    return out


def relative_robustness(rows):
    """mIoU perturbé / mIoU propre, moyenné par perturbation (et sur les graines).

    ``rows`` : lignes produites par ``evaluate_condition`` pour un même modèle.
    """
    clean = np.mean([r["miou"] for r in rows if r["corruption"] == CLEAN])
    out = {}
    for name in dict.fromkeys(r["corruption"] for r in rows if r["corruption"] != CLEAN):
        corrupted = np.mean([r["miou"] for r in rows if r["corruption"] == name])
        out[name] = float(corrupted / clean) if clean > 0 else float("nan")
    return out
