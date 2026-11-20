"""Trace la mIoU en fonction du niveau de perturbation, pour un ou plusieurs modèles.

Exemple :
    python scripts/plot_robustness.py outputs/robustesse/*.csv --out outputs/robustesse/courbes.png

Une case par perturbation ; une courbe par modèle (moyenne sur les graines,
barres d'erreur = écart-type s'il y a plusieurs graines). La ligne pointillée
rappelle la mIoU sur la scène propre.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TITLES = {
    "random_subsample": "Densité (tirage aléatoire)",
    "voxel_subsample": "Densité (voxels)",
    "gaussian_noise": "Bruit gaussien",
    "add_outliers": "Points aberrants",
    "local_dropout": "Occlusions",
    "drop_color": "Sans couleur",
}


def read_rows(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["severity"] = int(r["severity"])
                r["miou"] = 100 * float(r["miou"])
                rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    models = list(dict.fromkeys(r["model"] for r in rows))
    corruptions = list(dict.fromkeys(r["corruption"] for r in rows if r["corruption"] != "clean"))
    if not corruptions:
        parser.error("aucune perturbation dans les CSV")

    values = defaultdict(list)  # (modèle, perturbation, niveau) -> mIoU par graine
    for r in rows:
        values[r["model"], r["corruption"], r["severity"]].append(r["miou"])

    cols = min(3, len(corruptions))
    nrows = math.ceil(len(corruptions) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4.2 * cols, 3.4 * nrows), squeeze=False, sharey=True)
    colors = plt.cm.tab10.colors
    for ax, corruption in zip(axes.flat, corruptions):
        all_levels = sorted({s for (_, c, s) in values if c == corruption})
        for k, model in enumerate(models):
            levels = sorted(s for (m, c, s) in values if m == model and c == corruption)
            if not levels:
                continue
            mean = [np.mean(values[model, corruption, s]) for s in levels]
            std = [np.std(values[model, corruption, s]) for s in levels]
            ax.errorbar(levels, mean, yerr=std, marker="o", capsize=3, color=colors[k % 10], label=model)
            if values[model, "clean", 0]:
                ax.axhline(np.mean(values[model, "clean", 0]), ls="--", lw=1, color=colors[k % 10], alpha=0.6)
        ax.set_title(TITLES.get(corruption, corruption))
        ax.set_xlabel("niveau de sévérité")
        ax.set_xticks(all_levels)
        ax.set_ylim(0, 103)  # même échelle partout ; marge pour voir une courbe à 100 %
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("mIoU (%)")
    for ax in list(axes.flat)[len(corruptions):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
