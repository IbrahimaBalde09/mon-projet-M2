"""Statistiques descriptives des scènes, pour l'axe « complexité des scènes ».

Indicateurs par pièce :
    n_points, extent_*     taille du nuage et de la pièce
    area_m2, density_m2    surface au sol (boîte englobante) et points par m²
    n_objects              nombre d'objets annotés (hors structure : sol, murs...)
    n_classes              nombre de classes présentes
    class_entropy          entropie de la répartition des points par classe (bits)
    frac_<classe>          part des points de chaque classe
"""

import numpy as np

# Éléments de structure du bâtiment : ils ne comptent pas comme des « objets ».
STRUCTURE_CLASSES = ("ceiling", "floor", "wall", "beam", "column")


def room_stats(scene, class_names):
    coord, segment = scene["coord"], scene["segment"]
    extent = coord.max(axis=0) - coord.min(axis=0)
    area = float(extent[0] * extent[1])

    counts = np.bincount(segment, minlength=len(class_names))
    frac = counts / counts.sum()
    nonzero = frac[frac > 0]

    stats = {
        "n_points": int(len(coord)),
        "extent_x": float(extent[0]),
        "extent_y": float(extent[1]),
        "extent_z": float(extent[2]),
        "area_m2": area,
        "density_m2": len(coord) / area if area > 0 else float("nan"),
        "n_classes": int((counts > 0).sum()),
        "class_entropy": float(-(nonzero * np.log2(nonzero)).sum()),
    }

    if "instance" in scene:
        structure = [class_names.index(c) for c in STRUCTURE_CLASSES if c in class_names]
        is_object = ~np.isin(segment, structure)
        stats["n_objects"] = int(len(np.unique(scene["instance"][is_object])))

    for name, f in zip(class_names, frac):
        stats[f"frac_{name}"] = float(f)
    return stats
