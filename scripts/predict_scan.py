"""Segmente un scan LiDAR quelconque avec un modèle PointNet++ entraîné.

Exemples :
    python scripts/predict_scan.py --checkpoint outputs/pointnet2/best.pth \
        --input mes_scans/bureau.ply --out outputs/scans/bureau_segmente.ply

    # scan en millimètres, exporté avec l'axe y vers le haut (fréquent sur iPhone)
    python scripts/predict_scan.py ... --scale 0.001 --up y

Formats lus (via Open3D) : .ply, .pcd, .xyz, .xyzrgb, .pts. Pour un .las,
.laz ou .e57, convertir d'abord en .ply (par exemple avec CloudCompare).

Précautions pour que le modèle s'y retrouve :
  - unités en mètres et axe vertical z (options --scale et --up) ;
  - de préférence une pièce par fichier, comme dans S3DIS ;
  - si le scanner ne capte pas la couleur, utiliser un modèle entraîné avec
    --no-color.

Les scans LiDAR sont souvent bien plus denses que S3DIS : on sous-échantillonne
par voxels (--voxel-size, 2 cm par défaut), on prédit, puis chaque point
d'origine reçoit la classe du point retenu dans son voxel.

Sorties : un .ply coloré par classe, un .npy des classes par point, et la
répartition des classes affichée à l'écran.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from segbench.datasets.blocks import predict_room
from segbench.datasets.s3dis import CLASSES
from segbench.models.pointnet2 import load_pointnet2

PALETTE = np.array([
    [230, 230, 225], [150, 120, 90], [200, 200, 190], [255, 200, 0], [255, 120, 0],
    [100, 180, 255], [160, 90, 40], [220, 60, 60], [60, 60, 220], [200, 0, 200],
    [0, 160, 120], [250, 250, 120], [120, 120, 120],
]) / 255.0


def read_scan(path, scale, up):
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise SystemExit(f"impossible de lire des points dans {path}")
    coord = np.asarray(pcd.points, dtype=np.float64) * scale
    if up == "y":  # (x, y, z) avec y vers le haut -> (x, -z, y) avec z vers le haut
        coord = np.stack([coord[:, 0], -coord[:, 2], coord[:, 1]], axis=1)
    color = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
    return coord.astype(np.float32), color


def voxel_representatives(coord, voxel_size):
    """Un point par voxel ; renvoie (indices retenus, voxel de chaque point d'origine)."""
    keys = np.floor(coord / voxel_size).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return first, inverse.ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="fichier .ply de sortie")
    parser.add_argument("--scale", type=float, default=1.0, help="facteur vers les mètres (0.001 si en mm)")
    parser.add_argument("--up", choices=["z", "y"], default="z", help="axe vertical du fichier")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="0 pour garder tous les points")
    parser.add_argument("--num-point", type=int, default=4096)
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--show", action="store_true", help="afficher le résultat dans une fenêtre Open3D")
    args = parser.parse_args()

    coord, color = read_scan(args.input, args.scale, args.up)
    extent = coord.max(axis=0) - coord.min(axis=0)
    print(f"{len(coord):,} points, couleur : {'oui' if color is not None else 'non'}, "
          f"dimensions {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} m")
    if extent[2] > 10 or extent.max() > 100:
        print("Attention : dimensions inhabituelles pour une pièce. Vérifier --scale et --up.")

    model, use_color = load_pointnet2(args.checkpoint, len(CLASSES), args.device)
    if use_color and color is None:
        raise SystemExit("Ce scan n'a pas de couleur mais le modèle en attend : entraîner avec --no-color.")

    if args.voxel_size > 0:
        kept, inverse = voxel_representatives(coord, args.voxel_size)
    else:
        kept, inverse = np.arange(len(coord)), np.arange(len(coord))
    scene = {"coord": coord[kept], "color": None if color is None else color[kept]}
    print(f"{len(kept):,} points après sous-échantillonnage, prédiction sur {args.device}...")
    pred = predict_room(model, scene, len(CLASSES), num_point=args.num_point, stride=args.stride,
                        batch_size=args.batch_size, device=args.device, use_color=use_color)[inverse]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coord.astype(np.float64)))
    out.colors = o3d.utility.Vector3dVector(PALETTE[pred])
    o3d.io.write_point_cloud(str(args.out), out)
    np.save(args.out.with_suffix(".npy"), pred.astype(np.int16))

    print(f"-> {args.out} (+ {args.out.with_suffix('.npy').name})\nRépartition des classes prédites :")
    counts = np.bincount(pred, minlength=len(CLASSES))
    for i in np.argsort(-counts):
        if counts[i]:
            print(f"  {CLASSES[i]:<10} {100 * counts[i] / len(pred):5.1f} %")
    if args.show:
        o3d.visualization.draw_geometries([out], window_name="Segmentation")


if __name__ == "__main__":
    main()
