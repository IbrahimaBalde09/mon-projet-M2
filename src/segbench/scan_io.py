"""Lecture de scans LiDAR dans les formats d'export courants.

    .ply .pcd .xyz .xyzrgb .pts   via Open3D
    .las .laz                     via laspy  (pip install laspy[lazrs] pour .laz)
    .e57                          via pye57  (pip install pye57)

Renvoie un dictionnaire :
    coord      (N, 3) float32  coordonnées en mètres, z vers le haut
    color      (N, 3) uint8 ou None
    intensity  (N,)   float32 ou None (réflectance laser, .las et .e57)
    meta       dict : format, nombre de scans, et les informations sur le
               scanner trouvées dans le fichier (marque, modèle, logiciel)
"""

from pathlib import Path

import numpy as np


def _to_uint8(color):
    color = np.asarray(color, dtype=np.float64)
    if color.size and color.max() > 255:  # couleurs 16 bits (fréquent en .las / .e57)
        color = color / 257.0
    elif color.size and color.max() <= 1.0:  # couleurs entre 0 et 1
        color = color * 255.0
    return np.clip(color, 0, 255).astype(np.uint8)


def _read_open3d(path):
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    coord = np.asarray(pcd.points)
    color = _to_uint8(np.asarray(pcd.colors)) if pcd.has_colors() else None
    return coord, color, None, {}


def _read_las(path):
    import laspy

    las = laspy.read(str(path))
    coord = np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1)
    dims = set(las.point_format.dimension_names)
    color = _to_uint8(np.stack([las.red, las.green, las.blue], axis=1)) if {"red", "green", "blue"} <= dims else None
    intensity = np.asarray(las.intensity, dtype=np.float32) if "intensity" in dims else None
    meta = {
        "systeme": str(las.header.system_identifier).strip("\x00 ") or None,
        "logiciel": str(las.header.generating_software).strip("\x00 ") or None,
        "version_las": str(las.header.version),
    }
    return coord, color, intensity, meta


def _read_e57(path):
    import pye57

    e57 = pye57.E57(str(path))
    coords, colors, intensities, meta = [], [], [], {"scans": e57.scan_count}
    for i in range(e57.scan_count):
        data = e57.read_scan(i, colors=True, intensity=True, ignore_missing_fields=True, transform=True)
        coords.append(np.stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]], axis=1))
        if "colorRed" in data:
            colors.append(np.stack([data["colorRed"], data["colorGreen"], data["colorBlue"]], axis=1))
        if "intensity" in data:
            intensities.append(np.asarray(data["intensity"], dtype=np.float32))
        if i == 0:  # informations sur le scanner, si le logiciel d'export les a écrites
            header = e57.get_header(0)
            for key, label in (("sensorVendor", "marque"), ("sensorModel", "modele"),
                               ("sensorSerialNumber", "numero_serie"), ("sensorSoftwareVersion", "logiciel")):
                try:
                    meta[label] = header.node[key].value()
                except Exception:  # champ absent : on n'invente rien
                    pass
    color = _to_uint8(np.concatenate(colors)) if len(colors) == e57.scan_count and colors else None
    intensity = np.concatenate(intensities) if len(intensities) == e57.scan_count and intensities else None
    return np.concatenate(coords), color, intensity, meta


READERS = {
    ".ply": _read_open3d, ".pcd": _read_open3d, ".xyz": _read_open3d, ".xyzrgb": _read_open3d, ".pts": _read_open3d,
    ".las": _read_las, ".laz": _read_las, ".e57": _read_e57,
}


def read_scan(path, scale=1.0, up="z"):
    """Lit un scan ; ``scale`` convertit en mètres, ``up`` indique l'axe vertical du fichier."""
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"format non pris en charge : {path.suffix} (formats : {', '.join(sorted(READERS))})")
    try:
        coord, color, intensity, meta = reader(path)
    except ImportError as e:
        raise ImportError(f"lecture de {path.suffix} impossible : installer {e.name} (voir README)") from e
    if len(coord) == 0:
        raise ValueError(f"aucun point lu dans {path}")
    coord = np.asarray(coord, dtype=np.float64) * scale
    if up == "y":  # (x, y, z) avec y vers le haut -> (x, -z, y) avec z vers le haut
        coord = np.stack([coord[:, 0], -coord[:, 2], coord[:, 1]], axis=1)
    elif up != "z":
        raise ValueError("up doit valoir 'z' ou 'y'")
    meta = {"format": path.suffix.lower().lstrip("."), **{k: v for k, v in meta.items() if v not in (None, "")}}
    return {"coord": coord.astype(np.float32), "color": color, "intensity": intensity, "meta": meta}
