"""Lecture de S3DIS (Stanford3dDataset_v1.2_Aligned_Version).

Organisation des fichiers bruts :
    Area_1/
      office_1/
        office_1.txt              nuage complet (non utilisé ici)
        Annotations/
          chair_1.txt             un fichier par objet : lignes "x y z r g b"
          wall_3.txt
          ...

Chaque pièce est convertie en une scène au format Pointcept :
    coord    (N, 3) float32   coordonnées en mètres
    color    (N, 3) uint8     couleurs RGB
    segment  (N,)   int16     classe sémantique (0..12)
    instance (N,)   int32     identifiant de l'objet dans la pièce
"""

import re
from pathlib import Path

import numpy as np

CLASSES = [
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "table", "chair", "sofa", "bookcase", "board", "clutter",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

# Classes présentes dans les annotations mais hors des 13 classes officielles.
CLASS_ALIASES = {"stairs": "clutter"}

AREAS = [f"Area_{i}" for i in range(1, 7)]
TEST_AREA = "Area_5"

# Certains fichiers bruts contiennent des caractères parasites (par exemple
# dans Area_5/hallway_6) : on ne garde que ce qui peut former un nombre.
_NOT_NUMERIC = re.compile(rb"[^0-9eE+\-.\s]")


def read_points(path):
    """Lit un fichier "x y z r g b" et renvoie un tableau (N, 6) float64."""
    raw = _NOT_NUMERIC.sub(b"", Path(path).read_bytes())
    values = np.array(raw.split(), dtype=np.float64)
    if values.size % 6:
        raise ValueError(f"{path} : {values.size} valeurs, pas un multiple de 6")
    return values.reshape(-1, 6)


def class_of(annotation_file):
    """Classe sémantique d'un fichier d'annotation, d'après son nom (chair_1.txt -> chair)."""
    name = Path(annotation_file).stem.rsplit("_", 1)[0]
    name = CLASS_ALIASES.get(name, name)
    if name not in CLASS_TO_ID:
        raise ValueError(f"classe inconnue : {annotation_file}")
    return name


def room_type(room_name):
    """Type de pièce, d'après son nom (conferenceRoom_2 -> conferenceRoom)."""
    return room_name.rsplit("_", 1)[0]


def load_room(room_dir):
    """Assemble les fichiers d'annotation d'une pièce en une scène."""
    files = sorted(Path(room_dir, "Annotations").glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"aucune annotation dans {room_dir}")
    coords, colors, segments, instances = [], [], [], []
    for instance_id, f in enumerate(files):
        pts = read_points(f)
        coords.append(pts[:, :3])
        colors.append(pts[:, 3:6])
        segments.append(np.full(len(pts), CLASS_TO_ID[class_of(f)], dtype=np.int16))
        instances.append(np.full(len(pts), instance_id, dtype=np.int32))
    return {
        "coord": np.concatenate(coords).astype(np.float32),
        "color": np.clip(np.concatenate(colors), 0, 255).astype(np.uint8),
        "segment": np.concatenate(segments),
        "instance": np.concatenate(instances),
    }


def iter_rooms(root, areas=AREAS):
    """Parcourt les pièces : renvoie (area, nom de la pièce, dossier)."""
    for area in areas:
        area_dir = Path(root, area)
        if not area_dir.is_dir():
            raise FileNotFoundError(f"zone absente : {area_dir}")
        for room_dir in sorted(p for p in area_dir.iterdir() if (p / "Annotations").is_dir()):
            yield area, room_dir.name, room_dir


def save_scene(scene, out_dir):
    """Enregistre une scène : un fichier .npy par clé (coord.npy, color.npy...)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, value in scene.items():
        np.save(out_dir / f"{key}.npy", value)


def load_scene(scene_dir):
    """Recharge une scène enregistrée par ``save_scene``."""
    return {f.stem: np.load(f) for f in sorted(Path(scene_dir).glob("*.npy"))}
