"""Découpage des pièces en blocs, pour PointNet++.

Une pièce S3DIS contient de l'ordre du million de points : trop pour un
réseau en une passe. Comme dans l'article PointNet, on travaille sur des
colonnes verticales de 1 m x 1 m (toute la hauteur de la pièce), dont on tire
un nombre fixe de points (4096).

Entrées du réseau pour chaque point :
    xyz      x, y centrés sur le bloc ; z depuis le sol de la pièce
    features couleur RGB / 255 (sauf si use_color=False, pour les LiDAR
             sans couleur) et position normalisée dans la pièce (0..1)

Entraînement : blocs tirés au hasard (``RandomBlocks``).
Test : grille glissante couvrant toute la pièce, chaque point reçoit la
somme des scores de tous les blocs qui le contiennent (``predict_room``).
"""

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from segbench.datasets.s3dis import load_scene


def num_features(use_color):
    """Nombre de caractéristiques par point en entrée du réseau."""
    return 6 if use_color else 3


def block_inputs(coord, color, room_min, room_size, center):
    """Construit (xyz, features) pour les points d'un bloc ; ``color=None`` : sans couleur."""
    xyz = coord - room_min
    local = xyz.copy()
    local[:, :2] -= center[:2] - room_min[:2]
    features = [xyz / np.maximum(room_size, 1e-6)]
    if color is not None:
        features.insert(0, color / 255.0)
    return local.astype(np.float32), np.concatenate(features, axis=1).astype(np.float32)


def _in_block(coord, center, block_size):
    # Petite marge : sans elle, un point pile sur le bord de la pièce peut
    # échapper à tous les blocs à cause des arrondis float32 / float64.
    half = block_size / 2 + 1e-4
    return np.flatnonzero((np.abs(coord[:, 0] - center[0]) <= half) & (np.abs(coord[:, 1] - center[1]) <= half))


def _augment(xyz, rng):
    """Rotation autour de z, symétrie et changement d'échelle aléatoires."""
    theta = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    xyz = xyz @ rot.T
    if rng.random() < 0.5:
        xyz[:, 0] = -xyz[:, 0]
    return xyz * rng.uniform(0.9, 1.1)


class RandomBlocks(Dataset):
    """Blocs tirés au hasard dans des pièces préparées (voir prepare_s3dis.py).

    Les pièces sont choisies proportionnellement à leur nombre de points,
    pour que chaque zone de la scène ait la même chance d'être vue.
    ``samples_per_epoch`` fixe la longueur d'une époque.
    """

    def __init__(self, room_dirs, num_point=4096, block_size=1.0, samples_per_epoch=5000,
                 min_points=1024, augment=True, use_color=True):
        self.rooms = [load_scene(d) for d in room_dirs]
        if not self.rooms:
            raise ValueError("aucune pièce fournie")
        for r in self.rooms:
            r["min"] = r["coord"].min(axis=0)
            r["size"] = r["coord"].max(axis=0) - r["min"]
        sizes = np.array([len(r["coord"]) for r in self.rooms], dtype=np.float64)
        self.room_prob = sizes / sizes.sum()
        self.num_point, self.block_size = num_point, block_size
        self.samples_per_epoch, self.min_points = samples_per_epoch, min_points
        self.augment = augment
        self.use_color = use_color

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, i):
        # Graine tirée du générateur de PyTorch : reproductible avec
        # torch.manual_seed, et différente à chaque époque et dans chaque worker.
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        room = self.rooms[rng.choice(len(self.rooms), p=self.room_prob)]
        coord = room["coord"]
        for _ in range(10):  # on évite les blocs presque vides
            center = coord[rng.integers(len(coord))]
            idx = _in_block(coord, center, self.block_size)
            if len(idx) >= self.min_points:
                break
        idx = rng.choice(idx, self.num_point, replace=len(idx) < self.num_point)
        color = room["color"][idx] if self.use_color else None
        xyz, features = block_inputs(coord[idx], color, room["min"], room["size"], center)
        if self.augment:
            xyz = _augment(xyz, rng).astype(np.float32)
        return (torch.from_numpy(xyz), torch.from_numpy(features.T.copy()),
                torch.from_numpy(room["segment"][idx].astype(np.int64)))


def _grid_starts(lo, hi, block_size, stride):
    n = max(0, math.ceil((hi - lo - block_size) / stride)) + 1
    return lo + block_size / 2 + stride * np.arange(n)


@torch.no_grad()
def predict_room(model, scene, num_classes, num_point=4096, block_size=1.0, stride=0.5,
                 batch_size=16, device="cpu", seed=0, use_color=True):
    """Prédit une classe pour chaque point d'une pièce (grille glissante + vote)."""
    model.eval()
    rng = np.random.default_rng(seed)
    coord = scene["coord"]
    color = scene.get("color") if use_color else None
    if use_color and color is None:
        raise ValueError("le modèle attend des couleurs, mais la scène n'en a pas (entraîner avec --no-color)")
    room_min = coord.min(axis=0)
    room_size = coord.max(axis=0) - room_min
    scores = torch.zeros(len(coord), num_classes)

    chunks = []  # (indices des points, xyz, features)
    for cx in _grid_starts(room_min[0], room_min[0] + room_size[0], block_size, stride):
        for cy in _grid_starts(room_min[1], room_min[1] + room_size[1], block_size, stride):
            center = np.array([cx, cy, 0.0])
            idx = _in_block(coord, center, block_size)
            if len(idx) == 0:
                continue
            idx = rng.permutation(idx)
            n_chunks = math.ceil(len(idx) / num_point)
            pad = n_chunks * num_point - len(idx)
            idx = np.concatenate([idx, rng.choice(idx, pad, replace=True)])
            for chunk in idx.reshape(n_chunks, num_point):
                xyz, features = block_inputs(coord[chunk], None if color is None else color[chunk],
                                             room_min, room_size, center)
                chunks.append((chunk, xyz, features))

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        xyz = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
        features = torch.from_numpy(np.stack([b[2].T for b in batch])).to(device)
        probs = model(xyz, features).softmax(dim=1).transpose(1, 2).cpu()  # (B, N, K)
        for (chunk, _, _), p in zip(batch, probs):
            scores.index_add_(0, torch.from_numpy(chunk), p)

    missing = int((scores.sum(dim=1) == 0).sum())
    if missing:
        raise RuntimeError(f"{missing} points sans prédiction")
    return scores.argmax(dim=1).numpy()
