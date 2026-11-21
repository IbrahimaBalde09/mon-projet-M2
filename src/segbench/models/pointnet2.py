"""PointNet++ (version SSG) pour la segmentation sémantique, en PyTorch pur.

Référence : Qi et al., « PointNet++: Deep Hierarchical Feature Learning on
Point Sets in a Metric Space », NeurIPS 2017.

L'architecture suit l'encodeur-décodeur de l'article :
  - 4 couches de Set Abstraction (SA) : on choisit des centres par
    échantillonnage du point le plus lointain (FPS), on regroupe les voisins
    dans une boule (ball query), et un mini-PointNet partagé résume chaque
    groupe ;
  - 4 couches de Feature Propagation (FP) : on remonte vers les points
    d'origine par interpolation sur les 3 plus proches voisins, avec des
    connexions résiduelles (skip links) vers les couches de l'encodeur.

Tout est écrit en PyTorch standard (sans noyaux CUDA dédiés), pour que le
code soit lisible et tourne aussi sur CPU. C'est plus lent que les versions
optimisées, en particulier FPS, mais suffisant pour une baseline.

Conventions de forme : B = taille du lot, N = nombre de points,
C = nombre de canaux. Coordonnées (B, N, 3), caractéristiques (B, C, N).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Opérations géométriques
# ---------------------------------------------------------------------------

def square_distance(src, dst):
    """Distances au carré entre deux ensembles : (B, N, 3), (B, M, 3) -> (B, N, M)."""
    return torch.cdist(src, dst).pow(2)


def index_points(points, idx):
    """Sélectionne des points par indices : (B, N, C), (B, ...) -> (B, ..., C)."""
    batch = torch.arange(points.shape[0], device=points.device).view(-1, *([1] * (idx.dim() - 1)))
    return points[batch, idx]


def farthest_point_sample(xyz, npoint):
    """Échantillonnage du point le plus lointain (FPS) : (B, N, 3) -> indices (B, npoint).

    On part d'un point au hasard, puis on ajoute à chaque étape le point le
    plus éloigné de ceux déjà choisis. Les centres couvrent ainsi la scène
    de façon régulière, même là où la densité est faible.
    """
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), float("inf"), device=xyz.device)
    farthest = torch.randint(0, N, (B,), device=xyz.device)
    batch = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)
        distance = torch.minimum(distance, ((xyz - centroid) ** 2).sum(-1))
        farthest = distance.argmax(-1)
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Voisins de chaque centre dans une boule de rayon ``radius``.

    Renvoie (B, S, nsample) indices. S'il y a moins de ``nsample`` voisins,
    on répète le premier ; s'il y en a plus, on garde les premiers trouvés
    (comme l'implémentation d'origine).
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    sqrdists = square_distance(new_xyz, xyz)                                  # (B, S, N)
    idx = torch.arange(N, device=xyz.device).view(1, 1, N).expand(B, S, N).clone()
    idx[sqrdists > radius**2] = N
    idx = idx.sort(dim=-1)[0][:, :, :nsample]
    first = idx[:, :, :1].expand(-1, -1, nsample)
    # Le centre fait partie de xyz, donc chaque boule contient au moins un point.
    return torch.where(idx == N, first, idx)


# ---------------------------------------------------------------------------
# Couches
# ---------------------------------------------------------------------------

def _mlp(channels, conv):
    layers = []
    for c_in, c_out in zip(channels[:-1], channels[1:]):
        layers += [conv(c_in, c_out, 1), (nn.BatchNorm2d if conv is nn.Conv2d else nn.BatchNorm1d)(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class SetAbstraction(nn.Module):
    """Couche SA : FPS -> ball query -> mini-PointNet -> max-pooling par groupe."""

    def __init__(self, npoint, radius, nsample, in_channels, mlp):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        # +3 : coordonnées relatives au centre, concaténées aux caractéristiques
        self.mlp = _mlp([in_channels + 3] + mlp, nn.Conv2d)

    def forward(self, xyz, features):
        """xyz (B, N, 3), features (B, C, N) ou None -> new_xyz (B, S, 3), (B, C', S)."""
        new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)        # (B, S, K)
        grouped = index_points(xyz, idx) - new_xyz.unsqueeze(2)                 # (B, S, K, 3)
        if features is not None:
            grouped = torch.cat([grouped, index_points(features.transpose(1, 2), idx)], dim=-1)
        grouped = grouped.permute(0, 3, 2, 1)                                   # (B, C+3, K, S)
        return new_xyz, self.mlp(grouped).max(dim=2)[0]


class FeaturePropagation(nn.Module):
    """Couche FP : interpolation inverse à la distance sur les 3 plus proches voisins."""

    def __init__(self, in_channels, mlp):
        super().__init__()
        self.mlp = _mlp([in_channels] + mlp, nn.Conv1d)

    def forward(self, xyz_dense, xyz_sparse, feat_dense, feat_sparse):
        """Remonte feat_sparse (B, C2, S) sur xyz_dense (B, N, 3) -> (B, C', N)."""
        if xyz_sparse.shape[1] == 1:
            interpolated = feat_sparse.expand(-1, -1, xyz_dense.shape[1])
        else:
            dists, idx = square_distance(xyz_dense, xyz_sparse).topk(3, dim=-1, largest=False)
            weight = 1.0 / (dists + 1e-8)
            weight = weight / weight.sum(dim=-1, keepdim=True)                  # (B, N, 3)
            neighbors = index_points(feat_sparse.transpose(1, 2), idx)          # (B, N, 3, C2)
            interpolated = (neighbors * weight.unsqueeze(-1)).sum(dim=2).transpose(1, 2)
        if feat_dense is not None:
            interpolated = torch.cat([feat_dense, interpolated], dim=1)
        return self.mlp(interpolated)


# ---------------------------------------------------------------------------
# Réseau complet
# ---------------------------------------------------------------------------

class PointNet2SemSeg(nn.Module):
    """PointNet++ SSG pour la segmentation sémantique.

    Entrées : xyz (B, N, 3) coordonnées locales au bloc,
              features (B, C, N) caractéristiques par point (couleur...).
    Sortie : logits (B, num_classes, N).
    """

    def __init__(self, num_classes, in_channels=6, npoints=(1024, 256, 64, 16),
                 radii=(0.1, 0.2, 0.4, 0.8), nsample=32, dropout=0.5):
        super().__init__()
        self.sa1 = SetAbstraction(npoints[0], radii[0], nsample, in_channels, [32, 32, 64])
        self.sa2 = SetAbstraction(npoints[1], radii[1], nsample, 64, [64, 64, 128])
        self.sa3 = SetAbstraction(npoints[2], radii[2], nsample, 128, [128, 128, 256])
        self.sa4 = SetAbstraction(npoints[3], radii[3], nsample, 256, [256, 256, 512])
        self.fp4 = FeaturePropagation(512 + 256, [256, 256])
        self.fp3 = FeaturePropagation(256 + 128, [256, 256])
        self.fp2 = FeaturePropagation(256 + 64, [256, 128])
        self.fp1 = FeaturePropagation(128 + in_channels, [128, 128, 128])
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, xyz, features):
        l1_xyz, l1 = self.sa1(xyz, features)
        l2_xyz, l2 = self.sa2(l1_xyz, l1)
        l3_xyz, l3 = self.sa3(l2_xyz, l2)
        l4_xyz, l4 = self.sa4(l3_xyz, l3)
        l3 = self.fp4(l3_xyz, l4_xyz, l3, l4)
        l2 = self.fp3(l2_xyz, l3_xyz, l2, l3)
        l1 = self.fp2(l1_xyz, l2_xyz, l1, l2)
        l0 = self.fp1(xyz, l1_xyz, features, l1)
        return self.head(l0)


def load_pointnet2(checkpoint, num_classes, device="cpu"):
    """Recharge un modèle entraîné par train_pointnet2.py.

    Renvoie (modèle, use_color). Les checkpoints antérieurs à l'option
    --no-color n'ont pas la clé ``use_color`` : ils utilisaient la couleur.
    """
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    use_color = state.get("use_color", True)
    model = PointNet2SemSeg(num_classes, in_channels=6 if use_color else 3).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), use_color
