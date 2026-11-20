# Segmentation de nuages de points 3D en intérieur : évaluation comparative

Projet de M2 Data & IA. Objectif : comparer plusieurs méthodes de segmentation sémantique de nuages de points 3D (PointNet++, DGCNN, KPConv, MinkUNet, Point Transformer V3…) sur des scènes intérieures (S3DIS, ScanNet), avec les métriques OA, mAcc, IoU et mIoU, et étudier l'influence de la densité, du bruit et de la complexité des scènes.

## Structure

```
configs/         configurations d'expériences
data/            données (non versionnées, voir data/README.md)
docs/            bibliographie, fiches de lecture
scripts/         scripts exécutables (démos, entraînement, évaluation)
src/segbench/    code du projet
  metrics.py       OA, mAcc, IoU, mIoU (matrice de confusion)
  corruptions.py   perturbations de densité, bruit, outliers, occlusion, couleur
  datasets/s3dis.py  lecture de S3DIS brut, conversion au format Pointcept
  datasets/blocks.py découpage en blocs 1 m x 1 m (entraînement, test par grille glissante)
  models/pointnet2.py  PointNet++ (SSG) en PyTorch pur
  stats.py         statistiques de complexité des scènes
  robustness.py    banc d'évaluation de robustesse (mêmes perturbations pour tous les modèles)
tests/           tests unitaires (pytest)
```

## Installation (Windows, local)

Python 3.10 à 3.12 (Open3D et PyTorch ne suivent pas encore les versions plus récentes).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[viz,dev]"
```

PyTorch (pour l'entraînement) : version CPU sur un PC sans carte NVIDIA, version CUDA sur un serveur GPU (voir [pytorch.org](https://pytorch.org/get-started/locally/)).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Utilisation

```bash
pytest                                                   # tests
python scripts/demo_corruptions.py --corruption local_dropout --severity 4
```

Préparation de S3DIS (après téléchargement, voir [data/README.md](data/README.md)) :

```bash
python scripts/prepare_s3dis.py --raw data/raw/s3dis/Stanford3dDataset_v1.2_Aligned_Version --out data/processed/s3dis
python scripts/s3dis_stats.py --data data/processed/s3dis --out outputs/s3dis_stats.csv
```

Entraînement de PointNet++ (test sur l'Area 5) :

```bash
python scripts/train_pointnet2.py --data data/processed/s3dis --out outputs/pointnet2
```

Sur CPU, un pas d'entraînement (8 blocs de 4096 points) prend environ 5 s : l'entraînement complet nécessite un GPU. Pour un simple essai : `--epochs 1 --samples-per-epoch 64 --batch-size 4 --max-test-rooms 2`.

Évaluation de robustesse (Area 5 propre puis perturbée, 6 perturbations x 5 niveaux) et courbes :

```bash
python scripts/eval_robustness.py --data data/processed/s3dis --checkpoint outputs/pointnet2/best.pth --name pointnet2 --out outputs/robustesse/pointnet2.csv --seeds 0 1 2
python scripts/plot_robustness.py outputs/robustesse/*.csv --out outputs/robustesse/courbes.png
```

Protocole : chaque pièce est perturbée avant le découpage en blocs, avec une graine fixée par (graine, pièce) pour que tous les modèles voient les mêmes nuages ; les métriques portent sur les points de la scène perturbée (les points aberrants ajoutés sont ignorés).

## Avancement

- [x] Bibliographie ([docs/bibliographie.md](docs/bibliographie.md))
- [x] Métriques et perturbations
- [x] Scripts de préparation et de statistiques de S3DIS
- [ ] Téléchargement et préparation effective de S3DIS
- [x] Implémentation de PointNet++ (modèle, blocs, entraînement, évaluation)
- [ ] Entraînement de PointNet++ sur S3DIS (GPU)
- [ ] Méthodes via Pointcept (DGCNN, KPConv, MinkUNet, PTv3)
- [x] Banc d'évaluation de robustesse (évaluation, CSV, courbes)
- [ ] Étude de robustesse sur les modèles entraînés (densité, bruit, occlusion, complexité)
- [ ] Analyse et rédaction
