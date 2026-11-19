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
  stats.py         statistiques de complexité des scènes
tests/           tests unitaires (pytest)
```

## Installation (Windows, local)

Python 3.10 à 3.12 (Open3D et PyTorch ne suivent pas encore les versions plus récentes).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[viz,dev]"
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

## Avancement

- [x] Bibliographie ([docs/bibliographie.md](docs/bibliographie.md))
- [x] Métriques et perturbations
- [x] Scripts de préparation et de statistiques de S3DIS
- [ ] Téléchargement et préparation effective de S3DIS
- [ ] Baseline PointNet++
- [ ] Méthodes via Pointcept (DGCNN, KPConv, MinkUNet, PTv3)
- [ ] Étude de robustesse (densité, bruit, occlusion, complexité)
- [ ] Analyse et rédaction
