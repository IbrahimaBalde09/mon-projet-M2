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

## Avancement

- [x] Bibliographie ([docs/bibliographie.md](docs/bibliographie.md))
- [x] Métriques et perturbations
- [ ] Préparation de S3DIS
- [ ] Baseline PointNet++
- [ ] Méthodes via Pointcept (DGCNN, KPConv, MinkUNet, PTv3)
- [ ] Étude de robustesse (densité, bruit, occlusion, complexité)
- [ ] Analyse et rédaction
