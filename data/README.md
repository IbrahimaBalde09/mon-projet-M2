# Données

Les jeux de données ne sont pas versionnés (voir `.gitignore`). Organisation attendue :

```
data/
├── raw/         # archives et fichiers originaux téléchargés
│   ├── s3dis/
│   └── scannet/
└── processed/   # données prétraitées (format Pointcept : coord, color, segment)
```

## S3DIS
- Demande d'accès : formulaire sur la page du jeu de données (Stanford 2D-3D-Semantics / S3DIS).
- Version à télécharger : `Stanford3dDataset_v1.2_Aligned_Version`.
- Protocole : entraînement sur les Areas 1, 2, 3, 4, 6 ; test sur l'Area 5.

## ScanNet v2
- Accès sur demande : signer le *Terms of Use* (lien sur le dépôt GitHub officiel de ScanNet), puis utiliser le script de téléchargement fourni par mail.
- Protocole : découpage officiel train / val (20 classes).
