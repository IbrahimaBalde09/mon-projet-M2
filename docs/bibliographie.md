# Bibliographie commentée

Sujet : *Évaluation comparative de méthodes de segmentation de nuages de points 3D en environnement intérieur.*

Les références marquées ★ sont prioritaires (lecture détaillée). Les entrées BibTeX correspondantes sont dans [`references.bib`](references.bib) ; chaque article lu donne lieu à une fiche dans [`fiches/`](fiches/) à partir du [modèle](fiches/_modele.md).

## Méthode de lecture

- **Zotero** : une collection par catégorie (A à E ci-dessous), export BibTeX pour le mémoire.
- **Trois passes** : (1) résumé, figures, conclusion pour tout ; (2) méthode en détail pour les ★ ; (3) lecture du code en parallèle pour les méthodes qui seront entraînées.
- **Une fiche par article**, avec une rubrique obligatoire : *implications pour la robustesse* (densité, bruit, complexité de scène).

## A. Vue d'ensemble

| | Référence | Pourquoi |
|---|---|---|
| ★ | Guo et al., [*Deep Learning for 3D Point Clouds: A Survey*](https://arxiv.org/abs/1912.12033), TPAMI 2021 | Carte du domaine, taxonomie des méthodes |
| | Goyal et al., [*Revisiting Point Cloud Shape Classification with a Simple and Effective Baseline*](https://arxiv.org/abs/2106.05304) (SimpleView), ICML 2021 | Le protocole (augmentations, évaluation) pèse autant que l'architecture : argument pour une comparaison équitable |

## B. Méthodes

### Point-based (MLP)
| | Référence | Pourquoi |
|---|---|---|
| ★ | Qi et al., [*PointNet*](https://arxiv.org/abs/1612.00593), CVPR 2017 | Fondateur |
| ★ | Qi et al., [*PointNet++*](https://arxiv.org/abs/1706.02413), NeurIPS 2017 | Baseline implémentée à la main ; FPS, ball query, MSG/MRG face aux variations de densité |
| | Qian et al., [*PointNeXt*](https://arxiv.org/abs/2206.04670), NeurIPS 2022 | Un modèle ancien bien entraîné remonte très haut : enjeu d'équité |
| | Hu et al., [*RandLA-Net*](https://arxiv.org/abs/1911.11236), CVPR 2020 | Échantillonnage aléatoire efficace, lien direct avec l'axe densité |

### Graphe
| | Référence | Pourquoi |
|---|---|---|
| ★ | Wang et al., [*Dynamic Graph CNN for Learning on Point Clouds*](https://arxiv.org/abs/1801.07829) (DGCNN), ACM TOG 2019 | Voisinage dynamique (EdgeConv) |

### Convolution sur points
| | Référence | Pourquoi |
|---|---|---|
| ★ | Thomas et al., [*KPConv*](https://arxiv.org/abs/1904.08889), ICCV 2019 | Référence robuste ; section sur densité et sous-échantillonnage par grille |

### Voxels creux
| | Référence | Pourquoi |
|---|---|---|
| | Graham et al., [*Submanifold Sparse Convolutional Networks*](https://arxiv.org/abs/1711.10275), CVPR 2018 | Convolutions creuses |
| ★ | Choy et al., [*Minkowski Convolutional Neural Networks*](https://arxiv.org/abs/1904.08755), CVPR 2019 | MinkUNet, référence voxel |

### Transformers
| | Référence | Pourquoi |
|---|---|---|
| | Zhao et al., [*Point Transformer*](https://arxiv.org/abs/2012.09164), ICCV 2021 | Premier transformer local efficace |
| | Wu et al., [*Point Transformer V2*](https://arxiv.org/abs/2210.05666), NeurIPS 2022 | Attention vectorielle groupée |
| ★ | Wu et al., [*Point Transformer V3*](https://arxiv.org/abs/2312.10035), CVPR 2024 | État de l'art de référence dans Pointcept |
| | Lai et al., [*Stratified Transformer*](https://arxiv.org/abs/2203.14508), CVPR 2022 | Optionnel |
| | Peng et al., [*OA-CNNs*](https://arxiv.org/abs/2403.14418), CVPR 2024 | Optionnel |
| | Wu et al., [*Sonata*](https://arxiv.org/abs/2503.16429), CVPR 2025 | Optionnel : pré-entraînement auto-supervisé, ouverture vers la thèse |

## C. Jeux de données

| | Référence | Pourquoi |
|---|---|---|
| ★ | Armeni et al., [*3D Semantic Parsing of Large-Scale Indoor Spaces*](https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf) (S3DIS), CVPR 2016 | Jeu principal ; 13 classes, test Area 5, pièces typées |
| ★ | Dai et al., [*ScanNet*](https://arxiv.org/abs/1702.04405), CVPR 2017 | Second jeu ; **accès sur demande à faire tôt** |
| | Rozenberszki et al., [*ScanNet200*](https://arxiv.org/abs/2204.07761), ECCV 2022 | Classes rares (long tail) |
| | Yeshwanth et al., [*ScanNet++*](https://arxiv.org/abs/2308.11417), ICCV 2023 | Haute densité |

## D. Robustesse (angle de contribution)

| | Référence | Pourquoi |
|---|---|---|
| ★ | Sun et al., [*Benchmarking Robustness of 3D Point Cloud Recognition Against Common Corruptions*](https://arxiv.org/abs/2201.12296) (ModelNet40-C), 2022 | Corruptions sur objets |
| ★ | Ren et al., [*Benchmarking and Analyzing Point Cloud Classification under Corruptions*](https://arxiv.org/abs/2202.03377) (ModelNet-C), ICML 2022 | Taxonomie des corruptions à reprendre |
| | Kong et al., [*Robo3D*](https://arxiv.org/abs/2303.17597), ICCV 2023 | Équivalent en segmentation, LiDAR extérieur |
| | [Benchmarking the Robustness of LiDAR Semantic Segmentation Models](https://arxiv.org/abs/2301.00970), 2023 | Idem, SemanticKITTI-C / SemanticPOSS |
| | [BelHouse3D](https://arxiv.org/abs/2411.13251), 2024 | Intérieur, mais occlusion uniquement |
| | [CDSegNet](https://arxiv.org/abs/2411.16308), 2024 | Méthode revendiquant une robustesse au bruit sur ScanNet |

**Piste identifiée (à confirmer) :** pas de benchmark de corruptions systématique (densité + bruit, niveaux de sévérité, à la Robo3D) trouvé pour S3DIS / ScanNet. À vérifier via Google Scholar et les citations de ModelNet-C et Robo3D.

## E. Outils

- Zhou, Park, Koltun, [*Open3D: A Modern Library for 3D Data Processing*](https://arxiv.org/abs/1801.09847), 2018.
- [Pointcept](https://github.com/Pointcept/Pointcept) : codebase unifiée (à citer).

## Planning de lecture

| Semaine | Lectures | Livrable |
|---|---|---|
| 1 | Survey, PointNet, PointNet++, SimpleView | 4 fiches + schéma de taxonomie |
| 2 | S3DIS, ScanNet, DGCNN, KPConv | Fiches + demande d'accès ScanNet |
| 3 | Minkowski, PTv1→v3, ModelNet-C / 40-C, Robo3D | Fiches + liste des corruptions retenues |
| 4 | BelHouse3D, CDSegNet, PointNeXt, recherche du manque | Synthèse de 3-4 pages + tableau des mIoU publiés |

## Tableau des résultats publiés

À remplir à partir des articles : sert de référence pour valider les reproductions.

| Méthode | S3DIS Area 5 mIoU | ScanNet val mIoU | Source (table / page) |
|---|---|---|---|
| PointNet | | | |
| PointNet++ | | | |
| DGCNN | | | |
| KPConv | | | |
| MinkUNet | | | |
| PTv3 | | | |
