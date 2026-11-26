"""Analyse un scan LiDAR : format, scanner, couleur, densité, bruit, orientation.

Exemples :
    python scripts/inspect_scan.py --input mes_scans/bureau.e57
    python scripts/inspect_scan.py --input mes_scans/bureau.las \
        --reference data/processed/s3dis/Area_5/office_1 --json outputs/scans/bureau.json

Avec --reference (une pièce S3DIS préparée), le scan est comparé à S3DIS :
densité relative et bruit en plus, traduits en niveaux de random_subsample et
gaussian_noise. Les fichiers .e57 et .las contiennent souvent la marque et le
modèle du scanner : ils sont affichés s'ils sont présents.
"""

import argparse
import json
from pathlib import Path

from segbench.datasets.s3dis import load_scene
from segbench.scan_io import read_scan
from segbench.scan_stats import compare_to, describe


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0, help="facteur vers les mètres (0.001 si en mm)")
    parser.add_argument("--up", choices=["z", "y"], default="z", help="axe vertical du fichier")
    parser.add_argument("--reference", type=Path, help="pièce S3DIS préparée, pour comparaison")
    parser.add_argument("--json", type=Path, help="enregistrer le rapport en JSON")
    args = parser.parse_args()

    scan = read_scan(args.input, args.scale, args.up)
    d = describe(scan)
    meta = scan["meta"]

    print(f"Fichier      : {args.input.name} ({meta['format']})")
    infos = {k: v for k, v in meta.items() if k != "format"}
    print("Scanner      : " + (", ".join(f"{k} = {v}" for k, v in infos.items()) if infos
                                else "aucune information dans le fichier"))
    print(f"Points       : {d['n_points']:,}")
    print(f"Dimensions   : {' x '.join(f'{v:.2f}' for v in d['dimensions_m'])} m")
    print(f"Couleur RGB  : {'oui' if d['couleur'] else 'NON -> entraîner avec --no-color'}")
    print(f"Intensité    : {'oui' if d['intensite'] else 'non'}")
    print(f"Densité      : {d['densite_pts_m2']:,.0f} points par m² de surface "
          f"(soit un point tous les {1000 * d['espacement_m']:.1f} mm environ)")
    print(f"Bruit        : {1000 * d['bruit_m']:.1f} mm (écart-type sur le plus grand plan, borne haute)")
    p = d["plus_grand_plan"]
    print(f"Plus grand plan : normale selon {p['axe']} ({100 * p['part_points']:.0f} % des points)")

    alerts = []
    if max(d["dimensions_m"]) > 100 or d["dimensions_m"][2] > 20:
        alerts.append("dimensions énormes : fichier en mm ou en cm ? essayer --scale 0.001 ou 0.01")
    elif max(d["dimensions_m"]) < 0.5:
        alerts.append("dimensions minuscules : vérifier --scale")
    if p["axe"] == "y" and args.up == "z":
        alerts.append("le plus grand plan est horizontal selon y : le fichier a peut-être y vers le haut (--up y)")
    for a in alerts:
        print(f"Attention    : {a}")

    report = {"fichier": str(args.input), "meta": meta, **d}
    if args.reference:
        ref = describe(load_scene(args.reference))
        c = compare_to(d, ref)
        report["reference"] = {"scene": str(args.reference), **ref, "comparaison": c}
        print(f"\nComparaison avec {args.reference} :")
        print(f"  densité référence    : {ref['densite_pts_m2']:,.0f} points par m² ; bruit référence : {1000 * ref['bruit_m']:.1f} mm")
        print(f"  densité relative     : {c['densite_relative']:.2f} x celle de la référence"
              + (f"  (proche de random_subsample niveau {c['niveau_densite']})" if c["niveau_densite"] else ""))
        print(f"  bruit en plus        : {1000 * c['bruit_supplementaire_m']:.1f} mm"
              + (f"  (proche de gaussian_noise niveau {c['niveau_bruit']})" if c["niveau_bruit"] else "  (négligeable)"))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
