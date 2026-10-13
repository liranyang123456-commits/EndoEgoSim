"""Image-space decoupling metrics of r_v2 and the label-ablation checkpoints, side by side."""
import json
import os
import sys

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "egolite")
rows = {"r_v2 (sequence-level labels)": "decoupling_r_v2",
        "per-pair labels, last stage only": "decoupling_r_v2_pairlab",
        "per-pair labels, stages 5 and 6": "decoupling_r_v2_pairlabfull"}
for ds in ("abs", "r2s"):
    print("==", ds)
    for name, pre in rows.items():
        p = os.path.join(root, f"{pre}_{ds}.json")
        if not os.path.exists(p):
            print(f"  {name:36s} (missing)")
            continue
        js = json.load(open(p, encoding="utf-8"))
        ti = js["tissue_motion_image"]
        o = ti["ours"]
        print(f"  {name:36s} median {o['epe_px_median']:.3f} px  AUROC {o['moving_vs_static_auroc']:.3f}  "
              f"cos {o['moving_direction_cosine']:.3f}  | oracle median {ti['oracle']['epe_px_median']:.3f}  relonly {ti['relonly']['epe_px_median']:.3f}")
