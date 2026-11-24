"""Unified lossless-PNG reruns for Table 6 Top Block, Table R3/S3, Hamlyn-R2S depth, and Fig. 18 (8x fix).

Eliminates ALL JPEG-95 drift between Table 3 (4.27 / 2.52 / 27.30 / 6.29) and Table 6 Top / Table R3/S3
(formerly 4.40 / 2.59 / 27.38 / 6.49), adds source-disjoint Hamlyn-R2S (124 seqs) monocular depth evaluation,
and verifies Fig. 18 uncertainty calibration with exact 8*exp(s) pixel L1 units.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rev_eval_depth_and_mechanisms as rdm  # noqa: E402
from endosim.eval.metrics import evaluate_trajectory, load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames, select_frame_indices  # noqa: E402
from endosim.lite.data import H_IN, W_IN, load_frames  # noqa: E402
from endosim.lite.egolite import build_from_args  # noqa: E402
from endosim.lite.infer import _intrinsics  # noqa: E402

OUT_DIR = ROOT / "results" / "revision_vrih"
DEPTH_UNIT_MM = 50.0


def read_list(rel: str) -> list[str]:
    return [
        ln.strip()
        for ln in (ROOT / rel).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#") and os.path.isfile(os.path.join(ln.strip(), "pose_c2w.txt"))
    ]


@torch.no_grad()
def eval_hamlyn_depth(models: dict[str, torch.nn.Module]) -> dict:
    ham = sorted(glob.glob(r"E:\ego_data\sim_data_r2s_ham2\train\seq_*"))
    ham = [d for d in ham if os.path.isfile(os.path.join(d, "pose_c2w.txt"))]
    print(f"[Depth Hamlyn-R2S] Evaluating {len(ham)} unseen sequences ...", flush=True)
    return rdm.eval_depth_sim_cohort(models, ham, sample_pairs=4)


def main():
    t0 = time.time()
    # 1. Regenerate Fig. 18 uncertainty calibration (skip if already done)
    import shutil
    if not (ROOT / "VRIH_EgoLite_submission/figs/fig_unc_calibration.png").is_file() or "--all" in sys.argv:
        print("[1/4] Regenerating Fig. 18 uncertainty calibration (8x L1 pixel unit fix) ...", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/make_bspc_uncertainty_calibration.py")], check=True)
        shutil.copy2(ROOT / "bspc_paper/figs/fig_unc_calibration.png",
                     ROOT / "VRIH_EgoLite_submission/figs/fig_unc_calibration.png")

    # Load r_v2
    ck_rv2 = torch.load(ROOT / "results/egolite/r_v2/last.pth", map_location="cpu", weights_only=False)
    m_rv2 = build_from_args(ck_rv2["args"], pretrained=False).cuda().eval()
    m_rv2.load_state_dict(ck_rv2["model"])

    d_path = OUT_DIR / "r1_2_depth_accuracy.json"
    d_json = json.loads(d_path.read_text(encoding="utf-8"))
    if "hamlyn_r2s_124_unseen" not in d_json.get("r_v2", {}):
        ck_noph = torch.load(ROOT / "results/egolite/abl_noph/last.pth", map_location="cpu", weights_only=False)
        m_noph = build_from_args(ck_noph["args"], pretrained=False).cuda().eval()
        m_noph.load_state_dict(ck_noph["model"])
        print("[2/4] Evaluating lossless Hamlyn-R2S (124 unseen seqs) depth ...", flush=True)
        ham_d = eval_hamlyn_depth({"r_v2": m_rv2, "abl_noph": m_noph})
        for k in ("r_v2", "abl_noph"):
            d_json[k]["hamlyn_r2s_124_unseen"] = ham_d[k]
        d_path.write_text(json.dumps(d_json, indent=2), encoding="utf-8")
        del m_noph
        torch.cuda.empty_cache()

    # 3. Unified lossless PNG Table 6 Top Block (all 7 cohorts!) + Table R3/S3 (dev_92, abs_94, stereomis_11)
    rdm.BATCH = 32  # RTX 5090 24GB VRAM -> ~8x faster than BATCH=2
    print("[3/4] Evaluating lossless PNG Table 6 Top Block & Table R3/S3 solver/occlusion variants (BATCH=32) ...", flush=True)
    cohorts_all = {
        "dev_92": (read_list("lists/simtest92.txt"), True),
        "core_265": (read_list("lists/sim_confirmatory265.txt"), False),
        "abs_94": (read_list("lists/abs_test94.txt"), True),
        "r2s_40": (read_list("lists/r2s_test.txt"), False),
        "scared_11": (read_list("lists/scared_official.txt"), False),
        "stereomis_11": (read_list("lists/stereomis.txt"), True),
        "c3vd_10": (read_list("lists/c3vd_full.txt"), False),
    }
    t6_p = OUT_DIR / "r1_6_r2_3_table6_lossless.json"
    r3_p = OUT_DIR / "r1_3_design_choices_lossless.json"
    t6_out = json.loads(t6_p.read_text(encoding="utf-8")) if t6_p.is_file() else {}
    r3_out = json.loads(r3_p.read_text(encoding="utf-8")) if r3_p.is_file() else {"solver_and_occlusion_variants": {}, "region_distributions_r_v2": {}}
    for cn, (seqs, with_solver) in cohorts_all.items():
        if cn in t6_out and (not with_solver or cn in r3_out["solver_and_occlusion_variants"]):
            print(f"  [Skip cached {cn}]", flush=True)
            continue
        res = rdm.eval_all_infer_variants_cohort(m_rv2, seqs, max_frames=64, with_solver_choices=with_solver)
        t6_keys = ["full", "nogn", "uniform", "nolayer", "novar", "staticonly"]
        t6_out[cn] = {k: res[k] for k in t6_keys if k in res}
        if with_solver:
            r3_keys = ["full", "stride4_gn", "fullres_gn", "oob_mask", "fb_occ_mask", "_timing_ms_median"]
            r3_out["solver_and_occlusion_variants"][cn] = {k: res[k] for k in r3_keys if k in res}
        t6_p.write_text(json.dumps(t6_out, indent=2), encoding="utf-8")
        r3_p.write_text(json.dumps(r3_out, indent=2), encoding="utf-8")
        print(f"  [Lossless {cn:12s}] full={res['full']['mean_mm']:.2f} nogn={res['nogn']['mean_mm']:.2f} "
              f"unif={res['uniform']['mean_mm']:.2f} nolay={res['nolayer']['mean_mm']:.2f} "
              f"novar={res['novar']['mean_mm']:.2f} stat={res['staticonly']['mean_mm']:.2f} ({time.time()-t0:.0f}s)", flush=True)

    # 4. Region distributions on lossless PNGs (dev_92 and abs_94)
    print("[4/4] Evaluating lossless PNG region distributions (dev_92, abs_94) ...", flush=True)
    r3_out["region_distributions_r_v2"]["dev_92"] = rdm.audit_labels_and_regions(m_rv2, None, cohorts_all["dev_92"][0])["region_audit_r_v2"]
    r3_out["region_distributions_r_v2"]["abs_94"] = rdm.audit_labels_and_regions(m_rv2, None, cohorts_all["abs_94"][0])["region_audit_r_v2"]

    t6_p.write_text(json.dumps(t6_out, indent=2), encoding="utf-8")
    r3_p.write_text(json.dumps(r3_out, indent=2), encoding="utf-8")
    print(f"DONE Stage 3 lossless audits in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
