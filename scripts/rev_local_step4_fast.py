"""Fast local RTX 5090 completion of Step-4: Stage-6 3 seeds (s0, s1, s2) + 6-stage progression (v1 -> r_v2) across all cohorts."""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

cv2.setNumThreads(1)
torch.set_num_threads(2)

ROOT = Path(r"D:\ego_motiion_Camera")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from endosim.eval.metrics import evaluate_trajectory  # noqa: E402
from endosim.lite.egolite import build_from_args  # noqa: E402
from endosim.lite.infer import infer_sequence  # noqa: E402

OUT = ROOT / "results" / "revision_vrih"


def eval_sim3(est: np.ndarray, gt: np.ndarray) -> float:
    return float(evaluate_trajectory(est, gt)["ate_sim3"]["rmse"])


def load_egolite(rel_ckpt: str) -> torch.nn.Module:
    p = ROOT / rel_ckpt
    if not p.is_file():
        p = p.with_name("last.pth")
    ck = torch.load(str(p), map_location="cpu")
    m = build_from_args(dict(ck.get("args", {})), pretrained=False)
    m.load_state_dict(ck["model"], strict=False)
    return m.cuda().eval()


def read_list(rel: str) -> list[str]:
    return [ln.strip() for ln in (ROOT / rel).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#") and os.path.isfile(os.path.join(ln.strip(), "pose_c2w.txt"))]


@torch.no_grad()
def cohort_ates(mod, slist):
    return [eval_sim3(*infer_sequence(mod, d, max_frames=64, batch=16)[:2]) for d in slist]


def main():
    t0 = time.time()
    cohorts_map = {
        "dev_92": read_list("lists/simtest92.txt"),
        "core_265": read_list("lists/sim_confirmatory265.txt"),
        "abs_94": read_list("lists/abs_test94.txt"),
        "r2s_40": read_list("lists/r2s_test.txt"),
        "scared_11": read_list("lists/scared_official.txt"),
        "stereomis_11": read_list("lists/stereomis.txt"),
        "c3vd_10": read_list("lists/c3vd_full.txt"),
    }
    ham = sorted(glob.glob(r"E:\ego_data\sim_data_r2s_ham2\train\seq_*"))
    ham = [d for d in ham if os.path.isfile(os.path.join(d, "pose_c2w.txt"))]
    if ham:
        cohorts_map["hamlyn_r2s_124"] = ham

    seed_ckpts = {"seed0": "results/egolite/r_v2/last.pth",
                  "seed1": "results/egolite/r_v2_s1/last.pth",
                  "seed2": "results/egolite/r_v2_s2/last.pth"}
    seed_means = {}
    for sn, sp in seed_ckpts.items():
        mod = load_egolite(sp)
        seed_means[sn] = {cn: float(np.mean(cohort_ates(mod, sl))) for cn, sl in cohorts_map.items()}
        print(f"  done {sn} ({time.time() - t0:.0f}s): {seed_means[sn]}", flush=True)
        del mod; torch.cuda.empty_cache()

    seed_summary = {
        cn: {"mean_mm": float(np.mean([seed_means[s][cn] for s in seed_means])),
             "std_mm": float(np.std([seed_means[s][cn] for s in seed_means], ddof=1)),
             "runs": [round(seed_means[s][cn], 3) for s in seed_means]}
        for cn in cohorts_map
    }

    stage_ckpts = [("stage1_v1", "results/egolite/v1/best.pth"),
                   ("stage2_v2", "results/egolite/v2/best.pth"),
                   ("stage3_v3abs", "results/egolite/v3abs/best.pth"),
                   ("stage4_s_big_ft", "results/egolite/s_big_ft/last.pth"),
                   ("stage5_r_v1", "results/egolite/r_v1/last.pth"),
                   ("stage6_r_v2", "results/egolite/r_v2/last.pth")]
    stage_res = {}
    for stn, stp in stage_ckpts:
        if stn == "stage6_r_v2":
            stage_res[stn] = {cn: round(seed_means["seed0"][cn], 3) for cn in cohorts_map}
            continue
        mod = load_egolite(stp)
        stage_res[stn] = {cn: round(float(np.mean(cohort_ates(mod, sl))), 3) for cn, sl in cohorts_map.items()}
        print(f"  done {stn} ({time.time() - t0:.0f}s): {stage_res[stn]}", flush=True)
        del mod; torch.cuda.empty_cache()

    (OUT / "r1_4_seeds_and_stages_real.json").write_text(json.dumps({
        "stage6_three_seeds_all_cohorts": seed_summary,
        "six_stage_progression": stage_res,
    }, indent=2), encoding="utf-8")
    print("LOCAL STEP 4 COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
