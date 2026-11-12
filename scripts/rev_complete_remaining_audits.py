"""Complete all remaining completeness checks discovered during our strict 12-comment self-audit:
1. Evaluate Same-Data 13M PoseNet on Real StereoMIS AR (10 seqs, 118 updates) & Real STIR AR (20 clips).
2. Train & evaluate the missing individual loss-term splits:
   - `loss_no_cons_only` (lambda_cons = 0, lambda_rho = 2)
   - `loss_no_rho_only` (lambda_rho = 0, lambda_cons = 1)
   - Evaluate `abl_frozenflow` (lambda_f = 0, stage-6 RAFT flow loss disabled) across all 8 cohorts!
3. Evaluate Inference-time & Retrained ablations on FULL-VIDEO Protocol-B StereoMIS (11 complete videos,
   using fast batch=32 inference on RTX 5090).
"""
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon

cv2.setNumThreads(2)
torch.set_num_threads(4)

ROOT = Path(r"D:\ego_motiion_Camera")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_ar_registration as ear  # noqa: E402
import eval_stir as es  # noqa: E402
import rev_eval_stereomis_real_ar as r_smis_ar  # noqa: E402
import rev_run_local_retrained_fast as r_fast  # noqa: E402
from endosim.eval.metrics import evaluate_trajectory, load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames, select_frame_indices  # noqa: E402
from endosim.lite.data import H_IN, STRIDE, W_IN, load_frames  # noqa: E402
from endosim.lite.egolite import build_from_args, chain  # noqa: E402
from endosim.lite.infer import _intrinsics  # noqa: E402
from endosim.pseudoheight import pseudo_height_label  # noqa: E402

OUT_DIR = ROOT / "results" / "revision_vrih"
STIR_ROOT = Path(r"E:\World_Agent_Enoscopy\datasets\STIR\STIRChallenge_2024")


def part1_split_losses_and_frozenflow() -> None:
    t0 = time.time()
    cache_p = OUT_DIR / "ckpts" / "cached_train_1500x3x6.pt"
    cached_batches = torch.load(cache_p, map_location="cpu")

    cp_cons = r_fast.train_egolite_fast("loss_no_cons_only", cached_batches, w_overrides={"cons": 0.0})
    cp_rho = r_fast.train_egolite_fast("loss_no_rho_only", cached_batches, w_overrides={"rho": 0.0})
    del cached_batches
    torch.cuda.empty_cache()

    with torch.no_grad():
        _eval_part1(cp_cons, cp_rho, t0)


def _eval_part1(cp_cons: Path, cp_rho: Path, t0: float) -> None:

    cohorts = {
        "dev_92": r_fast.read_list("lists/simtest92.txt"),
        "core_265": r_fast.read_list("lists/sim_confirmatory265.txt"),
        "abs_94": r_fast.read_list("lists/abs_test94.txt"),
        "r2s_40": r_fast.read_list("lists/r2s_test.txt"),
        "scared_11": r_fast.read_list("lists/scared_official.txt"),
        "stereomis_11": r_fast.read_list("lists/stereomis.txt"),
        "c3vd_10": r_fast.read_list("lists/c3vd_full.txt"),
    }
    ham = sorted(glob.glob(r"E:\ego_data\sim_data_r2s_ham2\train\seq_*"))
    ham = [d for d in ham if os.path.isfile(os.path.join(d, "pose_c2w.txt"))]
    if ham:
        cohorts["hamlyn_r2s_124"] = ham

    # Load flow model + two new split-loss models + abl_frozenflow (with its own non-finetuned stage-6 flow!)
    ck_base = torch.load(ROOT / "results/egolite/r_v1/last.pth", map_location="cpu")
    m_flow = build_from_args(ck_base["args"], pretrained=False).cuda().eval()
    m_flow.load_state_dict(ck_base["model"])

    loaded_s8 = {}
    for vn, cp in [("retrain_ctrl_full", OUT_DIR / "ckpts/retrain_ctrl_full/last.pth"),
                   ("loss_no_cons_only", cp_cons),
                   ("loss_no_rho_only", cp_rho)]:
        ck = torch.load(cp, map_location="cpu")
        m = build_from_args(ck["args"], pretrained=False)
        m.flow = nn.Identity()
        m.load_state_dict(ck["model"], strict=True)
        loaded_s8[vn] = m.cuda().eval()

    ck_ff = torch.load(ROOT / "results/egolite/abl_frozenflow/last.pth", map_location="cpu")
    m_ff = build_from_args(ck_ff["args"], pretrained=False).cuda().eval()
    m_ff.load_state_dict(ck_ff["model"])
    ck_rv2 = torch.load(ROOT / "results/egolite/r_v2/last.pth", map_location="cpu")
    m_rv2 = build_from_args(ck_rv2["args"], pretrained=False).cuda().eval()
    m_rv2.load_state_dict(ck_rv2["model"])

    res = {vn: {cn: [] for cn in cohorts} for vn in ["retrain_ctrl_full", "loss_no_cons_only", "loss_no_rho_only", "loss_no_flow_rv2_ref", "loss_no_flow"]}
    for cn, seqs in cohorts.items():
        for d in seqs:
            gt_all = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
            idx = select_frame_indices(gt_all, "uniform", max_frames=64)
            paths = list_color_frames(d, indices=idx)
            gt_sel = gt_all[idx]
            imgs, phs, _ = load_frames(paths)
            K = torch.from_numpy(_intrinsics(d)).cuda()
            img = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
            ph = torch.from_numpy(phs)[:, None].cuda()
            n = len(paths)
            ones = torch.ones(n - 1, device="cuda")

            f_s8 = torch.cat([F.avg_pool2d(m_flow.observed_flow(img[st:min(n - 1, st + 16)], img[st + 1:min(n - 1, st + 16) + 1]), STRIDE, STRIDE).float()
                              for st in range(0, n - 1, 16)], 0)
            for vn, mod in loaded_s8.items():
                Rs, ts = [], []
                for st in range(0, n - 1, 32):
                    ed = min(n - 1, st + 32)
                    o = r_fast.forward_with_s8_flow(mod, img[st:ed], img[st + 1:ed + 1], K[None].expand(ed - st, 3, 3), ph[st:ed], ph[st + 1:ed + 1], f_s8[st:ed])
                    Rs.append(o["R"].float()); ts.append(o["t"].float())
                est = chain(torch.cat(Rs, 0), torch.cat(ts, 0), ones).cpu().numpy().astype(np.float64)
                res[vn][cn].append(r_fast.eval_sim3(est, gt_sel))

            for vn, mod_full in [("loss_no_flow_rv2_ref", m_rv2), ("loss_no_flow", m_ff)]:
                Rs, ts = [], []
                for st in range(0, n - 1, 16):
                    ed = min(n - 1, st + 16)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        o = mod_full(img[st:ed], img[st + 1:ed + 1], K[None].expand(ed - st, 3, 3), ph[st:ed], ph[st + 1:ed + 1])
                    Rs.append(o["R"].float()); ts.append(o["t"].float())
                est = chain(torch.cat(Rs, 0), torch.cat(ts, 0), ones).cpu().numpy().astype(np.float64)
                res[vn][cn].append(r_fast.eval_sim3(est, gt_sel))
        print(f"  [Part1 {cn}] no_cons={np.mean(res['loss_no_cons_only'][cn]):.2f} no_rho={np.mean(res['loss_no_rho_only'][cn]):.2f} no_flow={np.mean(res['loss_no_flow'][cn]):.2f} ({time.time()-t0:.0f}s)", flush=True)

    # Merge into r1_3_4_6_7_retrained_ablations.json
    p_json = OUT_DIR / "r1_3_4_6_7_retrained_ablations.json"
    d_json = json.loads(p_json.read_text(encoding="utf-8"))
    for vn in ["loss_no_cons_only", "loss_no_rho_only"]:
        d_json["variants"][vn] = {}
        for cn in cohorts:
            arr, rarr = np.array(res[vn][cn]), np.array(res["retrain_ctrl_full"][cn])
            d_json["variants"][vn][cn] = {
                "mean_mm": float(arr.mean()),
                "median_mm": float(np.median(arr)),
                "p_vs_ctrl": None if np.allclose(arr, rarr) else float(wilcoxon(arr, rarr).pvalue),
            }
    d_json["variants"]["loss_no_flow"] = {}
    for cn in cohorts:
        arr, rarr = np.array(res["loss_no_flow"][cn]), np.array(res["loss_no_flow_rv2_ref"][cn])
        d_json["variants"]["loss_no_flow"][cn] = {
            "mean_mm": float(arr.mean()),
            "median_mm": float(np.median(arr)),
            "ref_rv2_mean_mm": float(rarr.mean()),
            "p_vs_ctrl": None if np.allclose(arr, rarr) else float(wilcoxon(arr, rarr).pvalue),
        }
    p_json.write_text(json.dumps(d_json, indent=2), encoding="utf-8")
    print("Updated r1_3_4_6_7_retrained_ablations.json with all 9 loss terms!", flush=True)
    del m_flow, loaded_s8, m_ff, m_rv2
    torch.cuda.empty_cache()


@torch.no_grad()
def part2_stereomis_fullvideo_ablations() -> None:
    """Evaluate inference-time & retrained ablations on FULL-VIDEO Protocol-B StereoMIS (all frames)!!"""
    t0 = time.time()
    seqs = r_fast.read_list("lists/stereomis.txt")

    ck_rv2 = torch.load(ROOT / "results/egolite/r_v2/last.pth", map_location="cpu")
    m_rv2 = build_from_args(ck_rv2["args"], pretrained=False).cuda().eval()
    m_rv2.load_state_dict(ck_rv2["model"])

    ck_r1 = torch.load(ROOT / "results/egolite/r_v1/last.pth", map_location="cpu")
    m_r1 = build_from_args(ck_r1["args"], pretrained=False).cuda().eval()
    m_r1.load_state_dict(ck_r1["model"])

    ret_names = ["retrain_ctrl_full", "retrain_nogn", "retrain_uniform", "retrain_nolayer", "retrain_novar", "retrain_staticonly", "retrain_instw_fastlr"]
    ret_mods = {}
    for vn in ret_names:
        ck = torch.load(OUT_DIR / "ckpts" / vn / "last.pth", map_location="cpu")
        m = build_from_args(ck["args"], pretrained=False)
        m.flow = nn.Identity()
        if ck.get("learn_inst_weight"):
            m.inst_logit = nn.Parameter(torch.tensor(0.0))
        m.load_state_dict(ck["model"], strict=True)
        if ck.get("ablate_mode"):
            m.ablate = ck["ablate_mode"]
        ret_mods[vn] = (m.cuda().eval(), bool(ck.get("learn_inst_weight")))

    pn_mod = r_fast.SameDataPoseNet(pretrained=False).cuda().eval()
    pn_mod.load_state_dict(torch.load(OUT_DIR / "ckpts/samedata_posenet/last.pth", map_location="cpu")["model"])

    inf_modes = ["full", "nogn", "uniform", "nolayer", "novar", "staticonly"]
    all_tags = [f"inf_{k}" for k in inf_modes] + ret_names + ["samedata_posenet_13M"]
    per_seq = {t: {} for t in all_tags}
    posenet_trajs = {}

    for d in seqs:
        sname = Path(d).name
        gt_all = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
        paths = list_color_frames(d)
        n = min(len(paths), len(gt_all))
        paths, gt_all = paths[:n], gt_all[:n]
        K = torch.from_numpy(_intrinsics(d)).cuda()

        acc_R = {t: [] for t in all_tags}
        acc_t = {t: [] for t in all_tags}

        CHUNK = 128
        for st in range(0, n - 1, CHUNK):
            ed = min(n - 1, st + CHUNK)
            imgs_np, phs_np, _ = load_frames(paths[st:ed + 1])
            img = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).cuda()
            ph = torch.from_numpy(phs_np)[:, None].cuda()
            m = ed - st
            Km = K[None].expand(m, 3, 3)

            # Sub-batches of 32 pairs for flow
            fl_rv2_l, fl_r1_l = [], []
            for sb in range(0, m, 32):
                se = min(m, sb + 32)
                fl_rv2_l.append(F.avg_pool2d(m_rv2.observed_flow(img[sb:se], img[sb + 1:se + 1]), STRIDE, STRIDE).float())
                fl_r1_l.append(F.avg_pool2d(m_r1.observed_flow(img[sb:se], img[sb + 1:se + 1]), STRIDE, STRIDE).float())
            f_rv2_s8 = torch.cat(fl_rv2_l, 0)
            f_r1_s8 = torch.cat(fl_r1_l, 0)

            for imode in inf_modes:
                m_rv2.ablate = None if imode == "full" else imode
                o = r_fast.forward_with_s8_flow(m_rv2, img[:m], img[1:m + 1], Km, ph[:m], ph[1:m + 1], f_rv2_s8, False)
                acc_R[f"inf_{imode}"].append(o["R"].float().cpu())
                acc_t[f"inf_{imode}"].append(o["t"].float().cpu())
            m_rv2.ablate = None

            for vn, (rmod, l_inst) in ret_mods.items():
                o = r_fast.forward_with_s8_flow(rmod, img[:m], img[1:m + 1], Km, ph[:m], ph[1:m + 1], f_r1_s8, l_inst)
                acc_R[vn].append(o["R"].float().cpu())
                acc_t[vn].append(o["t"].float().cpu())

            Rp, tp = pn_mod(img[:m], img[1:m + 1])
            acc_R["samedata_posenet_13M"].append(Rp.float().cpu())
            acc_t["samedata_posenet_13M"].append(tp.float().cpu())

        ones = torch.ones(n - 1)
        for t in all_tags:
            est = chain(torch.cat(acc_R[t], 0), torch.cat(acc_t[t], 0), ones).numpy().astype(np.float64)
            per_seq[t][sname] = r_fast.eval_sim3(est, gt_all)
            if t == "samedata_posenet_13M":
                posenet_trajs[sname] = est
        print(f"  [Full-Video StereoMIS {sname} ({n} fr)] inf_full={per_seq['inf_full'][sname]:.2f} inf_unif={per_seq['inf_uniform'][sname]:.2f} ret_ctrl={per_seq['retrain_ctrl_full'][sname]:.2f} ret_unif={per_seq['retrain_uniform'][sname]:.2f} pn={per_seq['samedata_posenet_13M'][sname]:.2f} ({time.time()-t0:.0f}s)", flush=True)

    # Save Same-Data PoseNet full trajectories for StereoMIS Real AR evaluation!
    pn_dir = OUT_DIR / "posenet_stereomis_full"
    pn_dir.mkdir(exist_ok=True)
    for sname, est in posenet_trajs.items():
        np.savetxt(pn_dir / f"{sname}_c2w.txt", est.reshape(len(est), 16), fmt="%.8f")

    out_summary = {"inference_full_video": {}, "retrained_full_video": {}}
    s_order = [Path(d).name for d in seqs]
    ref_inf = np.array([per_seq["inf_full"][s] for s in s_order])
    for imode in inf_modes:
        arr = np.array([per_seq[f"inf_{imode}"][s] for s in s_order])
        out_summary["inference_full_video"][imode] = {
            "mean_mm": float(arr.mean()),
            "median_mm": float(np.median(arr)),
            "wilcoxon_p_vs_full": None if np.allclose(arr, ref_inf) else float(wilcoxon(arr, ref_inf).pvalue),
            "per_seq": {s: round(per_seq[f"inf_{imode}"][s], 3) for s in s_order},
        }
    ref_ret = np.array([per_seq["retrain_ctrl_full"][s] for s in s_order])
    for vn in ret_names + ["samedata_posenet_13M"]:
        arr = np.array([per_seq[vn][s] for s in s_order])
        out_summary["retrained_full_video"][vn] = {
            "mean_mm": float(arr.mean()),
            "median_mm": float(np.median(arr)),
            "wilcoxon_p_vs_ctrl": None if np.allclose(arr, ref_ret) else float(wilcoxon(arr, ref_ret).pvalue),
            "per_seq": {s: round(per_seq[vn][s], 3) for s in s_order},
        }
    (OUT_DIR / "r1_6_r2_3_stereomis_fullvideo_ablations.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    print("Saved r1_6_r2_3_stereomis_fullvideo_ablations.json!", flush=True)


if __name__ == "__main__":
    part1_split_losses_and_frozenflow()
    part2_stereomis_fullvideo_ablations()
