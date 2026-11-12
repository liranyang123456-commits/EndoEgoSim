"""Re-train learnable instrument weight with 1e-2 gate LR + add SameDataPoseNet to Sim-AR Table 10."""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, "scripts")
import eval_ar_registration as ear  # noqa: E402
from rev_run_local_retrained_fast import (  # noqa: E402
    CKPT_ROOT,
    OUT_DIR,
    SameDataPoseNet,
    eval_sim3,
    forward_with_s8_flow,
    precompute_train_batches,
    read_list,
    train_egolite_fast,
)
from endosim.eval.metrics import load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames, select_frame_indices  # noqa: E402
from endosim.lite.data import STRIDE, load_frames, scaled_K  # noqa: E402
from endosim.lite.egolite import build_from_args, chain  # noqa: E402
from endosim.lite.infer import _intrinsics  # noqa: E402
import torch.nn as nn
import torch.nn.functional as F

# 1. Retrain instw_fastlr with 1e-2 gate LR on the cached batches
cached = precompute_train_batches(n_steps=1500, batch_clips=3, clip_len=6)
inst_p = CKPT_ROOT / "retrain_instw_fastlr" / "last.pth"
if inst_p.is_file():
    inst_p.unlink()
ck_inst = train_egolite_fast("retrain_instw_fastlr", cached, learn_inst_weight=True)
del cached
torch.cuda.empty_cache()

ck_data = torch.load(ck_inst, map_location="cpu")
print("Converged alpha_inst:", ck_data["learned_alpha_inst"], "alpha_deform:", ck_data["learned_alpha_deform"])

# Evaluate instw_fastlr on dev_92, core_265, abs_94, stereomis_11
ck_base = torch.load("results/egolite/r_v1/last.pth", map_location="cpu")
m_flow = build_from_args(ck_base["args"], pretrained=False).cuda().eval()
m_flow.load_state_dict(ck_base["model"])

m_inst = build_from_args(ck_data["args"], pretrained=False)
m_inst.flow = nn.Identity()
m_inst.inst_logit = nn.Parameter(torch.tensor(0.0))
m_inst.load_state_dict(ck_data["model"], strict=True)
m_inst = m_inst.cuda().eval()

m_ctrl_ck = torch.load(CKPT_ROOT / "retrain_ctrl_full" / "last.pth", map_location="cpu")
m_ctrl = build_from_args(m_ctrl_ck["args"], pretrained=False)
m_ctrl.flow = nn.Identity()
m_ctrl.load_state_dict(m_ctrl_ck["model"], strict=True)
m_ctrl = m_ctrl.cuda().eval()

cohorts = {
    "dev_92": read_list("lists/simtest92.txt"),
    "core_265": read_list("lists/sim_confirmatory265.txt"),
    "abs_94": read_list("lists/abs_test94.txt"),
    "r2s_40": read_list("lists/r2s_test.txt"),
    "scared_11": read_list("lists/scared_official.txt"),
    "stereomis_11": read_list("lists/stereomis.txt"),
    "c3vd_10": read_list("lists/c3vd_full.txt"),
}
inst_res = {}
with torch.no_grad():
    for cn, seqs in cohorts.items():
        vi, vc = [], []
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
            f_s8 = torch.cat([F.avg_pool2d(m_flow.observed_flow(img[st:min(n-1,st+8)], img[st+1:min(n-1,st+8)+1]), STRIDE, STRIDE).float()
                              for st in range(0, n - 1, 8)], 0)
            ones = torch.ones(n - 1, device="cuda")
            oi = forward_with_s8_flow(m_inst, img[:-1], img[1:], K[None].expand(n - 1, 3, 3), ph[:-1], ph[1:], f_s8, True)
            oc = forward_with_s8_flow(m_ctrl, img[:-1], img[1:], K[None].expand(n - 1, 3, 3), ph[:-1], ph[1:], f_s8, False)
            vi.append(eval_sim3(chain(oi["R"].float(), oi["t"].float(), ones).cpu().numpy().astype(np.float64), gt_sel))
            vc.append(eval_sim3(chain(oc["R"].float(), oc["t"].float(), ones).cpu().numpy().astype(np.float64), gt_sel))
        inst_res[cn] = {
            "mean_mm": float(np.mean(vi)),
            "median_mm": float(np.median(vi)),
            "p_vs_ctrl": float(wilcoxon(np.array(vi), np.array(vc)).pvalue),
        }
        print(f"instw_fastlr {cn}: {inst_res[cn]['mean_mm']:.2f} vs ctrl {np.mean(vc):.2f} (p={inst_res[cn]['p_vs_ctrl']:.4f})")

summ_p = OUT_DIR / "r1_3_4_6_7_retrained_ablations.json"
summ = json.loads(summ_p.read_text(encoding="utf-8"))
summ["extra_learned_scalars"]["retrain_instw_fastlr"] = {
    "learned_alpha_inst": ck_data["learned_alpha_inst"],
    "learned_alpha_deform": ck_data["learned_alpha_deform"],
}
summ["variants"]["retrain_instw_fastlr"] = inst_res
summ_p.write_text(json.dumps(summ, indent=2), encoding="utf-8")

# 2. Add SameDataPoseNet (both native 50-mm scale and true step scale!) into r2_2_sim_ar_expanded.json
pn_mod = SameDataPoseNet(pretrained=False).cuda().eval()
pn_mod.load_state_dict(torch.load(CKPT_ROOT / "samedata_posenet" / "last.pth", map_location="cpu")["model"])

ar_p = OUT_DIR / "r2_2_sim_ar_expanded.json"
ar_d = json.loads(ar_p.read_text(encoding="utf-8"))
pn_native_meds, pn_ts_meds = [], []

with torch.no_grad():
    for r in ar_d["per_sequence_medians"]:
        seq = os.path.join("sim_data_abs/test", r["sequence"])
        K, pose, traj = ear.load_clip(seq); Xm = traj[0]; s = r["stride"]
        en_list, ets_list = [], []
        for t in range(0, len(pose) - s, s):
            pa = os.path.join(seq, "color", f"{t:06d}.png")
            pb = os.path.join(seq, "color", f"{t + s:06d}.png")
            g0, g1 = cv2.imread(pa, 0), cv2.imread(pb, 0)
            Xc = (pose[t][:3, :3].T @ (traj[t] - pose[t][:3, 3]).T).T
            R0, t0 = ear.kabsch(Xm, Xc)
            src, za = ear.project_w(traj[t], pose[t], K)
            truth, zb = ear.project_w(traj[t + s], pose[t + s], K)
            vis = (za > 1) & (zb > 1)
            if vis.mean() < 0.7:
                continue
            Re, te = ear.cam_static(pose[t], pose[t + s]); mag = float(np.linalg.norm(te))
            imgs, _, _ = load_frames([pa, pb])
            im_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
            Rp, tp = pn_mod(im_t[:1], im_t[1:])
            Rn = Rp[0].double().cpu().numpy()
            tn_nat = tp[0].double().cpu().numpy() * 50.0
            tn_ts = tn_nat / max(np.linalg.norm(tn_nat), 1e-8) * mag
            for tn_cur, lst in [(tn_nat, en_list), (tn_ts, ets_list)]:
                uvp, zp = ear.project_c(K, (Rn @ Xc.T).T + tn_cur)
                pred, guess = ((src, (R0, t0)) if np.median(zp) < 1 else (uvp, (Rn @ R0, Rn @ t0 + tn_cur)))
                got = ear.register_local(g0, g1, src, pred, Xm, K, guess, vis, 18)
                lst.append(ear.px(got[vis], truth[vis]))
        r["medians"]["local_samedata_posenet"] = round(float(np.median(en_list)), 2)
        r["medians"]["local_samedata_posenet_truescale"] = round(float(np.median(ets_list)), 2)
        pn_native_meds.append((float(np.median(en_list)), float(np.mean(np.array(en_list) < 10.0))))
        pn_ts_meds.append((float(np.median(ets_list)), float(np.mean(np.array(ets_list) < 10.0))))
        print(f"{r['sequence']}: samedata_pn={r['medians']['local_samedata_posenet']:.2f} (ts={r['medians']['local_samedata_posenet_truescale']:.2f}) vs egolite={r['medians']['local_egolite']:.2f}")

ego_arr = np.array([r["medians"]["local_egolite"] for r in ar_d["per_sequence_medians"]])
for key, tups in [("local_samedata_posenet", pn_native_meds), ("local_samedata_posenet_truescale", pn_ts_meds)]:
    arr = np.array([x[0] for x in tups])
    ar_d["table"][key] = {
        "median_px": float(np.median(arr)),
        "mean_px": float(np.mean(arr)),
        "within_10px": float(np.mean([x[1] for x in tups])),
        "wilcoxon_p_vs_egolite": float(wilcoxon(arr, ego_arr).pvalue),
    }
ar_p.write_text(json.dumps(ar_d, indent=2), encoding="utf-8")
print("SameDataPoseNet Sim-AR:", ar_d["table"]["local_samedata_posenet"], ar_d["table"]["local_samedata_posenet_truescale"])
