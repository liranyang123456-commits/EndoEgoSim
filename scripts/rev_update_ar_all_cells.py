"""Update Real StereoMIS AR (`r2_1_stereomis_real_ar_clean.json`) and Real STIR AR (`r2_1_stir_real_ar.json`)
to include BOTH:
  - `samedata_posenet_13M` (trained on EgoLite's exact synthetic + R2S corpus)
  - `droid` on STIR 2024 (`droid_stir20/*_c2w.txt`)
so that every cell across Simulated Abs, Real StereoMIS, and Real STIR in Table 9 has 100% complete coverage!
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import wilcoxon

cv2.setNumThreads(2)
torch.set_num_threads(4)

ROOT = Path(r"D:\ego_motiion_Camera")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_ar_registration as ear  # noqa: E402
import eval_stir as es  # noqa: E402
import rev_eval_stereomis_real_ar as r_smis  # noqa: E402
import rev_run_local_retrained_fast as r_fast  # noqa: E402
from endosim.eval.metrics import load_pose_txt  # noqa: E402
from endosim.lite.data import H_IN, W_IN  # noqa: E402
from endosim.lite.egolite import build_from_args, chain  # noqa: E402
from endosim.pseudoheight import pseudo_height_label  # noqa: E402

OUT_DIR = ROOT / "results" / "revision_vrih"
STIR_ROOT = Path(r"E:\World_Agent_Enoscopy\datasets\STIR\STIRChallenge_2024")


@torch.no_grad()
def update_stir_with_posenet_and_droid() -> None:
    stir_p = OUT_DIR / "r2_1_stir_real_ar.json"
    d_stir = json.loads(stir_p.read_text(encoding="utf-8"))

    ck = torch.load(ROOT / "results/egolite/r_v2/last.pth", map_location="cpu")
    ego_model = build_from_args(ck["args"], pretrained=False).cuda().eval()
    ego_model.load_state_dict(ck["model"])

    pn_mod = r_fast.SameDataPoseNet(pretrained=False).cuda().eval()
    pn_mod.load_state_dict(torch.load(OUT_DIR / "ckpts/samedata_posenet/last.pth", map_location="cpu")["model"])

    WIN_FULL, PATCH_FULL = 32, 15

    for row in d_stir["clips"]:
        cid = row["clip"]
        coll, seq = cid.split("/")
        calib = json.loads(Path(STIR_ROOT, coll, "calib.json").read_text(encoding="utf-8"))
        K_full = np.asarray(calib["leftcameramat"], np.float64)
        clip_dir = STIR_ROOT / coll / "left" / seq
        start_pts = es.centroids(str(clip_dir / "segmentation" / "icgstartseg.png"))
        end_pts = es.centroids(str(clip_dir / "segmentation" / "icgendseg.png"))
        mp4 = glob.glob(str(clip_dir / "frames" / "*.mp4"))[0]
        frames = es.decode(mp4)
        sel = np.unique(np.linspace(0, len(frames) - 1, min(len(frames), 16)).round().astype(int))
        f_sub = [frames[i] for i in sel]
        H0, W0 = f_sub[0].shape[:2]
        sx, sy = W_IN / W0, H_IN / H0

        rgb = [cv2.cvtColor(cv2.resize(f, (W_IN, H_IN), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
               .astype(np.float32) / 255.0 for f in f_sub]
        img_t = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).cuda()
        ph_t = torch.from_numpy(np.stack([pseudo_height_label(r) for r in rgb]))[:, None].cuda()
        K_in = np.array([[K_full[0, 0] * sx, 0, K_full[0, 2] * sx],
                         [0, K_full[1, 1] * sy, K_full[1, 2] * sy], [0, 0, 1]], np.float32)
        Kt = torch.from_numpy(K_in).cuda()

        o0 = ego_model(img_t[:1], img_t[1:2], Kt[None], ph_t[:1], ph_t[1:2])
        z0_map = o0["Z"][0:1].float().cpu()
        Rs_e, ts_e = [], []
        for k in range(len(f_sub) - 1):
            oe = ego_model(img_t[k:k + 1], img_t[k + 1:k + 2], Kt[None], ph_t[k:k + 1], ph_t[k + 1:k + 2])
            Rs_e.append(oe["R"].float().cpu()); ts_e.append(oe["t"].float().cpu())
        T_ego = chain(torch.cat(Rs_e, 0), torch.cat(ts_e, 0), torch.ones(len(f_sub) - 1)).double().numpy()
        T_es_ego = np.linalg.inv(T_ego[-1]) @ T_ego[0]
        ego_norm = float(np.linalg.norm(T_es_ego[:3, 3]))

        p_in = start_pts * np.array([sx, sy])
        z_pts = es.sample(z0_map, torch.from_numpy(p_in))[:, 0].double().numpy()

        # 1) Same-Data 13M PoseNet (native scale & ego-norm matched)
        Rp, tp = pn_mod(img_t[:-1], img_t[1:])
        T_pn = chain(Rp.float().cpu(), tp.float().cpu(), torch.ones(len(f_sub) - 1)).double().numpy()
        T_es_pn = np.linalg.inv(T_pn[-1]) @ T_pn[0]
        pred_pn = es.reproject(p_in, z_pts, T_es_pn, K_in) / np.array([sx, sy])

        # 2) DROID-SLAM trajectory from WSL2 (`rev_run_droid_stir20.py` saves c2w poses directly!)
        tag = cid.replace("/", "_")
        dr_p = OUT_DIR / "droid_stir20" / f"{tag}_c2w.txt"
        T_dr = np.loadtxt(str(dr_p)).reshape(-1, 4, 4)
        T_es_dr = np.linalg.inv(T_dr[-1]) @ T_dr[0]
        if np.linalg.norm(T_es_dr[:3, 3]) > 1e-8:
            T_es_dr[:3, 3] = T_es_dr[:3, 3] / np.linalg.norm(T_es_dr[:3, 3]) * ego_norm
        pred_dr = es.reproject(p_in, z_pts, T_es_dr, K_in) / np.array([sx, sy])

        g_first = cv2.cvtColor(f_sub[0], cv2.COLOR_BGR2GRAY)
        g_last = cv2.cvtColor(f_sub[-1], cv2.COLOR_BGR2GRAY)

        def refine_local(prior_pts):
            out_pts = prior_pts.copy()
            for ip, ((x0, y0), (xp, yp)) in enumerate(zip(start_pts, prior_pts)):
                x0i, y0i, x1, y1 = int(round(x0)), int(round(y0)), int(round(xp)), int(round(yp))
                if not (PATCH_FULL <= x0i < W0 - PATCH_FULL and PATCH_FULL <= y0i < H0 - PATCH_FULL):
                    continue
                x1c = int(np.clip(x1, WIN_FULL + PATCH_FULL, W0 - WIN_FULL - PATCH_FULL - 1))
                y1c = int(np.clip(y1, WIN_FULL + PATCH_FULL, H0 - WIN_FULL - PATCH_FULL - 1))
                templ = g_first[y0i - PATCH_FULL:y0i + PATCH_FULL + 1, x0i - PATCH_FULL:x0i + PATCH_FULL + 1]
                srch = g_last[y1c - WIN_FULL - PATCH_FULL:y1c + WIN_FULL + PATCH_FULL + 1,
                              x1c - WIN_FULL - PATCH_FULL:x1c + WIN_FULL + PATCH_FULL + 1]
                res = cv2.matchTemplate(srch, templ, cv2.TM_CCOEFF_NORMED)
                _, mv, _, loc = cv2.minMaxLoc(res)
                if mv >= 0.35:
                    out_pts[ip] = (x1c - WIN_FULL + loc[0], y1c - WIN_FULL + loc[1])
            return out_pts

        for m, p_pts in [("samedata_posenet", pred_pn), ("droid", pred_dr)]:
            reg_pts = refine_local(p_pts)
            pe = es.nearest_err(p_pts, end_pts)
            re_err = es.nearest_err(reg_pts, end_pts)
            row[f"{m}_prior_px"] = float(np.median(pe))
            row[f"{m}_reg_px"] = float(np.median(re_err))
            row[f"{m}_within25px"] = float(np.mean(re_err < 25.0))

    ego_arr = np.array([r["egolite_reg_px"] for r in d_stir["clips"]])
    for m in ["samedata_posenet", "droid"]:
        arr = np.array([r[f"{m}_reg_px"] for r in d_stir["clips"]])
        parr = np.array([r[f"{m}_prior_px"] for r in d_stir["clips"]])
        w25 = np.array([r[f"{m}_within25px"] for r in d_stir["clips"]])
        d_stir["summary"][m] = {
            "prior_median_px": float(np.median(parr)),
            "reg_median_px": float(np.median(arr)),
            "reg_mean_px": float(np.mean(arr)),
            "within_25px": float(np.mean(w25)),
            "wilcoxon_p_vs_egolite": float(wilcoxon(arr, ego_arr).pvalue),
        }
    stir_p.write_text(json.dumps(d_stir, indent=2), encoding="utf-8")
    print("STIR updated:", {k: (round(v["reg_median_px"], 1), round(v["within_25px"] * 100, 1), v["wilcoxon_p_vs_egolite"])
                            for k, v in d_stir["summary"].items()})


def update_stereomis_ar_with_samedata_posenet() -> None:
    """Re-run StereoMIS AR evaluation adding `samedata_posenet` from `posenet_stereomis_full`."""
    win_px = 18
    smis_refs = [ln.strip() for ln in (ROOT / "lists/stereomis.txt").read_text().splitlines() if ln.strip()]
    pn_dir = OUT_DIR / "posenet_stereomis_full"
    ego_dir = ROOT / "results/sota/egolite_r_v2_stereomis_full"

    seq_pn_c, seq_pn_50, seq_ego_c = [], [], []
    w10_pn_c, w10_pn_50 = [], []

    for ref_d in smis_refs:
        sid = os.path.basename(ref_d)
        meta = json.loads(Path(ref_d, "meta.json").read_text(encoding="utf-8"))
        cidx = json.loads(Path(ref_d, "color_index.json").read_text(encoding="utf-8"))
        src_dir = meta["src_path"]
        mp4s = glob.glob(os.path.join(src_dir, "*.mp4"))
        ini_p = os.path.join(src_dir, "StereoCalibration.ini")
        if not mp4s or not os.path.isfile(ini_p):
            continue
        K_full, cx_r_full, baseline_mm = r_smis.parse_stereomis_calib(ini_p)
        sc = 0.5
        K = K_full.copy(); K[0] *= sc; K[1] *= sc
        cx_l, cx_r = K[0, 2], cx_r_full * sc
        offset = int(cidx.get("frame_offset", 0))
        gt_poses = load_pose_txt(os.path.join(ref_d, "pose_c2w.txt"))
        n_pose = len(gt_poses)

        tp_ego = ego_dir / f"{sid}_est_c2w.txt"
        tp_pn = pn_dir / f"{sid}_c2w.txt"
        if not (tp_ego.is_file() and tp_pn.is_file()):
            continue
        tr_ego = np.loadtxt(str(tp_ego)).reshape(-1, 4, 4)
        tr_pn = np.loadtxt(str(tp_pn)).reshape(-1, 4, 4)
        if len(tr_ego) != n_pose or len(tr_pn) != n_pose:
            continue

        gt_steps = np.linalg.norm(gt_poses[1:, :3, 3] - gt_poses[:-1, :3, 3], axis=1)
        moving_all = np.where(gt_steps > 0.15)[0]
        if len(moving_all) < 15:
            continue
        calib_idx = moving_all[:min(40, len(moving_all) // 4)]
        test_start = int(calib_idx[-1]) + 30

        sc_ego = float(np.median(gt_steps[calib_idx] / np.maximum(np.linalg.norm(tr_ego[calib_idx + 1, :3, 3] - tr_ego[calib_idx, :3, 3], axis=1), 1e-6)))
        sc_pn = float(np.median(gt_steps[calib_idx] / np.maximum(np.linalg.norm(tr_pn[calib_idx + 1, :3, 3] - tr_pn[calib_idx, :3, 3], axis=1), 1e-6)))

        step_gap = 25
        disp_gap = np.linalg.norm(gt_poses[test_start + step_gap:, :3, 3] - gt_poses[test_start:-step_gap, :3, 3], axis=1)
        active = np.where((disp_gap >= 3.0) & (disp_gap <= 25.0))[0] + test_start
        if len(active) < 6:
            step_gap = 40
            disp_gap = np.linalg.norm(gt_poses[test_start + step_gap:, :3, 3] - gt_poses[test_start:-step_gap, :3, 3], axis=1)
            active = np.where((disp_gap >= 3.0) & (disp_gap <= 25.0))[0] + test_start
        if len(active) < 6:
            continue

        cand_starts = active[np.linspace(0, len(active) - 1, min(20, len(active)), dtype=int)]
        need_raw = {offset + int(t) for t in cand_starts} | {offset + int(t) + step_gap for t in cand_starts}
        decoded = r_smis.decode_needed_stereo_frames(mp4s[0], need_raw, scale=sc)

        e_ego_c, e_pn_c, e_pn_50 = [], [], []
        for t in cand_starts:
            t = int(t)
            r0_idx, r1_idx = offset + t, offset + t + step_gap
            if r0_idx not in decoded or r1_idx not in decoded:
                continue
            L0, R0_im = decoded[r0_idx]; L1, R1_im = decoded[r1_idx]
            gL0, gR0 = cv2.cvtColor(L0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(R0_im, cv2.COLOR_BGR2GRAY)
            gL1 = cv2.cvtColor(L1, cv2.COLOR_BGR2GRAY)
            roi = np.zeros_like(gL0); roi[70:440, 90:550] = 255
            p_l0 = cv2.goodFeaturesToTrack(gL0, 90, 0.01, 10, mask=roi)
            if p_l0 is None or len(p_l0) < 20:
                continue
            p_l0 = p_l0.reshape(-1, 2)
            init_r0 = p_l0.copy().astype(np.float32); init_r0[:, 0] += float(cx_r - cx_l - 35.0)
            p_r0, st_lr, _ = cv2.calcOpticalFlowPyrLK(gL0, gR0, p_l0.astype(np.float32), init_r0, winSize=(25, 25), maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            p_l0_bk, st_rl, _ = cv2.calcOpticalFlowPyrLK(gR0, gL0, p_r0, p_l0.astype(np.float32), winSize=(25, 25), maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            disp0 = (p_l0[:, 0] - cx_l) - (p_r0[:, 0] - cx_r)
            epi0 = np.abs(p_l0[:, 1] - p_r0[:, 1]); cyc0 = np.linalg.norm(p_l0 - p_l0_bk, axis=1)
            ok0 = (st_lr.ravel() == 1) & (st_rl.ravel() == 1) & (disp0 > 5.0) & (disp0 < 120.0) & (epi0 < 2.0) & (cyc0 < 1.2)
            if ok0.sum() < 14:
                continue
            p_s = p_l0[ok0].astype(np.float32)
            p_l1, st_f, _ = cv2.calcOpticalFlowPyrLK(gL0, gL1, p_s, None, winSize=(31, 31), maxLevel=4)
            p_s_bk, st_b, _ = cv2.calcOpticalFlowPyrLK(gL1, gL0, p_l1, None, winSize=(31, 31), maxLevel=4)
            fb_err = np.linalg.norm(p_s - p_s_bk, axis=1)
            in_im = (p_l1[:, 0] > 25) & (p_l1[:, 0] < 615) & (p_l1[:, 1] > 25) & (p_l1[:, 1] < 487)
            ok1 = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (fb_err < 1.2) & in_im
            if ok1.sum() < 14:
                continue
            src = p_s[ok1].astype(np.float64); truth = p_l1[ok1].astype(np.float64)
            if ear.px(src, truth) < win_px + 4.0:
                continue
            d0 = disp0[ok0][ok1].astype(np.float64); z0 = K[0, 0] * baseline_mm / d0
            Xm = np.c_[(src[:, 0] - K[0, 2]) * z0 / K[0, 0], (src[:, 1] - K[1, 2]) * z0 / K[1, 1], z0]
            Re_gt, te_gt = ear.cam_static(gt_poses[t], gt_poses[t + step_gap])
            _, zp_gt = ear.project_c(K, (Re_gt @ Xm.T).T + te_gt)
            if np.median(zp_gt) < 5.0:
                continue
            vis = np.ones(len(Xm), dtype=bool); R_id, t_zero = np.eye(3), np.zeros(3)

            Re_e, te_e = ear.cam_static(tr_ego[t], tr_ego[t + step_gap])
            uv_ec, z_ec = ear.project_c(K, (Re_e @ Xm.T).T + te_e * sc_ego)
            pr_ec = (src, (R_id, t_zero)) if np.median(z_ec) < 2 else (uv_ec, (Re_e, te_e * sc_ego))
            e_ego_c.append(ear.px(ear.register_local(gL0, gL1, src, pr_ec[0], Xm, K, pr_ec[1], vis, win_px), truth))

            Re_p, te_p = ear.cam_static(tr_pn[t], tr_pn[t + step_gap])
            uv_pc, z_pc = ear.project_c(K, (Re_p @ Xm.T).T + te_p * sc_pn)
            pr_pc = (src, (R_id, t_zero)) if np.median(z_pc) < 2 else (uv_pc, (Re_p, te_p * sc_pn))
            e_pn_c.append(ear.px(ear.register_local(gL0, gL1, src, pr_pc[0], Xm, K, pr_pc[1], vis, win_px), truth))

            uv_p50, z_p50 = ear.project_c(K, (Re_p @ Xm.T).T + te_p * 50.0)
            pr_p50 = (src, (R_id, t_zero)) if np.median(z_p50) < 2 else (uv_p50, (Re_p, te_p * 50.0))
            e_pn_50.append(ear.px(ear.register_local(gL0, gL1, src, pr_p50[0], Xm, K, pr_p50[1], vis, win_px), truth))

        if len(e_ego_c) >= 3:
            seq_ego_c.append(float(np.median(e_ego_c)))
            seq_pn_c.append(float(np.median(e_pn_c)))
            seq_pn_50.append(float(np.median(e_pn_50)))
            w10_pn_c.append(float(np.mean(np.array(e_pn_c) < 10.0)))
            w10_pn_50.append(float(np.mean(np.array(e_pn_50) < 10.0)))

    smis_p = OUT_DIR / "r2_1_stereomis_real_ar_clean.json"
    d_smis = json.loads(smis_p.read_text(encoding="utf-8"))
    d_smis["table"]["local_samedata_posenet_causal"] = {
        "n_seq": len(seq_pn_c),
        "median_px": float(np.median(seq_pn_c)),
        "mean_px": float(np.mean(seq_pn_c)),
        "within_10px": float(np.mean(w10_pn_c)),
        "wilcoxon_p_vs_egolite": float(wilcoxon(np.array(seq_pn_c), np.array(seq_ego_c)).pvalue),
    }
    d_smis["table"]["local_samedata_posenet_fixed50"] = {
        "n_seq": len(seq_pn_50),
        "median_px": float(np.median(seq_pn_50)),
        "mean_px": float(np.mean(seq_pn_50)),
        "within_10px": float(np.mean(w10_pn_50)),
        "wilcoxon_p_vs_egolite": float(wilcoxon(np.array(seq_pn_50), np.array(seq_ego_c)).pvalue),
    }
    smis_p.write_text(json.dumps(d_smis, indent=2), encoding="utf-8")
    print("StereoMIS AR updated with Same-Data PoseNet:", d_smis["table"]["local_samedata_posenet_causal"], d_smis["table"]["local_samedata_posenet_fixed50"])


if __name__ == "__main__":
    update_stir_with_posenet_and_droid()
    update_stereomis_ar_with_samedata_posenet()
