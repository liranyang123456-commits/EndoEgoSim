"""Seeded AR benchmark updates (P0-1, P0-8, and STIR/StereoMIS/Sim-AR consistency fixes).

1. Updates STIR AR (`r2_1_stir_real_ar.json`) with FixedSameDataPoseNet WITH proper ego_norm rescaling.
2. Updates StereoMIS AR (`r2_1_stereomis_real_ar_clean.json`) with FixedSameDataPoseNet (causal, fixed50,
   truescale) AND strictly causal ORB (`local_orb_causal`), plus Wilcoxon p-values against both
   `local_egolite_fixed50` and `local_egolite_causal`.
3. Updates Sim-AR (`r2_2_sim_ar_expanded.json`) with FixedSameDataPoseNet (native & true-scaled) under
   seeded `cv2.setRNGSeed(0)`.
4. Regenerates `fig_real_ar_registration.png` (fixing panel-5 white border).
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_ar_registration as ear  # noqa: E402
import eval_stir as es  # noqa: E402
import rev_eval_stereomis_real_ar as r_smis  # noqa: E402
from endosim.eval.metrics import load_pose_txt  # noqa: E402
from endosim.lite.data import H_IN, W_IN, load_frames  # noqa: E402
from endosim.lite.egolite import build_from_args, chain  # noqa: E402
from endosim.pseudoheight import pseudo_height_label  # noqa: E402
from rev_fix_stage1_posenet import FixedSameDataPoseNet  # noqa: E402

OUT_DIR = ROOT / "results" / "revision_vrih"
STIR_ROOT = Path(r"E:\World_Agent_Enoscopy\datasets\STIR\STIRChallenge_2024")


@torch.no_grad()
def update_stir(pn_mod: FixedSameDataPoseNet) -> None:
    cv2.setRNGSeed(0)
    stir_p = OUT_DIR / "r2_1_stir_real_ar.json"
    d_stir = json.loads(stir_p.read_text(encoding="utf-8"))

    ck = torch.load(ROOT / "results/egolite/r_v2/last.pth", map_location="cpu", weights_only=False)
    ego_model = build_from_args(ck["args"], pretrained=False).cuda().eval()
    ego_model.load_state_dict(ck["model"])
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
            Rs_e.append(oe["R"].float().cpu())
            ts_e.append(oe["t"].float().cpu())
        T_ego = chain(torch.cat(Rs_e, 0), torch.cat(ts_e, 0), torch.ones(len(f_sub) - 1)).double().numpy()
        T_es_ego = np.linalg.inv(T_ego[-1]) @ T_ego[0]
        ego_norm = float(np.linalg.norm(T_es_ego[:3, 3]))

        p_in = start_pts * np.array([sx, sy])
        z_pts = es.sample(z0_map, torch.from_numpy(p_in))[:, 0].double().numpy()

        Rp, tp = pn_mod(img_t[:-1], img_t[1:])
        T_pn = chain(Rp.float().cpu(), tp.float().cpu(), torch.ones(len(f_sub) - 1)).double().numpy()
        T_es_pn_nat = np.linalg.inv(T_pn[-1]) @ T_pn[0]
        T_es_pn_matched = T_es_pn_nat.copy()
        if np.linalg.norm(T_es_pn_matched[:3, 3]) > 1e-8:
            T_es_pn_matched[:3, 3] = T_es_pn_matched[:3, 3] / np.linalg.norm(T_es_pn_matched[:3, 3]) * ego_norm

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

        for m, T_es in [("samedata_posenet", T_es_pn_matched), ("samedata_posenet_native", T_es_pn_nat)]:
            p_pts = es.reproject(p_in, z_pts, T_es, K_in) / np.array([sx, sy])
            reg_pts = refine_local(p_pts)
            pe = es.nearest_err(p_pts, end_pts)
            re_err = es.nearest_err(reg_pts, end_pts)
            row[f"{m}_prior_px"] = float(np.median(pe))
            row[f"{m}_reg_px"] = float(np.median(re_err))
            row[f"{m}_within25px"] = float(np.mean(re_err < 25.0))

    ego_arr = np.array([r["egolite_reg_px"] for r in d_stir["clips"]])
    for m in ["samedata_posenet", "samedata_posenet_native"]:
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
    print("[STIR FixedPoseNet] matched:", d_stir["summary"]["samedata_posenet"])
    print("[STIR FixedPoseNet] native:", d_stir["summary"]["samedata_posenet_native"])


def update_stereomis() -> None:
    """Update StereoMIS AR with FixedSameDataPoseNet (causal, fixed50, truescale) & causal ORB."""
    cv2.setRNGSeed(0)
    win_px = 18
    smis_refs = [ln.strip() for ln in (ROOT / "lists/stereomis.txt").read_text().splitlines() if ln.strip()]
    pn_dir = OUT_DIR / "posenet_stereomis_full_fixed"
    ego_dir = ROOT / "results/sota/egolite_r_v2_stereomis_full"

    seq_vals = {k: [] for k in ("ego_50", "ego_c", "ego_ts", "pn_c", "pn_50", "pn_ts", "orb_c")}
    w10_vals = {k: [] for k in ("pn_c", "pn_50", "pn_ts", "orb_c")}

    for ref_d in smis_refs:
        cv2.setRNGSeed(0)
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
        K = K_full.copy()
        K[0] *= sc
        K[1] *= sc
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
        pn_steps = np.linalg.norm(tr_pn[calib_idx + 1, :3, 3] - tr_pn[calib_idx, :3, 3], axis=1)
        ok_cal = pn_steps > 1e-4
        sc_pn = float(np.median(gt_steps[calib_idx][ok_cal] / pn_steps[ok_cal])) if ok_cal.sum() >= 5 else 50.0

        step_gap = 25
        disp_gap = np.linalg.norm(gt_poses[test_start + step_gap:, :3, 3] - gt_poses[test_start:-step_gap, :3, 3], axis=1)
        active = np.where((disp_gap >= 3.0) & (disp_gap <= 25.0))[0] + test_start
        if len(active) < 6:
            step_gap = 40
            disp_gap = np.linalg.norm(gt_poses[test_start + step_gap:, :3, 3] - gt_poses[test_start:-step_gap, :3, 3], axis=1)
            active = np.where((disp_gap >= 3.0) & (disp_gap <= 25.0))[0] + test_start
        if len(active) < 6:
            continue

        # Strictly causal step-scale estimate for ORB (median GT displacement over same step_gap in calibration prefix)
        cal_end = min(int(calib_idx[-1]) + step_gap, test_start)
        cal_gaps = np.linalg.norm(gt_poses[step_gap:cal_end, :3, 3] - gt_poses[:max(1, cal_end - step_gap), :3, 3], axis=1) if cal_end > step_gap else np.array([float(np.median(gt_steps[calib_idx]) * step_gap)])
        orb_causal_mag = float(np.median(cal_gaps[cal_gaps > 0.5])) if (cal_gaps > 0.5).any() else float(np.median(gt_steps[calib_idx]) * step_gap)

        cand_starts = active[np.linspace(0, len(active) - 1, min(20, len(active)), dtype=int)]
        need_raw = {offset + int(t) for t in cand_starts} | {offset + int(t) + step_gap for t in cand_starts}
        decoded = r_smis.decode_needed_stereo_frames(mp4s[0], need_raw, scale=sc)

        ep = {k: [] for k in seq_vals}
        for t in cand_starts:
            t = int(t)
            r0_idx, r1_idx = offset + t, offset + t + step_gap
            if r0_idx not in decoded or r1_idx not in decoded:
                continue
            L0, R0_im = decoded[r0_idx]
            L1, _ = decoded[r1_idx]
            gL0, gR0 = cv2.cvtColor(L0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(R0_im, cv2.COLOR_BGR2GRAY)
            gL1 = cv2.cvtColor(L1, cv2.COLOR_BGR2GRAY)
            roi = np.zeros_like(gL0)
            roi[70:440, 90:550] = 255
            p_l0 = cv2.goodFeaturesToTrack(gL0, 90, 0.01, 10, mask=roi)
            if p_l0 is None or len(p_l0) < 20:
                continue
            p_l0 = p_l0.reshape(-1, 2)
            init_r0 = p_l0.copy().astype(np.float32)
            init_r0[:, 0] += float(cx_r - cx_l - 35.0)
            p_r0, st_lr, _ = cv2.calcOpticalFlowPyrLK(gL0, gR0, p_l0.astype(np.float32), init_r0, winSize=(25, 25), maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            p_l0_bk, st_rl, _ = cv2.calcOpticalFlowPyrLK(gR0, gL0, p_r0, p_l0.astype(np.float32), winSize=(25, 25), maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            disp0 = (p_l0[:, 0] - cx_l) - (p_r0[:, 0] - cx_r)
            epi0 = np.abs(p_l0[:, 1] - p_r0[:, 1])
            cyc0 = np.linalg.norm(p_l0 - p_l0_bk, axis=1)
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
            src = p_s[ok1].astype(np.float64)
            truth = p_l1[ok1].astype(np.float64)
            if ear.px(src, truth) < win_px + 4.0:
                continue
            d0 = disp0[ok0][ok1].astype(np.float64)
            z0 = K[0, 0] * baseline_mm / d0
            Xm = np.c_[(src[:, 0] - K[0, 2]) * z0 / K[0, 0], (src[:, 1] - K[1, 2]) * z0 / K[1, 1], z0]
            Re_gt, te_gt = ear.cam_static(gt_poses[t], gt_poses[t + step_gap])
            _, zp_gt = ear.project_c(K, (Re_gt @ Xm.T).T + te_gt)
            if np.median(zp_gt) < 5.0:
                continue
            vis = np.ones(len(Xm), dtype=bool)
            R_id, t_zero = np.eye(3), np.zeros(3)
            mag_gt = float(np.linalg.norm(te_gt))

            def run_prior(Rm, tm):
                uv, zp = ear.project_c(K, (Rm @ Xm.T).T + tm)
                pr = (src, (R_id, t_zero)) if np.median(zp) < 2 else (uv, (Rm, tm))
                return ear.px(ear.register_local(gL0, gL1, src, pr[0], Xm, K, pr[1], vis, win_px), truth)

            Re_e, te_e = ear.cam_static(tr_ego[t], tr_ego[t + step_gap])
            ep["ego_50"].append(run_prior(Re_e, te_e * 50.0))
            ep["ego_c"].append(run_prior(Re_e, te_e * sc_ego))
            ep["ego_ts"].append(run_prior(Re_e, te_e / max(np.linalg.norm(te_e), 1e-8) * mag_gt))

            Re_p, te_p = ear.cam_static(tr_pn[t], tr_pn[t + step_gap])
            ep["pn_c"].append(run_prior(Re_p, np.clip(te_p * sc_pn, -50.0, 50.0)))
            ep["pn_50"].append(run_prior(Re_p, te_p * 50.0))
            ep["pn_ts"].append(run_prior(Re_p, te_p / max(np.linalg.norm(te_p), 1e-8) * mag_gt))

            # Causal ORB + essential matrix
            orb_res = ear.orb_motion(gL0, gL1, K, orb_causal_mag)
            if orb_res is None:
                ep["orb_c"].append(run_prior(R_id, t_zero))
            else:
                ep["orb_c"].append(run_prior(orb_res[0], orb_res[1]))

        if len(ep["ego_c"]) >= 3:
            for k in seq_vals:
                seq_vals[k].append(float(np.median(ep[k])))
            for k in w10_vals:
                w10_vals[k].append(float(np.mean(np.array(ep[k]) < 10.0)))

    smis_p = OUT_DIR / "r2_1_stereomis_real_ar_clean.json"
    d_smis = json.loads(smis_p.read_text(encoding="utf-8"))
    for jkey, skey in [
        ("local_samedata_posenet_causal", "pn_c"),
        ("local_samedata_posenet_fixed50", "pn_50"),
        ("local_samedata_posenet_truescale", "pn_ts"),
        ("local_orb_causal", "orb_c"),
    ]:
        arr = np.array(seq_vals[skey])
        ref_arr = np.array(seq_vals["ego_ts"] if jkey.endswith("_truescale") else seq_vals["ego_c"])
        d_smis["table"][jkey] = {
            "n_seq": len(arr),
            "median_px": float(np.median(arr)),
            "mean_px": float(np.mean(arr)),
            "within_10px": float(np.mean(w10_vals[skey])),
            "wilcoxon_p_vs_egolite": float(wilcoxon(arr, ref_arr).pvalue),
            "wilcoxon_p_vs_egolite_fixed50": float(wilcoxon(arr, np.array(seq_vals["ego_50"])).pvalue),
        }
    smis_p.write_text(json.dumps(d_smis, indent=2), encoding="utf-8")
    for k in ("local_samedata_posenet_causal", "local_samedata_posenet_fixed50", "local_samedata_posenet_truescale", "local_orb_causal"):
        print(f"[StereoMIS AR] {k}: {d_smis['table'][k]}")


@torch.no_grad()
def update_sim_ar(pn_mod: FixedSameDataPoseNet) -> None:
    cv2.setRNGSeed(0)
    ar_p = OUT_DIR / "r2_2_sim_ar_expanded.json"
    ar_d = json.loads(ar_p.read_text(encoding="utf-8"))
    pn_nat, pn_ts = [], []
    for r in ar_d["per_sequence_medians"]:
        cv2.setRNGSeed(0)
        seq = os.path.join("sim_data_abs/test", r["sequence"])
        K, pose, traj = ear.load_clip(seq)
        Xm = traj[0]
        s = r["stride"]
        en, ets = [], []
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
            Re, te = ear.cam_static(pose[t], pose[t + s])
            mag = float(np.linalg.norm(te))
            imgs, _, _ = load_frames([pa, pb])
            im_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
            Rp, tp = pn_mod(im_t[:1], im_t[1:])
            Rn = Rp[0].double().cpu().numpy()
            tn_nat = tp[0].double().cpu().numpy() * 50.0
            tn_ts = tn_nat / max(np.linalg.norm(tn_nat), 1e-8) * mag
            for tn_cur, lst in [(tn_nat, en), (tn_ts, ets)]:
                uvp, zp = ear.project_c(K, (Rn @ Xc.T).T + tn_cur)
                pred, guess = ((src, (R0, t0)) if np.median(zp) < 1 else (uvp, (Rn @ R0, Rn @ t0 + tn_cur)))
                got = ear.register_local(g0, g1, src, pred, Xm, K, guess, vis, 18)
                lst.append(ear.px(got[vis], truth[vis]))
        r["medians"]["local_samedata_posenet"] = round(float(np.median(en)), 2)
        r["medians"]["local_samedata_posenet_truescale"] = round(float(np.median(ets)), 2)
        pn_nat.append((float(np.median(en)), float(np.mean(np.array(en) < 10.0))))
        pn_ts.append((float(np.median(ets)), float(np.mean(np.array(ets) < 10.0))))
    ego_arr = np.array([r["medians"]["local_egolite"] for r in ar_d["per_sequence_medians"]])
    for key, tups in [("local_samedata_posenet", pn_nat), ("local_samedata_posenet_truescale", pn_ts)]:
        meds = np.array([x[0] for x in tups])
        w10 = np.array([x[1] for x in tups])
        ar_d["table"][key] = {
            "median_px": float(np.median(meds)),
            "mean_px": float(np.mean(meds)),
            "within_10px": float(np.mean(w10)),
            "wilcoxon_p_vs_egolite": float(wilcoxon(meds, ego_arr).pvalue),
        }
        print(f"[Sim AR] {key}: {ar_d['table'][key]}")
    ar_p.write_text(json.dumps(ar_d, indent=2), encoding="utf-8")


def main():
    if "--stereomis-only" not in sys.argv:
        pn_mod = FixedSameDataPoseNet(pretrained=False).cuda().eval()
        pn_mod.load_state_dict(torch.load(OUT_DIR / "ckpts/samedata_posenet_fixed/last.pth", map_location="cpu", weights_only=False)["model"])
        update_sim_ar(pn_mod)
        update_stir(pn_mod)
    update_stereomis()
    print("[Fig 17] Regenerating fig_real_ar_registration.png (fixed panel-5 bottom border) ...", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/make_real_ar_figure.py")], check=True)
    print("DONE Stage 4!", flush=True)


if __name__ == "__main__":
    main()
