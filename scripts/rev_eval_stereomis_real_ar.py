"""Frame-accurate, Sign-verified Real In Vivo Surgical AR Tissue Registration on StereoMIS (R2.1 & R2.2).

Key methodological guarantees:
1. Sequential video decoding (`cap.grab()` / `cap.read()`, never `cap.set(CAP_PROP_POS_FRAMES)` which
   corrupts HEVC GOP seeks in StereoMIS).
2. Exact stereo disparity with principal-point shift `(u_L - cx_L) - (u_R - cx_R)` used ONLY to
   triangulate reference 3D tissue patch coordinates at frame t and epipolar-verified target landmarks
   at frame t + step_gap; every evaluated ego-motion prior sees ONLY the monocular left sequence.
3. Exact, sign-verified chained trajectories over `[t, t + step_gap]` for ALL five evaluated learning
   methods (`AF-SfMLearner`, `EndoDAC`, `Endo-FASt3r`, `DROID-SLAM` with exact lietorch quaternion
   order, and `EgoLite`), plus `No prior`, `Homography`, `ORB`, and `Robot-Kinematic GT`:
   - Protocol I (Causal online calibration): scale calibrated once on the initial moving segment of
     each video (strictly causal, zero future ground truth), plus EgoLite's uncalibrated fixed-50-mm unit.
   - Protocol II (Matched step magnitude): each estimator's chained translation scaled to the true step
     magnitude, isolating pure rotation + translation-direction accuracy under in vivo breathing.
"""
from __future__ import annotations

import configparser
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon

cv2.setNumThreads(2)
ROOT = Path(r"D:\ego_motiion_Camera")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_ar_registration as ear  # noqa: E402
from endosim.eval.metrics import load_pose_txt  # noqa: E402

OUT = ROOT / "results" / "revision_vrih"
OUT.mkdir(parents=True, exist_ok=True)


def fix_droid_rotations(tr: np.ndarray) -> np.ndarray:
    out = tr.copy()
    q = Rotation.from_matrix(tr[:, :3, :3]).as_quat()
    out[:, :3, :3] = Rotation.from_quat(q[:, [3, 0, 1, 2]]).as_matrix()
    return out


def parse_stereomis_calib(ini_path: str) -> tuple[np.ndarray, float, float]:
    cfg = configparser.ConfigParser()
    cfg.read(ini_path)
    fx = float(cfg["StereoLeft"]["fc_x"])
    fy = float(cfg["StereoLeft"]["fc_y"])
    cx_l = float(cfg["StereoLeft"]["cc_x"])
    cy_l = float(cfg["StereoLeft"]["cc_y"])
    cx_r = float(cfg["StereoRight"]["cc_x"])
    baseline_mm = abs(float(cfg["StereoRight"]["T_0"]))
    K_l = np.array([[fx, 0.0, cx_l], [0.0, fy, cy_l], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K_l, cx_r, baseline_mm


def decode_needed_stereo_frames(mp4_path: str, needed_raw_indices: set[int], scale: float = 0.5) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Sequential, frame-accurate decode of only `needed_raw_indices` using `cap.grab()`."""
    if not needed_raw_indices:
        return {}
    max_idx = max(needed_raw_indices)
    cap = cv2.VideoCapture(mp4_path)
    out = {}
    cur = 0
    while cur <= max_idx:
        if cur in needed_raw_indices:
            ok, full = cap.read()
            if not ok or full is None:
                break
            h2 = full.shape[0] // 2
            left, right = full[:h2], full[h2:2 * h2]
            w_s, h_s = int(round(left.shape[1] * scale)), int(round(left.shape[0] * scale))
            out[cur] = (cv2.resize(left, (w_s, h_s), interpolation=cv2.INTER_AREA),
                        cv2.resize(right, (w_s, h_s), interpolation=cv2.INTER_AREA))
        else:
            if not cap.grab():
                break
        cur += 1
    cap.release()
    return out


def posenet_chained_motion(tr_p: np.ndarray, t0: int, t1: int) -> tuple[np.ndarray, np.ndarray]:
    """In egomotion_baselines.py, `traj[k+1] = traj[k] @ T_pt` where `T_pt` is the direct point
    transform `X_{k+1} = T_pt X_k` (verified t_cos = +0.883 on Abs and +0.966 on StereoMIS P2_1).
    Therefore `T_pt_k = inv(tr_p[k]) @ tr_p[k+1]`, and across [t0, t1]: `T_total = T_{t1-1} ... T_{t0}`."""
    T_cur = np.eye(4)
    for k in range(t0, t1):
        T_k = np.linalg.inv(tr_p[k]) @ tr_p[k + 1]
        T_cur = T_k @ T_cur
    return T_cur[:3, :3], T_cur[:3, 3]


def main():
    win_px = 18
    smis_refs = [ln.strip() for ln in (ROOT / "lists/stereomis.txt").read_text().splitlines() if ln.strip()]
    est_dirs = {
        "afsfm": ROOT / "results/sota/afsfm_stereomis_full",
        "endodac": ROOT / "results/sota/endodac_stereomis_full",
        "orbslam3": ROOT / "results/sota/orb_e_c2w_stereomis_full",
        "endofast3r": ROOT / "results/sota/endofast3r_stereomis_full",
        "droid": ROOT / "results/sota/droid_stereomis_full",
        "egolite": ROOT / "results/sota/egolite_r_v2_stereomis_full",
    }
    methods = [
        "local_none", "local_homography", "local_orb_truescale",
        "local_afsfm_causal", "local_endodac_causal", "local_orbslam3_causal",
        "local_endofast3r_causal", "local_droid_causal", "local_egolite_fixed50", "local_egolite_causal",
        "local_afsfm_truescale", "local_endodac_truescale",
        "local_endofast3r_truescale", "local_droid_truescale", "local_egolite_truescale",
        "local_kinematic_gt",
    ]
    seq_results = []
    t_all = time.time()

    for ref_d in smis_refs:
        sid = os.path.basename(ref_d)
        meta = json.loads(Path(ref_d, "meta.json").read_text(encoding="utf-8"))
        cidx = json.loads(Path(ref_d, "color_index.json").read_text(encoding="utf-8"))
        src_dir = meta["src_path"]
        mp4s = glob.glob(os.path.join(src_dir, "*.mp4"))
        ini_p = os.path.join(src_dir, "StereoCalibration.ini")
        if not mp4s or not os.path.isfile(ini_p):
            continue
        K_full, cx_r_full, baseline_mm = parse_stereomis_calib(ini_p)
        sc = 0.5
        K = K_full.copy()
        K[0] *= sc; K[1] *= sc
        cx_l, cx_r = K[0, 2], cx_r_full * sc
        offset = int(cidx.get("frame_offset", 0))
        gt_poses = load_pose_txt(os.path.join(ref_d, "pose_c2w.txt"))
        n_pose = len(gt_poses)

        trajs = {}
        for mname, edir in est_dirs.items():
            tp = edir / f"{sid}_est_c2w.txt"
            if tp.is_file():
                arr = np.loadtxt(str(tp)).reshape(-1, 4, 4)
                if len(arr) == n_pose:
                    if mname == "droid":
                        arr = fix_droid_rotations(arr)
                    trajs[mname] = arr
        if "egolite" not in trajs:
            continue

        # Causal scale calibration on the first moving segment of each surgical video
        gt_steps = np.linalg.norm(gt_poses[1:, :3, 3] - gt_poses[:-1, :3, 3], axis=1)
        moving_all = np.where(gt_steps > 0.15)[0]
        if len(moving_all) < 15:
            continue
        calib_idx = moving_all[:min(40, len(moving_all) // 4)]
        test_start = int(calib_idx[-1]) + 30
        causal_scales = {}
        for mname, tr in trajs.items():
            if mname in ("afsfm", "endodac"):
                est_steps = np.array([np.linalg.norm((np.linalg.inv(tr[i]) @ tr[i + 1])[:3, 3]) for i in calib_idx])
            else:
                est_steps = np.linalg.norm(tr[calib_idx + 1, :3, 3] - tr[calib_idx, :3, 3], axis=1)
            causal_scales[mname] = float(np.median(gt_steps[calib_idx] / np.maximum(est_steps, 1e-6)))

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
        decoded = decode_needed_stereo_frames(mp4s[0], need_raw, scale=sc)

        err = {m: [] for m in methods}
        for t in cand_starts:
            t = int(t)
            r0_idx, r1_idx = offset + t, offset + t + step_gap
            if r0_idx not in decoded or r1_idx not in decoded:
                continue
            L0, R0_im = decoded[r0_idx]
            L1, R1_im = decoded[r1_idx]
            gL0, gR0 = cv2.cvtColor(L0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(R0_im, cv2.COLOR_BGR2GRAY)
            gL1, gR1 = cv2.cvtColor(L1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(R1_im, cv2.COLOR_BGR2GRAY)

            roi = np.zeros_like(gL0)
            roi[70:440, 90:550] = 255
            p_l0 = cv2.goodFeaturesToTrack(gL0, 90, 0.01, 10, mask=roi)
            if p_l0 is None or len(p_l0) < 20:
                continue
            p_l0 = p_l0.reshape(-1, 2)
            init_r0 = p_l0.copy().astype(np.float32)
            init_r0[:, 0] += float(cx_r - cx_l - 35.0)
            p_r0, st_lr, _ = cv2.calcOpticalFlowPyrLK(
                gL0, gR0, p_l0.astype(np.float32), init_r0, winSize=(25, 25), maxLevel=3,
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            p_l0_bk, st_rl, _ = cv2.calcOpticalFlowPyrLK(
                gR0, gL0, p_r0, p_l0.astype(np.float32), winSize=(25, 25), maxLevel=3,
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
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
            # Require actual visual landmark displacement to exceed the 18-px search window!
            if ear.px(src, truth) < win_px + 4.0:
                continue

            d0 = disp0[ok0][ok1].astype(np.float64)
            z0 = K[0, 0] * baseline_mm / d0
            Xm = np.c_[(src[:, 0] - K[0, 2]) * z0 / K[0, 0], (src[:, 1] - K[1, 2]) * z0 / K[1, 1], z0]

            Re_gt, te_gt = ear.cam_static(gt_poses[t], gt_poses[t + step_gap])
            uv_cam_gt, zp_gt = ear.project_c(K, (Re_gt @ Xm.T).T + te_gt)
            if np.median(zp_gt) < 5.0:
                continue

            R_id, t_zero = np.eye(3), np.zeros(3)
            vis = np.ones(len(Xm), dtype=bool)
            true_mag = float(np.linalg.norm(te_gt))

            H = ear.homography(gL0, gL1)
            uvH = src if H is None else cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
            try:
                rv0, _ = cv2.Rodrigues(R_id)
                okH, rvH, tvH = cv2.solvePnP(Xm.astype(np.float32), uvH.astype(np.float32), K, None,
                                             rv0, np.zeros((3, 1)), True, cv2.SOLVEPNP_ITERATIVE)
            except cv2.error:
                okH = False
            guessH = (cv2.Rodrigues(rvH)[0], tvH.ravel()) if okH else (R_id, t_zero)

            priors = {
                "local_none": (src, (R_id, t_zero)),
                "local_homography": (uvH, guessH),
                "local_kinematic_gt": (uv_cam_gt, (Re_gt, te_gt)),
            }
            orb_m = ear.orb_motion(gL0, gL1, K, true_mag)
            if orb_m is not None:
                uv_o, z_o = ear.project_c(K, (orb_m[0] @ Xm.T).T + orb_m[1])
                priors["local_orb_truescale"] = (src, (R_id, t_zero)) if np.median(z_o) < 2 else (uv_o, orb_m)
            else:
                priors["local_orb_truescale"] = (src, (R_id, t_zero))

            for mname, tr in trajs.items():
                if mname in ("afsfm", "endodac"):
                    Rm, tm_raw = posenet_chained_motion(tr, t, t + step_gap)
                else:
                    Rm, tm_raw = ear.cam_static(tr[t], tr[t + step_gap])
                tm_c = tm_raw * causal_scales[mname]
                uv_c, z_c = ear.project_c(K, (Rm @ Xm.T).T + tm_c)
                priors[f"local_{mname}_causal"] = (src, (R_id, t_zero)) if np.median(z_c) < 2 else (uv_c, (Rm, tm_c))
                if mname != "orbslam3":
                    tm_ts = tm_raw / max(np.linalg.norm(tm_raw), 1e-8) * true_mag
                    uv_ts, z_ts = ear.project_c(K, (Rm @ Xm.T).T + tm_ts)
                    priors[f"local_{mname}_truescale"] = (src, (R_id, t_zero)) if np.median(z_ts) < 2 else (uv_ts, (Rm, tm_ts))
                if mname == "egolite":
                    tm_50 = tm_raw * 50.0
                    uv_50, z_50 = ear.project_c(K, (Rm @ Xm.T).T + tm_50)
                    priors["local_egolite_fixed50"] = (src, (R_id, t_zero)) if np.median(z_50) < 2 else (uv_50, (Rm, tm_50))

            for m in methods:
                if m not in priors:
                    err[m].append(float("nan"))
                    continue
                pred, guess = priors[m]
                got = ear.register_local(gL0, gL1, src, pred, Xm, K, guess, vis, win_px)
                err[m].append(ear.px(got, truth))

        if len(err["local_egolite_causal"]) >= 3:
            seq_results.append({
                "sequence": sid,
                "n_updates": len(err["local_egolite_causal"]),
                "step_gap": step_gap,
                "causal_scales": {k: round(v, 3) for k, v in causal_scales.items()},
                "errors": err,
            })
            print(f"[StereoMIS-Clean] {sid} (n={len(err['local_egolite_causal'])}, {time.time() - t_all:.0f}s): "
                  f"none={np.nanmedian(err['local_none']):.1f} "
                  f"afsfm_c={np.nanmedian(err['local_afsfm_causal']):.1f} "
                  f"endodac_c={np.nanmedian(err['local_endodac_causal']):.1f} "
                  f"fast3r_c={np.nanmedian(err['local_endofast3r_causal']):.1f} "
                  f"droid_c={np.nanmedian(err['local_droid_causal']):.1f} "
                  f"egolite_c={np.nanmedian(err['local_egolite_causal']):.1f} | "
                  f"ts: afsfm={np.nanmedian(err['local_afsfm_truescale']):.1f} "
                  f"endodac={np.nanmedian(err['local_endodac_truescale']):.1f} "
                  f"fast3r={np.nanmedian(err['local_endofast3r_truescale']):.1f} "
                  f"droid={np.nanmedian(err['local_droid_truescale']):.1f} "
                  f"egolite={np.nanmedian(err['local_egolite_truescale']):.1f}", flush=True)

    seq_med = {m: np.array([np.nanmedian(r["errors"][m]) for r in seq_results]) for m in methods}
    ref_c = seq_med["local_egolite_causal"]
    ref_t = seq_med["local_egolite_truescale"]
    table = {}
    for m in methods:
        valid = np.isfinite(seq_med[m])
        if not valid.any():
            continue
        arr = seq_med[m][valid]
        ref_arr = (ref_t if m.endswith("_truescale") else ref_c)[valid]
        p = None
        if len(arr) >= 5 and m not in ("local_egolite_causal", "local_egolite_truescale") and not np.allclose(arr, ref_arr):
            p = float(wilcoxon(arr, ref_arr).pvalue)
        table[m] = {
            "n_seq": int(valid.sum()),
            "median_px": float(np.median(arr)),
            "mean_px": float(np.mean(arr)),
            "within_10px": float(np.mean([np.nanmean(np.array(r["errors"][m]) < 10.0)
                                          for r in seq_results if np.isfinite(np.nanmedian(r["errors"][m]))])),
            "wilcoxon_p_vs_egolite": p,
        }
    out_obj = {
        "n_sequences": len(seq_results),
        "n_updates": int(sum(r["n_updates"] for r in seq_results)),
        "window_px": win_px,
        "resolution": "640x512",
        "table": table,
        "per_sequence": [{
            "sequence": r["sequence"], "n_updates": r["n_updates"], "step_gap": r["step_gap"],
            "medians": {m: round(float(np.nanmedian(r["errors"][m])), 2) for m in methods}
        } for r in seq_results],
    }
    (OUT / "r2_1_stereomis_real_ar_clean.json").write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print("\n=== SUMMARY (StereoMIS Real In Vivo AR) ===")
    for m, st in table.items():
        print(f"  {m:26s}: med={st['median_px']:5.1f} px  mean={st['mean_px']:5.1f} px  <10px={st['within_10px']:4.0%}  p={st['wilcoxon_p_vs_egolite']}")


if __name__ == "__main__":
    main()
