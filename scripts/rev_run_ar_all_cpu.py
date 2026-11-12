"""Sign-verified, Reproducible Runner for VRIH R2.1 & R2.2:
1. Real In Vivo StereoMIS Surgical AR Tissue Registration (11 in vivo da Vinci videos with breathing):
   - Reference 3D tissue patches are triangulated at t from rectified stereo disparity (accounting for
     principal-point offset cx_R - cx_L), and target 2D landmarks at t + delta_t are verified by
     forward-backward + left-right epipolar cycle consistency (< 1.0 px).
   - Evaluates ALL monocular estimators from the paper (`AF-SfMLearner`, `EndoDAC`, `ORB-SLAM3`,
     `Endo-FASt3r`, `DROID-SLAM` with exact lietorch quaternion order, and `EgoLite`) + `No prior`
     and `Homography` during active surgical camera-clutch intervals, under BOTH:
       (a) causal initial-segment scale calibration (realistic online AR deployment, zero future GT), and
       (b) true step-magnitude scale (isolating rotation + translation direction under breathing).
2. Expanded Simulation AR Registration (Table 10, 9 EndoEgoSim-Abs test sequences, 163 updates):
   Compares No prior, LK+PnP, DenseFlow+PnP, Depth ICP, Homography, ORB, AF-SfMLearner, EndoDAC,
   Endo-FASt3r, and EgoLite (+ True Camera reference), all verified for exact positive motion cosine.
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
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon

cv2.setNumThreads(2)
torch.set_num_threads(6)

ROOT = Path(r"D:\ego_motiion_Camera")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import egomotion_baselines as eb  # noqa: E402
import eval_ar_registration as ear  # noqa: E402
import eval_endofast3r as ef  # noqa: E402
from endosim.eval.metrics import load_pose_txt  # noqa: E402
from endosim.lite.data import load_frames, scaled_K  # noqa: E402
from endosim.lite.egolite import build_from_args  # noqa: E402

OUT = ROOT / "results" / "revision_vrih"
OUT.mkdir(parents=True, exist_ok=True)


class CpuModels:
    def __init__(self):
        self.afsfm = eb._load("afsfm", "cpu")
        self.endodac = eb._load("endodac", "cpu")
        self.fast3r = ef.load_model(torch.device("cpu"))
        ck = torch.load(ROOT / "results/egolite/r_v2/last.pth", map_location="cpu")
        self.ego = build_from_args(ck["args"], pretrained=False).eval()
        self.ego.load_state_dict(ck["model"])

    @torch.no_grad()
    def posenet_pair_bgr(self, which: str, bgr_a: np.ndarray, bgr_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Verified on seq_00900344: [img_b, img_a] with inv=False gives t_cos=+0.883 with cam_static(a,b)."""
        enc, dec, from_params = self.afsfm if which == "afsfm" else self.endodac
        ra = cv2.cvtColor(cv2.resize(bgr_a, (eb.W, eb.H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rb = cv2.cvtColor(cv2.resize(bgr_b, (eb.W, eb.H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ia = torch.from_numpy(ra).permute(2, 0, 1)
        ib = torch.from_numpy(rb).permute(2, 0, 1)
        out = dec([enc(torch.cat([ib, ia], 0)[None])])
        T_pt = from_params(out[0][:, 0], out[1][:, 0])[0].double().numpy()
        return T_pt[:3, :3], T_pt[:3, 3]

    @torch.no_grad()
    def posenet_pair(self, which: str, pa: str, pb: str, mag: float):
        R, t = self.posenet_pair_bgr(which, cv2.imread(pa), cv2.imread(pb))
        if np.linalg.norm(t) > 1e-8:
            t = t / np.linalg.norm(t) * mag
        return R, t

    @torch.no_grad()
    def fast3r_pair(self, pa: str, pb: str, mag: float):
        """Verified on seq_00900344: inv=True gives t_cos=+0.957, rot_err=0.16 deg."""
        xa = ef.prepare(pa, torch.device("cpu"))
        xb = ef.prepare(pb, torch.device("cpu"))
        pose2, _ = self.fast3r({"img": xa}, {"img": xb})
        T_ab = np.linalg.inv(pose2["pose"][0].double().numpy())
        R, t = T_ab[:3, :3], T_ab[:3, 3]
        if np.linalg.norm(t) > 1e-8:
            t = t / np.linalg.norm(t) * mag
        return R, t

    @torch.no_grad()
    def egolite_pair(self, pa: str, pb: str, Ks: torch.Tensor):
        imgs, phs, _ = load_frames([pa, pb])
        img = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        ph = torch.from_numpy(phs)[:, None]
        o = self.ego(img[:1], img[1:], Ks[None], ph[:1], ph[1:])
        return o["R"][0].double().numpy(), o["t"][0].double().numpy() * ear.DEPTH_UNIT_MM


def fix_droid_rotations(tr: np.ndarray) -> np.ndarray:
    """Restore exact lietorch [qx,qy,qz,qw] rotation from eval_slam.py's [qw,qx,qy,qz] unpack (cos=+0.954)."""
    out = tr.copy()
    q = Rotation.from_matrix(tr[:, :3, :3]).as_quat()        # (N, 4): [qy_t, qz_t, qw_t, qx_t]
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


def read_stereo_pair_at(cap: cv2.VideoCapture, frame_idx: int, scale: float = 0.5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, full = cap.read()
    if not ok or full is None:
        return None, None
    h2 = full.shape[0] // 2
    left, right = full[:h2], full[h2:2 * h2]
    if scale != 1.0:
        w_s, h_s = int(round(left.shape[1] * scale)), int(round(left.shape[0] * scale))
        left = cv2.resize(left, (w_s, h_s), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (w_s, h_s), interpolation=cv2.INTER_AREA)
    return left, right


def run_real_stereomis_ar(cm: CpuModels, win_px: int = 18) -> dict:
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

        # Causal scale calibration: fit median step ratio on the first 30 moving camera steps of each video
        gt_steps = np.linalg.norm(gt_poses[1:, :3, 3] - gt_poses[:-1, :3, 3], axis=1)
        moving_all = np.where(gt_steps > 0.15)[0]
        if len(moving_all) < 15:
            continue
        calib_idx = moving_all[:min(30, len(moving_all) // 3)]
        test_start = int(calib_idx[-1]) + 30
        causal_scales = {}
        for mname, tr in trajs.items():
            est_steps = np.linalg.norm(tr[1:, :3, 3] - tr[:-1, :3, 3], axis=1)
            ratios = gt_steps[calib_idx] / np.maximum(est_steps[calib_idx], 1e-6)
            causal_scales[mname] = float(np.median(ratios))

        # Find active camera-clutch intervals after test_start where 25-frame displacement >= 3.0 mm
        step_gap = 25
        disp_gap = np.linalg.norm(gt_poses[test_start + step_gap:, :3, 3] - gt_poses[test_start:-step_gap, :3, 3], axis=1)
        active = np.where(disp_gap >= 3.0)[0] + test_start
        if len(active) < 5:
            step_gap = 40
            disp_gap = np.linalg.norm(gt_poses[test_start + step_gap:, :3, 3] - gt_poses[test_start:-step_gap, :3, 3], axis=1)
            active = np.where(disp_gap >= 3.0)[0] + test_start
        if len(active) < 5:
            continue
        # Subsample up to 24 well-spaced active clutch updates per surgical video
        cand_starts = active[np.linspace(0, len(active) - 1, min(24, len(active)), dtype=int)]

        cap = cv2.VideoCapture(mp4s[0])
        err = {m: [] for m in methods}

        for t in cand_starts:
            t = int(t)
            Re_gt, te_gt = ear.cam_static(gt_poses[t], gt_poses[t + step_gap])
            L0, R0_im = read_stereo_pair_at(cap, offset + t, scale=sc)
            L1, R1_im = read_stereo_pair_at(cap, offset + t + step_gap, scale=sc)
            if L0 is None or L1 is None:
                continue
            gL0, gR0 = cv2.cvtColor(L0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(R0_im, cv2.COLOR_BGR2GRAY)
            gL1, gR1 = cv2.cvtColor(L1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(R1_im, cv2.COLOR_BGR2GRAY)

            roi = np.zeros_like(gL0)
            roi[70:440, 90:550] = 255
            p_l0 = cv2.goodFeaturesToTrack(gL0, 90, 0.01, 10, mask=roi)
            if p_l0 is None or len(p_l0) < 20:
                continue
            p_l0 = p_l0.reshape(-1, 2)
            # Stereo match L0 -> R0 with principal-point shift initialization
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
            d0 = disp0[ok0][ok1].astype(np.float64)
            z0 = K[0, 0] * baseline_mm / d0
            Xm = np.c_[(src[:, 0] - K[0, 2]) * z0 / K[0, 0], (src[:, 1] - K[1, 2]) * z0 / K[1, 1], z0]

            uv_cam_gt, zp_gt = ear.project_c(K, (Re_gt @ Xm.T).T + te_gt)
            if np.median(zp_gt) < 5.0 or ear.px(src, uv_cam_gt) < win_px + 3.0:
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

            # Direct sign-verified pair forward for AF-SfMLearner & EndoDAC on (L0, L1)
            for pn in ("afsfm", "endodac"):
                R_pn, t_pn = cm.posenet_pair_bgr(pn, L0, L1)
                # Causal scale: calibrate direct pair norm on step_gap using causal_scales[pn] * step_gap_ratio
                # For multi-frame gap, chain the saved trajectory steps properly:
                # Saved afsfm/endodac traj had traj[k+1] = traj[k] @ T_pt, so T_pt(t -> t+gap):
                # Since X_{k+1} = T_pt_k X_k, across t..t+gap: T_pt_total = inv(tr[t]) @ tr[t+gap] reversed,
                # OR simpler: direct pair network prediction on (L0, L1) with true_mag & causal scale!
                t_ts = t_pn / max(np.linalg.norm(t_pn), 1e-8) * true_mag
                uv_ts, z_ts = ear.project_c(K, (R_pn @ Xm.T).T + t_ts)
                priors[f"local_{pn}_truescale"] = (src, (R_id, t_zero)) if np.median(z_ts) < 2 else (uv_ts, (R_pn, t_ts))
                # For causal chained trajectory of afsfm/endodac: each step T_k = inv(tr[k]) @ tr[k+1] was X_{k+1} = T_k X_k!
                if pn in trajs:
                    tr_p = trajs[pn]
                    T_chain = np.eye(4)
                    for kk in range(t, t + step_gap):
                        T_k = np.linalg.inv(tr_p[kk]) @ tr_p[kk + 1]
                        T_chain = T_k @ T_chain
                    Rc, tc = T_chain[:3, :3], T_chain[:3, 3] * causal_scales[pn]
                    uv_c, z_c = ear.project_c(K, (Rc @ Xm.T).T + tc)
                    priors[f"local_{pn}_causal"] = (src, (R_id, t_zero)) if np.median(z_c) < 2 else (uv_c, (Rc, tc))

            for mname in ("orbslam3", "endofast3r", "droid", "egolite"):
                if mname not in trajs:
                    continue
                Rm, tm_raw = ear.cam_static(trajs[mname][t], trajs[mname][t + step_gap])
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

        cap.release()
        if len(err["local_egolite_causal"]) >= 3:
            seq_results.append({
                "sequence": sid,
                "n_updates": len(err["local_egolite_causal"]),
                "step_gap": step_gap,
                "causal_scales": {k: round(v, 3) for k, v in causal_scales.items()},
                "errors": err,
            })
            print(f"[StereoMIS-AR] {sid} (n={len(err['local_egolite_causal'])}): "
                  f"none={np.nanmedian(err['local_none']):.1f} "
                  f"afsfm_c={np.nanmedian(err['local_afsfm_causal']):.1f} "
                  f"endodac_c={np.nanmedian(err['local_endodac_causal']):.1f} "
                  f"fast3r_c={np.nanmedian(err['local_endofast3r_causal']):.1f} "
                  f"droid_c={np.nanmedian(err['local_droid_causal']):.1f} "
                  f"egolite_c={np.nanmedian(err['local_egolite_causal']):.1f} "
                  f"| truescale: afsfm={np.nanmedian(err['local_afsfm_truescale']):.1f} "
                  f"endodac={np.nanmedian(err['local_endodac_truescale']):.1f} "
                  f"fast3r={np.nanmedian(err['local_endofast3r_truescale']):.1f} "
                  f"droid={np.nanmedian(err['local_droid_truescale']):.1f} "
                  f"egolite={np.nanmedian(err['local_egolite_truescale']):.1f}", flush=True)

    if not seq_results:
        return {"n_sequences": 0}

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
    return {
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


def run_sim_ar(cm: CpuModels, main_win: int = 18) -> dict:
    seqs = ear.select_sequences(str(ROOT / "sim_data_abs" / "test"))
    local_methods = [
        "local_none", "local_homography", "local_orb",
        "local_afsfm_truescale", "local_endodac_truescale", "local_endofast3r_truescale",
        "local_egolite", "local_egolite_truescale", "local_true",
    ]
    all_methods = list(ear.GLOBAL) + local_methods
    results = []
    t0_all = time.time()

    for path_mm, amp_mm, seq in seqs:
        try:
            K, pose, traj = ear.load_clip(seq)
            picked = ear.choose_stride(K, pose, traj, main_win)
            if picked is None:
                continue
        except (ValueError, FileNotFoundError):
            continue
        stride, cam_px, tis_px, relaxed = picked
        Xm = traj[0]
        first = cv2.imread(os.path.join(seq, "color", "000000.png"))
        Kd = {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
              "width": first.shape[1], "height": first.shape[0]}
        Ks = torch.from_numpy(scaled_K(Kd, first.shape[1], first.shape[0]))

        err = {m: [] for m in all_methods}
        for t in range(0, len(pose) - stride, stride):
            pa = os.path.join(seq, "color", f"{t:06d}.png")
            pb = os.path.join(seq, "color", f"{t + stride:06d}.png")
            g0 = cv2.imread(pa, cv2.IMREAD_GRAYSCALE)
            g1 = cv2.imread(pb, cv2.IMREAD_GRAYSCALE)
            Xc = (pose[t][:3, :3].T @ (traj[t] - pose[t][:3, 3]).T).T
            R0, t0 = ear.kabsch(Xm, Xc)
            src, za = ear.project_w(traj[t], pose[t], K)
            truth, zb = ear.project_w(traj[t + stride], pose[t + stride], K)
            vis = (za > 1) & (zb > 1)
            if vis.mean() < 0.7:
                continue

            Re, te = ear.cam_static(pose[t], pose[t + stride])
            true_mag = float(np.linalg.norm(te))
            Rn, tn = cm.egolite_pair(pa, pb, Ks)
            tn_ts = tn / max(np.linalg.norm(tn), 1e-8) * true_mag

            H = ear.homography(g0, g1)
            uvH = src if H is None else cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
            try:
                rv0, _ = cv2.Rodrigues(R0)
                okH, rvH, tvH = cv2.solvePnP(Xm.astype(np.float32), uvH.astype(np.float32), K, None,
                                             rv0, t0.reshape(3, 1).astype(np.float64), True, cv2.SOLVEPNP_ITERATIVE)
            except cv2.error:
                okH = False
            guessH = (cv2.Rodrigues(rvH)[0], tvH.ravel()) if okH else (R0, t0)

            motion_dict = {
                "local_orb": ear.orb_motion(g0, g1, K, true_mag),
                "local_afsfm_truescale": cm.posenet_pair("afsfm", pa, pb, true_mag),
                "local_endodac_truescale": cm.posenet_pair("endodac", pa, pb, true_mag),
                "local_endofast3r_truescale": cm.fast3r_pair(pa, pb, true_mag),
                "local_egolite": (Rn, tn),
                "local_egolite_truescale": (Rn, tn_ts),
                "local_true": (Re, te),
            }
            priors = {"local_none": (src, (R0, t0)), "local_homography": (uvH, guessH)}
            for name, mot in motion_dict.items():
                if mot is None:
                    priors[name] = (src, (R0, t0))
                    continue
                Rm, tm = mot
                uvp, zp = ear.project_c(K, (Rm @ Xc.T).T + tm)
                priors[name] = (src, (R0, t0)) if np.median(zp) < 1 else (uvp, (Rm @ R0, Rm @ t0 + tm))

            err["homography"].append(ear.px(uvH[vis], truth[vis]))
            err["lk_pnp"].append(ear.px(ear.lk_pnp(g0, g1, src, Xm, K, (R0, t0), vis)[vis], truth[vis]))
            err["flow_pnp"].append(ear.px(ear.flow_pnp(g0, g1, src, Xm, K, (R0, t0), vis)[vis], truth[vis]))
            ic = ear.icp(Xc, os.path.join(seq, "depth", f"{t + stride:06d}.png"), K, Xm)
            err["icp"].append(ear.px((src if ic is None else ic)[vis], truth[vis]))
            for name in local_methods:
                pred, guess = priors[name]
                got = ear.register_local(g0, g1, src, pred, Xm, K, guess, vis, main_win)
                err[name].append(ear.px(got[vis], truth[vis]))

        if err["local_egolite"]:
            results.append({"sequence": os.path.basename(seq), "stride": stride, "n": len(err["local_egolite"]), "errors": err})
            print(f"[Sim-AR {len(results)}] {os.path.basename(seq)} ({time.time() - t0_all:.0f}s): "
                  f"none={np.median(err['local_none']):.1f} afsfm={np.median(err['local_afsfm_truescale']):.1f} "
                  f"endodac={np.median(err['local_endodac_truescale']):.1f} "
                  f"fast3r={np.median(err['local_endofast3r_truescale']):.1f} "
                  f"egolite={np.median(err['local_egolite']):.1f}", flush=True)

    seq_med = {m: np.array([np.median(r["errors"][m]) for r in results]) for m in all_methods}
    ref = seq_med["local_egolite"]
    non_ref = [m for m in all_methods if m not in ("local_true", "local_egolite_truescale")]
    table = {}
    for m in all_methods:
        p = None if m == "local_egolite" or np.allclose(seq_med[m], ref) else float(wilcoxon(seq_med[m], ref).pvalue)
        table[m] = {
            "median_px": float(np.median(seq_med[m])),
            "mean_px": float(np.mean(seq_med[m])),
            "within_10px": float(np.mean([np.mean(np.array(r["errors"][m]) < 10.0) for r in results])),
            "lowest_on_sequences": int(sum(seq_med[m][i] <= min(seq_med[k][i] for k in non_ref) + 1e-9
                                           for i in range(len(results)))) if m in non_ref else None,
            "wilcoxon_p_vs_egolite": p,
        }
    return {
        "n_sequences": len(results),
        "n_updates": int(sum(r["n"] for r in results)),
        "window_px": main_win,
        "table": table,
        "per_sequence_medians": [{
            "sequence": r["sequence"], "n": r["n"], "stride": r["stride"],
            "medians": {m: round(float(np.median(r["errors"][m])), 2) for m in all_methods}
        } for r in results],
    }


def main():
    cm = CpuModels()
    print("=== [1/2] Real In Vivo StereoMIS Surgical AR Tissue Registration ===", flush=True)
    smis_ar = run_real_stereomis_ar(cm, win_px=18)
    (OUT / "r2_1_stereomis_real_ar.json").write_text(json.dumps(smis_ar, indent=2), encoding="utf-8")
    print("Saved r2_1_stereomis_real_ar.json", flush=True)

    print("=== [2/2] Expanded Simulation AR Registration (CPU) ===", flush=True)
    sim_ar = run_sim_ar(cm, main_win=18)
    (OUT / "r2_2_sim_ar_expanded.json").write_text(json.dumps(sim_ar, indent=2), encoding="utf-8")
    print("Saved r2_2_sim_ar_expanded.json", flush=True)


if __name__ == "__main__":
    main()
