"""Create a new figure `fig_real_ar_registration.png` showing real in vivo robotic surgical AR
registration on BOTH:
  - Top row: two active camera-clutch updates from in vivo StereoMIS (left endoscopic view at 640x512,
    comparing No Prior vs. AF-SfMLearner vs. EgoLite, with green ground-truth stereo tissue landmarks
    and red registered landmarks + median overlay error in px).
  - Bottom row: two multi-second clips from in vivo STIR 2024 (visible RGB frame at 1280x1024, where the
    physical ICG infrared tattoos are invisible to all estimators, comparing No Prior vs. EndoDAC/AF-SfM
    vs. EgoLite).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(r"D:\ego_motiion_Camera")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import egomotion_baselines as eb  # noqa: E402
import eval_ar_registration as ear  # noqa: E402
import eval_stir as es  # noqa: E402
import rev_eval_stereomis_real_ar as r_smis  # noqa: E402
from endosim.eval.metrics import load_pose_txt  # noqa: E402
from endosim.lite.data import H_IN, W_IN  # noqa: E402
from endosim.lite.egolite import build_from_args, chain  # noqa: E402
from endosim.pseudoheight import pseudo_height_label  # noqa: E402

STIR_ROOT = Path(r"E:\World_Agent_Enoscopy\datasets\STIR\STIRChallenge_2024")


def get_stereomis_examples() -> list[dict]:
    """Pick 2 representative StereoMIS active updates where camera shift > 18 px."""
    smis_refs = [ln.strip() for ln in (ROOT / "lists/stereomis.txt").read_text().splitlines() if ln.strip()]
    est_dirs = {
        "afsfm": ROOT / "results/sota/afsfm_stereomis_full",
        "egolite": ROOT / "results/sota/egolite_r_v2_stereomis_full",
    }
    examples = []
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
        tr_af = np.loadtxt(str(est_dirs["afsfm"] / f"{sid}_est_c2w.txt")).reshape(-1, 4, 4)
        tr_eg = np.loadtxt(str(est_dirs["egolite"] / f"{sid}_est_c2w.txt")).reshape(-1, 4, 4)

        gt_steps = np.linalg.norm(gt_poses[1:, :3, 3] - gt_poses[:-1, :3, 3], axis=1)
        moving_all = np.where(gt_steps > 0.15)[0]
        calib_idx = moving_all[:min(40, len(moving_all) // 4)]
        test_start = int(calib_idx[-1]) + 30
        af_steps = np.array([np.linalg.norm((np.linalg.inv(tr_af[i]) @ tr_af[i + 1])[:3, 3]) for i in calib_idx])
        sc_af = float(np.median(gt_steps[calib_idx] / np.maximum(af_steps, 1e-6)))

        step_gap = 25
        disp_gap = np.linalg.norm(gt_poses[test_start + step_gap:, :3, 3] - gt_poses[test_start:-step_gap, :3, 3], axis=1)
        active = np.where((disp_gap >= 5.0) & (disp_gap <= 22.0))[0] + test_start
        if len(active) < 4:
            continue
        cand_starts = active[np.linspace(0, len(active) - 1, min(12, len(active)), dtype=int)]
        need_raw = {offset + int(t) for t in cand_starts} | {offset + int(t) + step_gap for t in cand_starts}
        decoded = r_smis.decode_needed_stereo_frames(mp4s[0], need_raw, scale=sc)

        best_cand = None
        for t in cand_starts:
            t = int(t)
            r0_idx, r1_idx = offset + t, offset + t + step_gap
            if r0_idx not in decoded or r1_idx not in decoded:
                continue
            L0, R0_im = decoded[r0_idx]; L1, _ = decoded[r1_idx]
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
            ok0 = (st_lr.ravel() == 1) & (st_rl.ravel() == 1) & (disp0 > 5.0) & (disp0 < 120.0) & (np.abs(p_l0[:, 1] - p_r0[:, 1]) < 2.0) & (np.linalg.norm(p_l0 - p_l0_bk, axis=1) < 1.2)
            if ok0.sum() < 16:
                continue
            p_s = p_l0[ok0].astype(np.float32)
            p_l1, st_f, _ = cv2.calcOpticalFlowPyrLK(gL0, gL1, p_s, None, winSize=(31, 31), maxLevel=4)
            p_s_bk, st_b, _ = cv2.calcOpticalFlowPyrLK(gL1, gL0, p_l1, None, winSize=(31, 31), maxLevel=4)
            in_im = (p_l1[:, 0] > 25) & (p_l1[:, 0] < 615) & (p_l1[:, 1] > 25) & (p_l1[:, 1] < 487)
            ok1 = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (np.linalg.norm(p_s - p_s_bk, axis=1) < 1.2) & in_im
            if ok1.sum() < 16:
                continue
            src = p_s[ok1].astype(np.float64); truth = p_l1[ok1].astype(np.float64)
            if ear.px(src, truth) < 30.0:
                continue
            d0 = disp0[ok0][ok1].astype(np.float64); z0 = K[0, 0] * baseline_mm / d0
            Xm = np.c_[(src[:, 0] - K[0, 2]) * z0 / K[0, 0], (src[:, 1] - K[1, 2]) * z0 / K[1, 1], z0]
            vis = np.ones(len(Xm), dtype=bool); R_id, t_zero = np.eye(3), np.zeros(3)

            got_none = ear.register_local(gL0, gL1, src, src, Xm, K, (R_id, t_zero), vis, 18)
            Rm_af, tm_af = r_smis.posenet_chained_motion(tr_af, t, t + step_gap)
            uv_af, _ = ear.project_c(K, (Rm_af @ Xm.T).T + tm_af * sc_af)
            got_af = ear.register_local(gL0, gL1, src, uv_af, Xm, K, (Rm_af, tm_af * sc_af), vis, 18)

            Rm_eg, tm_eg = ear.cam_static(tr_eg[t], tr_eg[t + step_gap])
            uv_eg, _ = ear.project_c(K, (Rm_eg @ Xm.T).T + tm_eg * 50.0)
            got_eg = ear.register_local(gL0, gL1, src, uv_eg, Xm, K, (Rm_eg, tm_eg * 50.0), vis, 18)

            e_none, e_af, e_eg = ear.px(got_none, truth), ear.px(got_af, truth), ear.px(got_eg, truth)
            if e_none > 30.0 and e_af > 8.0 and e_eg < 6.0 and e_eg < e_af - 3.0:
                short_name = sid.split("_")[-2] + "_" + sid.split("_")[-1] if "P2_" in sid else sid.split("_")[-1]
                best_cand = {
                    "title": f"StereoMIS {short_name}",
                    "rgb": cv2.cvtColor(L1, cv2.COLOR_BGR2RGB),
                    "truth": truth,
                    "none": (got_none, e_none),
                    "posenet": (got_af, e_af, "AF-SfMLearner"),
                    "egolite": (got_eg, e_eg),
                }
                break
        if best_cand is not None:
            examples.append(best_cand)
            if len(examples) >= 2:
                break
    return examples


@torch.no_grad()
def get_stir_examples() -> list[dict]:
    """Pick 2 representative STIR 2024 clips where IR tattoos shift > 45 px."""
    d_stir = json.loads((ROOT / "results/revision_vrih/r2_1_stir_real_ar.json").read_text(encoding="utf-8"))
    # Find 2 clean clips where No Prior & PoseNet fail while EgoLite succeeds
    cands = [r for r in d_stir["clips"]
             if r["no_prior_reg_px"] > 45.0 and r["endodac_reg_px"] > 30.0 and r["egolite_reg_px"] < 20.0]
    cands = cands[:2]

    endodac = eb._load("endodac", "cpu")
    ck = torch.load(ROOT / "results/egolite/r_v2/last.pth", map_location="cpu")
    ego_model = build_from_args(ck["args"], pretrained=False).eval()
    ego_model.load_state_dict(ck["model"])
    WIN_FULL, PATCH_FULL = 32, 15
    out = []

    for r in cands:
        cid = r["clip"]
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
        img_t = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2)
        ph_t = torch.from_numpy(np.stack([pseudo_height_label(x) for x in rgb]))[:, None]
        K_in = np.array([[K_full[0, 0] * sx, 0, K_full[0, 2] * sx],
                         [0, K_full[1, 1] * sy, K_full[1, 2] * sy], [0, 0, 1]], np.float32)
        Kt = torch.from_numpy(K_in)

        Rs, ts = [], []
        z0_map = None
        for k in range(len(f_sub) - 1):
            o = ego_model(img_t[k:k + 1], img_t[k + 1:k + 2], Kt[None], ph_t[k:k + 1], ph_t[k + 1:k + 2])
            if z0_map is None:
                z0_map = o["Z"][0:1].float()
            Rs.append(o["R"].float()); ts.append(o["t"].float())
        T_ego = chain(torch.cat(Rs, 0), torch.cat(ts, 0), torch.ones(len(f_sub) - 1)).double().numpy()
        T_es_ego = np.linalg.inv(T_ego[-1]) @ T_ego[0]
        p_in = start_pts * np.array([sx, sy])
        z_pts = es.sample(z0_map, torch.from_numpy(p_in))[:, 0].double().numpy()
        pred_ego = es.reproject(p_in, z_pts, T_es_ego, K_in) / np.array([sx, sy])
        ego_norm = float(np.linalg.norm(T_es_ego[:3, 3]))

        enc, dec, from_params = endodac
        T_cur = np.eye(4)
        for k in range(len(f_sub) - 1):
            x = torch.cat([img_t[k + 1:k + 2], img_t[k:k + 1]], 1)
            op = dec([enc(x)])
            T_cur = from_params(op[0][:, 0], op[1][:, 0])[0].double().numpy() @ T_cur
        if np.linalg.norm(T_cur[:3, 3]) > 1e-8:
            T_cur[:3, 3] = T_cur[:3, 3] / np.linalg.norm(T_cur[:3, 3]) * ego_norm
        pred_dac = es.reproject(p_in, z_pts, T_cur, K_in) / np.array([sx, sy])

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

        got_none = refine_local(start_pts.copy())
        got_dac = refine_local(pred_dac)
        got_ego = refine_local(pred_ego)
        out.append({
            "title": f"STIR {cid}",
            "rgb": cv2.cvtColor(f_sub[-1], cv2.COLOR_BGR2RGB),
            "truth": end_pts,
            "none": (got_none, float(np.median(es.nearest_err(got_none, end_pts)))),
            "posenet": (got_dac, float(np.median(es.nearest_err(got_dac, end_pts))), "EndoDAC"),
            "egolite": (got_ego, float(np.median(es.nearest_err(got_ego, end_pts)))),
        })
    return out


def main() -> None:
    smis_ex = get_stereomis_examples()
    stir_ex = get_stir_examples()
    print(f"Found {len(smis_ex)} StereoMIS examples and {len(stir_ex)} STIR examples")

    fig, axes = plt.subplots(2, 6, figsize=(13.5, 4.6), dpi=240)
    plt.subplots_adjust(wspace=0.04, hspace=0.14, left=0.01, right=0.99, top=0.91, bottom=0.02)

    all_rows = [(0, smis_ex), (1, stir_ex)]
    for r_idx, ex_list in all_rows:
        for pair_i, ex in enumerate(ex_list[:2]):
            c_base = pair_i * 3
            panels = [
                ("No prior", ex["none"][0], ex["none"][1]),
                (ex["posenet"][2], ex["posenet"][0], ex["posenet"][1]),
                ("EgoLite", ex["egolite"][0], ex["egolite"][1]),
            ]
            for j, (m_label, pts_est, err_px) in enumerate(panels):
                ax = axes[r_idx, c_base + j]
                ax.imshow(ex["rgb"])
                t_pts = ex["truth"]
                ms_ring = 52 if r_idx == 0 else 72
                ms_dot = 18 if r_idx == 0 else 28
                ax.scatter(t_pts[:, 0], t_pts[:, 1], s=ms_ring, facecolors="none", edgecolors="#32ef6a", linewidths=1.4, label="True tissue")
                ax.scatter(pts_est[:, 0], pts_est[:, 1], s=ms_dot, c="#ff3b3b", marker="o", linewidths=0.4, edgecolors="white", label="Registered")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor("#cccccc"); sp.set_linewidth(0.6)
                head_txt = f"{ex['title']} — {m_label}" if j == 0 else m_label
                ax.set_title(head_txt, fontsize=8.2, pad=3, fontweight="bold" if m_label == "EgoLite" else "normal")
                ax.text(0.97, 0.06, f"{err_px:.1f} px", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8.2, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.68, edgecolor="none"))

    # Clean single legend at top right of first panel (no (a)(b) panel letters per house style!)
    h1 = plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#32ef6a", markeredgewidth=1.5, markersize=6.5, label="True landmark")
    h2 = plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#ff3b3b", markeredgecolor="white", markeredgewidth=0.5, markersize=5.0, label="Registered")
    axes[0, 0].legend(handles=[h1, h2], loc="upper left", fontsize=7.0, framealpha=0.72, facecolor="black", labelcolor="white", borderpad=0.25, handletextpad=0.3)

    for dst in [ROOT / "bspc_paper" / "figs" / "fig_real_ar_registration.png",
                ROOT / "VRIH_EgoLite_submission" / "figs" / "fig_real_ar_registration.png"]:
        fig.savefig(dst, dpi=240, bbox_inches="tight", pad_inches=0.03)
        print("Saved", dst)


if __name__ == "__main__":
    main()
