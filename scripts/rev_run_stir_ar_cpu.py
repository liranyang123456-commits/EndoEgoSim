"""Real In Vivo Porcine Surgery AR Landmark Registration on STIR Challenge 2024 (R2.1 & R2.2).

Ground truth: physical infrared-fluorescent (ICG) tissue tattoos segmented in start & end IR frames
(invisible in the visible-light RGB video fed to the estimators).
Compares monocular ego-motion priors + bounded-window (32 px at 1280x1024) local template registration:
  - No prior (previous position)
  - Whole-frame Homography
  - Classical ORB + Essential matrix
  - AF-SfMLearner (13M)
  - EndoDAC (13M)
  - Endo-FASt3r (428M)
  - EgoLite (20.9M)
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
from PIL import Image
from scipy.stats import wilcoxon

cv2.setNumThreads(2)
torch.set_num_threads(6)

ROOT = Path(r"D:\ego_motiion_Camera")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import egomotion_baselines as eb  # noqa: E402
import eval_ar_registration as ear  # noqa: E402
import eval_endofast3r as ef  # noqa: E402
import eval_stir as es  # noqa: E402
from endosim.lite.data import H_IN, W_IN  # noqa: E402
from endosim.lite.egolite import build_from_args, chain  # noqa: E402
from endosim.pseudoheight import pseudo_height_label  # noqa: E402

OUT = ROOT / "results" / "revision_vrih"
STIR_ROOT = Path(r"E:\World_Agent_Enoscopy\datasets\STIR\STIRChallenge_2024")


@torch.no_grad()
def main():
    afsfm = eb._load("afsfm", "cpu")
    endodac = eb._load("endodac", "cpu")
    fast3r = ef.load_model(torch.device("cpu"))
    ck = torch.load(ROOT / "results/egolite/r_v2/last.pth", map_location="cpu")
    ego_model = build_from_args(ck["args"], pretrained=False).eval()
    ego_model.load_state_dict(ck["model"])

    methods = ["no_prior", "homography", "orb", "afsfm", "endodac", "endofast3r", "egolite"]
    clip_rows = []
    t0 = time.time()

    all_clips = sorted(glob.glob(str(STIR_ROOT / "*" / "left" / "seq*")))
    for clip in all_clips:
        coll = clip.split(os.sep)[-3]
        calib = json.loads(Path(STIR_ROOT, coll, "calib.json").read_text(encoding="utf-8"))
        K_full = np.asarray(calib["leftcameramat"], np.float64)
        start_pts = es.centroids(os.path.join(clip, "segmentation", "icgstartseg.png"))
        end_pts = es.centroids(os.path.join(clip, "segmentation", "icgendseg.png"))
        mp4 = glob.glob(os.path.join(clip, "frames", "*.mp4"))
        if len(start_pts) < 3 or len(end_pts) < 3 or not mp4:
            continue
        raw_shift = es.nearest_err(start_pts, end_pts)
        med_shift = float(np.median(raw_shift))
        # Select clips with substantial camera+tissue motion (>= 32 px, exceeding search window)
        if med_shift < 32.0 or med_shift > 220.0:
            continue

        frames = es.decode(mp4[0])
        if len(frames) < 4:
            continue
        # Subsample up to 16 frames across the clip so multi-step chaining captures smooth trajectory
        sel = np.unique(np.linspace(0, len(frames) - 1, min(len(frames), 16)).round().astype(int))
        f_sub = [frames[i] for i in sel]
        H0, W0 = f_sub[0].shape[:2]
        sx, sy = W_IN / W0, H_IN / H0

        rgb = [cv2.cvtColor(cv2.resize(f, (W_IN, H_IN), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
               .astype(np.float32) / 255.0 for f in f_sub]
        img_t = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2)
        ph_t = torch.from_numpy(np.stack([pseudo_height_label(r) for r in rgb]))[:, None]
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

        def chain_posenet(bundle):
            enc, dec, from_params = bundle
            T_cur = np.eye(4)
            for k in range(len(f_sub) - 1):
                x = torch.cat([img_t[k + 1:k + 2], img_t[k:k + 1]], 1)
                out = dec([enc(x)])
                T_ba = from_params(out[0][:, 0], out[1][:, 0])[0].double().numpy()
                T_cur = T_ba @ T_cur
            if np.linalg.norm(T_cur[:3, 3]) > 1e-8:
                T_cur[:3, 3] = T_cur[:3, 3] / np.linalg.norm(T_cur[:3, 3]) * ego_norm
            return es.reproject(p_in, z_pts, T_cur, K_in) / np.array([sx, sy])

        def chain_fast3r():
            def prep_bgr(bgr):
                im = ef.resize_pil_image(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), 512)
                W, H = im.size
                cx, cy = W // 2, H // 2
                hw, hh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
                return ef.ImgNorm(im.crop((cx - hw, cy - hh, cx + hw, cy + hh)))[None]
            prepped = [prep_bgr(f) for f in f_sub]
            T_cur = np.eye(4)
            for k in range(len(f_sub) - 1):
                pose2, _ = fast3r({"img": prepped[k]}, {"img": prepped[k + 1]})
                T_cur = np.linalg.inv(pose2["pose"][0].double().numpy()) @ T_cur
            if np.linalg.norm(T_cur[:3, 3]) > 1e-8:
                T_cur[:3, 3] = T_cur[:3, 3] / np.linalg.norm(T_cur[:3, 3]) * ego_norm
            return es.reproject(p_in, z_pts, T_cur, K_in) / np.array([sx, sy])

        g_first = cv2.cvtColor(f_sub[0], cv2.COLOR_BGR2GRAY)
        g_last = cv2.cvtColor(f_sub[-1], cv2.COLOR_BGR2GRAY)
        H_mat = ear.homography(g_first, g_last)
        pred_hom = start_pts if H_mat is None else cv2.perspectiveTransform(
            start_pts.reshape(-1, 1, 2).astype(np.float32), H_mat).reshape(-1, 2)

        orb_mot = ear.orb_motion(cv2.resize(g_first, (W_IN, H_IN)), cv2.resize(g_last, (W_IN, H_IN)), K_in, ego_norm)
        if orb_mot is not None:
            T_orb = np.eye(4); T_orb[:3, :3], T_orb[:3, 3] = orb_mot
            pred_orb = es.reproject(p_in, z_pts, T_orb, K_in) / np.array([sx, sy])
        else:
            pred_orb = start_pts.copy()

        WIN_FULL, PATCH_FULL = 32, 15

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

        priors_map = {
            "no_prior": start_pts,
            "homography": pred_hom,
            "orb": pred_orb,
            "afsfm": chain_posenet(afsfm),
            "endodac": chain_posenet(endodac),
            "endofast3r": chain_fast3r(),
            "egolite": pred_ego,
        }
        rec = {"clip": f"{coll}/{os.path.basename(clip)}", "n_frames": len(frames),
               "n_tattoos": len(start_pts), "raw_shift_px": med_shift}
        for mn, pp in priors_map.items():
            rp = refine_local(pp)
            er = es.nearest_err(rp, end_pts)
            rec[f"{mn}_prior_px"] = float(np.median(es.nearest_err(pp, end_pts)))
            rec[f"{mn}_reg_px"] = float(np.median(er))
            rec[f"{mn}_within25px"] = float(np.mean(er < 25.0))
        clip_rows.append(rec)
        print(f"[STIR {len(clip_rows)}] {rec['clip']} (shift {med_shift:.1f}): "
              f"none={rec['no_prior_reg_px']:.1f} afsfm={rec['afsfm_reg_px']:.1f} "
              f"endodac={rec['endodac_reg_px']:.1f} fast3r={rec['endofast3r_reg_px']:.1f} "
              f"egolite={rec['egolite_reg_px']:.1f} ({time.time() - t0:.0f}s)", flush=True)
        if len(clip_rows) >= 20:
            break

    ref = np.array([r["egolite_reg_px"] for r in clip_rows])
    summ = {}
    for mn in methods:
        arr = np.array([r[f"{mn}_reg_px"] for r in clip_rows])
        apri = np.array([r[f"{mn}_prior_px"] for r in clip_rows])
        aw25 = np.array([r[f"{mn}_within25px"] for r in clip_rows])
        p = None if mn == "egolite" or np.allclose(arr, ref) else float(wilcoxon(arr, ref).pvalue)
        summ[mn] = {
            "prior_median_px": float(np.median(apri)),
            "reg_median_px": float(np.median(arr)),
            "reg_mean_px": float(np.mean(arr)),
            "within_25px": float(np.mean(aw25)),
            "wilcoxon_p_vs_egolite": p,
        }
    (OUT / "r2_1_stir_real_ar.json").write_text(
        json.dumps({"n_clips": len(clip_rows), "window_px": 32, "summary": summ, "clips": clip_rows}, indent=2),
        encoding="utf-8",
    )
    print("Saved r2_1_stir_real_ar.json!", flush=True)


if __name__ == "__main__":
    main()
