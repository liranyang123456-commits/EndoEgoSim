"""Extended StereoMIS diagnostics (pure CPU on saved full-video trajectories):

1. Decision 1 (R2.1): Automatically detect ALL non-overlapping near-static 20-s camera windows
   across all 11 full-length StereoMIS sequences (GT spread < 1.0 mm, plus < 0.5 mm sensitivity
   and sequence-level clustered summary), evaluating EgoLite, EndoDAC, AF-SfMLearner,
   Same-corpus 12.5-M PoseNet, Endo-FASt3r, DROID-SLAM and ORB-SLAM3.
2. Decision 4 (R1.2): Windowed Sim(3) alignment (60 s, 30 s, 15 s, 10 s) versus whole-sequence
   Sim(3) and unaligned fixed-50-mm SE(3), quantifying intra-sequence scale drift directly.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from endosim.eval.align import align_trajectories  # noqa: E402
from endosim.eval.metrics import load_pose_txt  # noqa: E402

METHODS = {
    "EgoLite": ("results/sota/egolite_r_v2_stereomis_full", "{sid}_est_c2w.txt"),
    "Same-corpus PoseNet": ("results/revision_vrih/posenet_stereomis_full_fixed", "{sid}_c2w.txt"),
    "EndoDAC": ("results/sota/endodac_stereomis_full", "{sid}_est_c2w.txt"),
    "AF-SfMLearner": ("results/sota/afsfm_stereomis_full", "{sid}_est_c2w.txt"),
    "Endo-FASt3r": ("results/sota/endofast3r_stereomis_full", "{sid}_est_c2w.txt"),
    "DROID-SLAM": ("results/sota/droid_stereomis_full", "{sid}_est_c2w.txt"),
    "ORB-SLAM3": ("results/sota/orb_e_c2w_stereomis_full", "{sid}_est_c2w.txt"),
}


KNOWN_STEREOMIS_FPS = {
    "stereomis_extracted_StereoMIS_0_0_1_P1": 60.003128,
    "stereomis_extracted_StereoMIS_0_0_1_P2_0": 52.699617,
    "stereomis_extracted_StereoMIS_0_0_1_P2_1": 54.317561,
    "stereomis_extracted_StereoMIS_0_0_1_P2_2": 54.353815,
    "stereomis_extracted_StereoMIS_0_0_1_P2_3": 54.323916,
    "stereomis_extracted_StereoMIS_0_0_1_P2_4": 54.426660,
    "stereomis_extracted_StereoMIS_0_0_1_P2_5": 54.204199,
    "stereomis_extracted_StereoMIS_0_0_1_P2_6": 26.090665,
    "stereomis_extracted_StereoMIS_0_0_1_P2_7": 37.033519,
    "stereomis_extracted_StereoMIS_0_0_1_P2_8": 38.959833,
    "stereomis_extracted_StereoMIS_0_0_1_P3": 60.003128,
}


def video_fps(seq_dir: str) -> float:
    sid = os.path.basename(str(seq_dir).rstrip("/\\"))
    idx_p = Path(seq_dir, "color_index.json")
    if idx_p.is_file():
        idx = json.loads(idx_p.read_text(encoding="utf-8"))
        cap = cv2.VideoCapture(idx["path"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 1.0:
            return float(fps)
    return float(KNOWN_STEREOMIS_FPS.get(sid, 30.0))


def resp_peak(sig_1d: np.ndarray, fs: float, lo: float = 0.05, hi: float = 3.0) -> tuple[float, float]:
    x = sig_1d - sig_1d.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freq = np.fft.rfftfreq(len(x), 1.0 / fs)
    band = (freq >= lo) & (freq <= hi)
    k = int(np.argmax(spec * band))
    resp = (freq >= 0.1) & (freq <= 0.6)
    return float(freq[k]), float(spec[resp].sum() / max(float(spec[band].sum()), 1e-12))


def run_all() -> dict:
    seqs = [
        str(ROOT / ln.strip().replace("\\", "/").split("ego_motiion_Camera/")[-1].lstrip("/"))
        if not os.path.isdir(ln.strip())
        else ln.strip()
        for ln in (ROOT / "lists/stereomis.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]

    # Cache GT and aligned trajectories once per sequence
    seq_data = []
    for d in seqs:
        sid = os.path.basename(d.rstrip("/\\"))
        gt = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
        fps = video_fps(d)
        aligned_c = {}
        raw_est = {}
        global_scales = {}
        for m, (rel_dir, pat) in METHODS.items():
            p = ROOT / rel_dir / pat.format(sid=sid)
            if not p.is_file():
                continue
            est = np.loadtxt(p).reshape(-1, 4, 4)
            n = min(len(est), len(gt))
            if abs(len(est) - len(gt)) > 5:
                continue
            al, (sc, _, _) = align_trajectories(est[:n], gt[:n], True)
            aligned_c[m] = al[:, :3, 3]
            raw_est[m] = est[:n]
            global_scales[m] = float(sc)
        seq_data.append({
            "dir": d, "sid": sid, "gt": gt, "fps": fps,
            "aligned_c": aligned_c, "raw_est": raw_est, "global_scales": global_scales,
        })

    # --- 1. Decision 1: automatic all non-overlapping 20-s near-static windows ---
    def collect_windows(thr_mm: float, win_s: float = 20.0) -> dict:
        windows = []
        per_seq_counts = {}
        for sd in seq_data:
            sid, gt, fps = sd["sid"], sd["gt"], sd["fps"]
            W = int(win_s * fps)
            c = gt[:, :3, 3]
            cands = []
            for a in range(0, len(gt) - W + 1, max(1, W // 20)):
                q = c[a:a + W]
                sp = float(np.linalg.norm(q - q.mean(0), axis=1).max())
                if sp < thr_mm:
                    cands.append((sp, a))
            cands.sort()
            chosen = []
            for sp, a in cands:
                if any(not (a + W <= ca or a >= ca + W) for _, ca in chosen):
                    continue
                chosen.append((sp, a))
            chosen.sort(key=lambda x: x[1])
            per_seq_counts[sid] = len(chosen)
            for sp, a in chosen:
                rec = {"seq": sid, "start": int(a), "W": int(W), "fps": float(fps), "gt_spread_mm": float(sp)}
                for m, ac in sd["aligned_c"].items():
                    seg = ac[a:a + W]
                    if len(seg) < W // 2:
                        continue
                    e0 = seg - seg.mean(0)
                    axis = np.linalg.svd(e0, full_matrices=False)[2][0]
                    pk_hz, r_pow = resp_peak(e0 @ axis, fps)
                    rec[m] = {
                        "spread_mm": float(np.linalg.norm(e0, axis=1).max()),
                        "rms_mm": float(np.sqrt(np.mean(np.sum(e0 ** 2, axis=1)))),
                        "peak_hz": pk_hz,
                        "resp_power": r_pow,
                    }
                windows.append(rec)

        summary = {
            "threshold_mm": thr_mm,
            "n_windows": len(windows),
            "n_sequences": sum(v > 0 for v in per_seq_counts.values()),
            "per_sequence_window_counts": per_seq_counts,
            "gt_median_spread_mm": float(np.median([w["gt_spread_mm"] for w in windows])) if windows else None,
            "methods_window_level": {},
            "methods_sequence_clustered": {},
        }
        seq_ids_with = [s for s, k in per_seq_counts.items() if k > 0]
        for m in METHODS:
            w_vals = [w[m]["spread_mm"] for w in windows if m in w]
            ego_w = [w["EgoLite"]["spread_mm"] for w in windows if m in w]
            r_pows = [w[m]["resp_power"] for w in windows if m in w]
            p_w = float(wilcoxon(ego_w, w_vals).pvalue) if (m != "EgoLite" and len(w_vals) >= 3) else None
            summary["methods_window_level"][m] = {
                "n": len(w_vals),
                "median_spread_mm": float(np.median(w_vals)),
                "mean_spread_mm": float(np.mean(w_vals)),
                "median_resp_power": float(np.median(r_pows)),
                "wilcoxon_p_vs_egolite": p_w,
            }
            s_vals = [float(np.mean([w[m]["spread_mm"] for w in windows if w["seq"] == s and m in w])) for s in seq_ids_with]
            ego_s = [float(np.mean([w["EgoLite"]["spread_mm"] for w in windows if w["seq"] == s])) for s in seq_ids_with]
            p_s = float(wilcoxon(ego_s, s_vals).pvalue) if (m != "EgoLite" and len(s_vals) >= 3) else None
            summary["methods_sequence_clustered"][m] = {
                "n_seq": len(s_vals),
                "median_of_seq_means_mm": float(np.median(s_vals)),
                "mean_of_seq_means_mm": float(np.mean(s_vals)),
                "wilcoxon_p_vs_egolite": p_s,
            }
        return {"summary": summary, "windows": windows}

    dec1_1mm = collect_windows(1.0, 20.0)
    dec1_05mm = collect_windows(0.5, 20.0)

    # --- 2. Decision 4: Windowed Sim(3) scale drift diagnostics ---
    win_diag = {}
    for m in ["EgoLite", "Same-corpus PoseNet", "EndoDAC", "AF-SfMLearner", "Endo-FASt3r", "DROID-SLAM"]:
        win_diag[m] = {}
        for win_s in [None, 60.0, 30.0, 15.0, 10.0]:
            lbl = "global_sim3" if win_s is None else f"window_{int(win_s)}s"
            ates = []
            se3_gscale_ates = []
            rel_cvs = []
            scale_iqrs = []
            for sd in seq_data:
                gt, fps = sd["gt"], sd["fps"]
                est = sd["raw_est"].get(m)
                if est is None:
                    continue
                n = min(len(est), len(gt))
                sc_g = sd["global_scales"][m]
                if win_s is None:
                    errs = np.linalg.norm(sd["aligned_c"][m] - gt[:n, :3, 3], axis=1)
                    ates.append(float(np.sqrt(np.mean(errs ** 2))))
                    se3_gscale_ates.append(float(np.sqrt(np.mean(errs ** 2))))
                else:
                    W = int(win_s * fps)
                    sq_errs, sq_errs_se3g, rel_scs = [], [], []
                    est_gscaled = est[:n].copy()
                    est_gscaled[:, :3, 3] *= sc_g
                    for a in range(0, n, W):
                        b = min(n, a + W)
                        if b - a < max(20, W // 3):
                            continue
                        al_w, (sc_w, _, _) = align_trajectories(est[a:b], gt[a:b], True)
                        al_w_se3g, _ = align_trajectories(est_gscaled[a:b], gt[a:b], False)
                        sq_errs.append(np.sum((al_w[:, :3, 3] - gt[a:b, :3, 3]) ** 2, axis=1))
                        sq_errs_se3g.append(np.sum((al_w_se3g[:, :3, 3] - gt[a:b, :3, 3]) ** 2, axis=1))
                        if np.isfinite(sc_w) and sc_w > 0 and sc_g > 0:
                            rel_scs.append(float(sc_w / sc_g))
                    ates.append(float(np.sqrt(np.mean(np.concatenate(sq_errs)))))
                    se3_gscale_ates.append(float(np.sqrt(np.mean(np.concatenate(sq_errs_se3g)))))
                    if len(rel_scs) >= 2:
                        arr = np.array(rel_scs)
                        rel_cvs.append(float(arr.std() / max(arr.mean(), 1e-8)))
                        scale_iqrs.append(float(np.percentile(arr, 75) - np.percentile(arr, 25)))
            win_diag[m][lbl] = {
                "mean_ate_mm": float(np.mean(ates)),
                "median_ate_mm": float(np.median(ates)),
                "mean_ate_se3_global_scale_mm": float(np.mean(se3_gscale_ates)),
                "median_ate_se3_global_scale_mm": float(np.median(se3_gscale_ates)),
                "per_seq_ate_mm": [round(x, 3) for x in ates],
                "mean_intra_seq_scale_cv": float(np.mean(rel_cvs)) if rel_cvs else None,
                "mean_intra_seq_scale_iqr": float(np.mean(scale_iqrs)) if rel_cvs else None,
            }

    # Quantify EgoLite decomposition: SE(3) -> global Sim(3) -> piecewise Sim(3)
    # Compute exact fixed-50-mm SE(3) ATE on EgoLite (est translations are in 50-mm depth units)
    se3_ates = []
    for sd in seq_data:
        gt = sd["gt"]
        est_mm = sd["raw_est"]["EgoLite"].copy()
        est_mm[:, :3, 3] *= 50.0
        n = min(len(est_mm), len(gt))
        al_se3, _ = align_trajectories(est_mm[:n], gt[:n], False)
        errs = np.linalg.norm(al_se3[:, :3, 3] - gt[:n, :3, 3], axis=1)
        se3_ates.append(float(np.sqrt(np.mean(errs ** 2))))
    win_diag["EgoLite"]["se3_fixed50"] = {
        "mean_ate_mm": float(np.mean(se3_ates)),
        "median_ate_mm": float(np.median(se3_ates)),
        "per_seq_ate_mm": [round(x, 3) for x in se3_ates],
    }

    res = {
        "decision1_all_near_static_lt1mm": dec1_1mm["summary"],
        "decision1_all_near_static_lt05mm": dec1_05mm["summary"],
        "decision1_windows_lt1mm": dec1_1mm["windows"],
        "decision4_windowed_scale_drift": win_diag,
    }
    out_p = ROOT / "results/revision_vrih/r2_1_r1_2_stereomis_static_and_scale.json"
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n=== DECISION 1 (<1.0 mm GT spread, all non-overlapping 20-s windows) ===")
    print(f"Windows: {dec1_1mm['summary']['n_windows']} across {dec1_1mm['summary']['n_sequences']} sequences; "
          f"GT median spread: {dec1_1mm['summary']['gt_median_spread_mm']:.3f} mm")
    print("Per-seq counts:", {k.replace('stereomis_extracted_StereoMIS_0_0_1_', ''): v
                              for k, v in dec1_1mm['summary']['per_sequence_window_counts'].items()})
    for m, d in dec1_1mm["summary"]["methods_window_level"].items():
        cs = dec1_1mm["summary"]["methods_sequence_clustered"][m]
        print(f"  {m:20s} | win(n={d['n']}): med={d['median_spread_mm']:.2f}, mean={d['mean_spread_mm']:.2f}, "
              f"respP={d['median_resp_power']:.2f}, p={d['wilcoxon_p_vs_egolite']} | "
              f"seq(n={cs['n_seq']}): med={cs['median_of_seq_means_mm']:.2f}, p={cs['wilcoxon_p_vs_egolite']}")

    print("\n=== DECISION 4 (Windowed Sim(3) scale drift on StereoMIS, n=11) ===")
    for m, wd in win_diag.items():
        g = wd["global_sim3"]["mean_ate_mm"]
        w60 = wd["window_60s"]["mean_ate_mm"]
        w30 = wd["window_30s"]["mean_ate_mm"]
        w15 = wd["window_15s"]["mean_ate_mm"]
        w10 = wd["window_10s"]["mean_ate_mm"]
        cv30 = wd["window_30s"]["mean_intra_seq_scale_cv"]
        print(f"  {m:20s}: global={g:5.2f} -> 60s={w60:5.2f} -> 30s={w30:5.2f} (CV={cv30:.2f}) -> 15s={w15:5.2f} -> 10s={w10:5.2f}")
    print(f"  EgoLite SE(3) fixed-50 mean: {win_diag['EgoLite']['se3_fixed50']['mean_ate_mm']:.2f} mm")
    print("Wrote", out_p)
    return res


if __name__ == "__main__":
    run_all()
