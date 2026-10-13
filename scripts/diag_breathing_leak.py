"""Does respiratory tissue motion leak into the estimated camera motion?

In each StereoMIS sequence, finds the window where the kinematic camera moves
least, runs EgoLite on consecutive frames (every --step-th), and compares the
estimated camera displacement with the ground truth and with the tissue-motion
signal (median observed flow). A leak shows up as estimated camera oscillation
at the respiratory frequency of the flow signal while the true camera is still.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diag_real_gap import run_pairs  # noqa: E402
from endosim.eval.metrics import load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames  # noqa: E402
from endosim.lite.data import load_frames  # noqa: E402
from endosim.lite.egolite import build_from_args, chain  # noqa: E402
from endosim.lite.infer import _intrinsics  # noqa: E402

UNIT = 50.0


def video_fps(seq):
    idx = json.load(open(os.path.join(seq, "color_index.json"), encoding="utf-8"))
    cap = cv2.VideoCapture(idx["path"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 1 else 30.0


def peak(x, fs, lo=0.05, hi=3.0):
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    band = (f >= lo) & (f <= hi)
    k = np.argmax(spec * band)
    resp = (f >= 0.1) & (f <= 0.6)
    return float(f[k]), float(spec[resp].sum() / max(spec[band].sum(), 1e-12))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/egolite/s_big_ft/last.pth")
    ap.add_argument("--list", default="lists/stereomis.txt")
    ap.add_argument("--window-s", type=float, default=20.0)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--out", default="notes/diag_breathing_leak.json")
    ap.add_argument("--fig", default="results/egolite/breathing_leak.png")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_from_args(ck["args"])
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()
    seqs = [l.strip() for l in open(args.list, encoding="utf-8") if l.strip() and not l.startswith("#")]
    rows, curves = [], []
    for d in seqs:
        gt_all = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
        fps = video_fps(d)
        W = int(args.window_s * fps)
        if len(gt_all) < W + 1:
            continue
        c = gt_all[:, :3, 3]
        best, a0 = None, 0
        for a in range(0, len(gt_all) - W, max(1, W // 10)):
            seg = c[a:a + W]
            spread = float(np.linalg.norm(seg - seg.mean(0), axis=1).max())
            if best is None or spread < best:
                best, a0 = spread, a
        idx = np.arange(a0, a0 + W + 1, args.step)
        imgs, phs, _ = load_frames(list_color_frames(d, indices=idx))
        K = torch.from_numpy(_intrinsics(d)).cuda()
        o = run_pairs(model, imgs, phs, K)
        T = chain(o["R"], o["t"], torch.ones(len(idx) - 1, device="cuda")).cpu().numpy()
        est = T[:, :3, 3] * UNIT
        gt = (np.linalg.inv(gt_all[idx[0]])[None] @ gt_all[idx])[:, :3, 3]
        fs = fps / args.step
        flow_sig = o["f_obs"].norm(dim=1).flatten(1).median(1).values.cpu().numpy()
        e0 = est - est.mean(0)
        axis = np.linalg.svd(e0, full_matrices=False)[2][0]
        est_1d = e0 @ axis
        f_est, r_est = peak(est_1d, fs)
        f_flow, r_flow = peak(flow_sig, fs)
        row = {"seq": os.path.basename(d.rstrip("/\\")), "fps": fps, "start": int(a0),
               "gt_spread_mm": best, "gt_std_mm": float(np.linalg.norm(gt - gt.mean(0), axis=1).std()),
               "est_spread_mm": float(np.linalg.norm(e0, axis=1).max()),
               "est_peak_hz": f_est, "est_resp_power": r_est,
               "flow_peak_hz": f_flow, "flow_resp_power": r_flow,
               "flow_med_px": float(np.median(flow_sig))}
        rows.append(row)
        curves.append((row["seq"], np.arange(len(est_1d)) / fs, est_1d,
                       (gt - gt.mean(0)) @ axis, flow_sig))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    json.dump({"ckpt": args.ckpt, "rows": rows}, open(args.out, "w", encoding="utf-8"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = curves if len(curves) <= 3 else sorted(curves, key=lambda c: -np.ptp(c[2]))[:3]
    fig, axes = plt.subplots(len(show), 1, figsize=(9, 2.6 * len(show)), squeeze=False)
    for ax, (name, tt, e, g, fl) in zip(axes[:, 0], show, ):
        ax.plot(tt, e, label="EgoLite camera (mm)", color="tab:red")
        ax.plot(tt[: len(g)], g, label="kinematic GT camera (mm)", color="k")
        ax2 = ax.twinx()
        ax2.plot(tt[: len(fl)], fl, color="tab:blue", alpha=0.4, label="median flow (px)")
        ax.set_title(name.replace("stereomis_extracted_", ""), fontsize=9)
        ax.set_xlabel("time (s)")
        ax.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=120)
    print("figure:", args.fig)


if __name__ == "__main__":
    main()
