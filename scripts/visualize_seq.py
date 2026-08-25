"""序列可视化: 帧拼接图 + 轨迹3D图 + 深度/mask 预览。

用法: python scripts/visualize_seq.py --seq sim_data/train/seq_00000001
输出: <seq>/_preview.png, <seq>/_trajectory.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.metrics import load_pose_txt


def montage(seq_dir: str, n_show: int = 8) -> np.ndarray:
    color_files = sorted(glob.glob(os.path.join(seq_dir, "color", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(seq_dir, "depth", "*.png")))
    idxs = np.linspace(0, len(color_files) - 1, min(n_show, len(color_files))).astype(int)
    rows = []
    for i in idxs:
        c = cv2.imread(color_files[i])
        d = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(np.float64)
        dn = np.zeros_like(d)
        m = d > 0
        if m.any():
            dn[m] = ((d[m] - d[m].min()) / max(float(np.ptp(d[m])), 1) * 255)
        d_rgb = cv2.applyColorMap(dn.astype(np.uint8), cv2.COLORMAP_TURBO)
        mp = os.path.join(seq_dir, "object_mask", os.path.basename(color_files[i]).replace(".png", ".png"))
        if os.path.exists(mp):
            mk = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
            mk_rgb = np.zeros((*mk.shape, 3), np.uint8)
            for k, col in [(1, (90, 60, 60)), (2, (0, 200, 255)), (3, (255, 80, 80))]:
                mk_rgb[mk == k] = col
        else:
            mk_rgb = np.zeros_like(c)
        rows.append(np.hstack([c, d_rgb, mk_rgb]))
    return np.vstack(rows)


def trajectory_plot(seq_dir: str, poses: np.ndarray, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    t = poses[:, :3, 3]
    ax1.plot(t[:, 0], t[:, 1], t[:, 2], "b-", lw=2)
    ax1.scatter(*t[0], c="g", s=80, label="start")
    ax1.scatter(*t[-1], c="r", s=80, label="end")
    for i in range(0, len(poses), max(len(poses) // 8, 1)):
        R = poses[i, :3, :3]
        for ax, col in [(0, "r"), (1, "g"), (2, "b")]:
            d = R[:, ax] * 12.0
            ax1.plot([t[i, 0], t[i, 0] + d[0]], [t[i, 1], t[i, 1] + d[1]],
                     [t[i, 2], t[i, 2] + d[2]], col, lw=1)
    ax1.set_xlabel("X (mm)"); ax1.set_ylabel("Y (mm)"); ax1.set_zlabel("Z (mm)")
    ax1.legend(); ax1.set_title("Camera trajectory (world = first frame)")
    ax2 = fig.add_subplot(122)
    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    ax2.plot(steps, "k-")
    ax2.set_xlabel("frame"); ax2.set_ylabel("step (mm)")
    ax2.set_title("Per-frame translation")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    args = ap.parse_args()
    poses = load_pose_txt(os.path.join(args.seq, "pose_c2w.txt"))
    m = montage(args.seq)
    p1 = os.path.join(args.seq, "_preview.png")
    cv2.imwrite(p1, m)
    p2 = os.path.join(args.seq, "_trajectory.png")
    trajectory_plot(args.seq, poses, p2)
    print(f"已输出:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
