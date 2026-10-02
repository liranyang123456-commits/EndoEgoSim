"""gtraj 多方法轨迹对比图（论文用）：所有方法 + GT，三平面正交投影（绝对 mm）。

用法：python scripts/plot_gtraj_comparison.py --session gtraj_20261002_002134 --seg segment_00_0s_17s
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from endosim.eval.metrics import align_trajectories, load_pose_txt  # noqa: E402

SOTA = os.path.join(ROOT, "results", "sota")
METHODS = [
    ("ours_v7_egomo_gtraj", "MD-VGGT-R (ours)", "#00b4d8", 2.6),
    ("pi3_egomo_gtraj", "pi3", "#7bd389", 1.6),
    ("reloc3r_egomo_gtraj", "Reloc3r-512", "#f5a623", 1.6),
    ("cut3r_egomo_gtraj", "CUT3R", "#bd7be8", 1.6),
    ("vggt_egomo_gtraj", "VGGT zero-shot", "#e06c9f", 1.6),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="gtraj_20261002_002134")
    ap.add_argument("--seg", default="segment_00_0s_17s")
    ap.add_argument("--out", default=os.path.join(ROOT, "mia_paper", "fig_gtraj_traj.pdf"))
    args = ap.parse_args()

    seg_dir = os.path.join(ROOT, "sim_data", "egomo_gtraj", args.session, args.seg)
    gt = load_pose_txt(os.path.join(seg_dir, "pose_c2w.txt"))
    gtc = gt[:, :3, 3]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    planes = [(0, 1, "X (mm)", "Y (mm)"), (0, 2, "X (mm)", "Z (mm)"), (2, 1, "Z (mm)", "Y (mm)")]
    for ax, (a0, a1, xl, yl) in zip(axes, planes):
        ax.plot(gtc[:, a0], gtc[:, a1], color="#222222", lw=3.0, label="Ground truth", zorder=5)
        for tag, name, color, wd in METHODS:
            p = os.path.join(SOTA, tag, f"{args.seg}_est_c2w.txt")
            if not os.path.isfile(p):
                continue
            est = np.loadtxt(p).reshape(-1, 4, 4)
            n = min(len(est), len(gt))
            aligned, _ = align_trajectories(est[:n], gt[:n], with_scale=True)
            c = aligned[:, :3, 3]
            ax.plot(c[:, a0], c[:, a1], color=color, lw=wd, label=name, alpha=0.9)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_title(f"{'XY' if (a0, a1) == (0, 1) else 'XZ' if (a0, a1) == (0, 2) else 'ZY'} plane")
    axes[0].legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle(f"Absolute camera trajectory on real capture ({args.seg.replace('_', ' ')}, GT mm coords)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=200)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
