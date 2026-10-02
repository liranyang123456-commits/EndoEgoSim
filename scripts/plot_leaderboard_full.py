"""综合 leaderboard 图：所有数据集 × 所有方法，公平标注协议/样本量/显著性。

每面板一个 cohort；方法按均值排序；我们的方法高亮；标注数值。
协议差异（32/64 帧、完整/部分轨迹、独立/训练暴露）在面板副标题标注。

用法：python scripts/plot_leaderboard_full.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "mia_paper", "fig_leaderboard_full.pdf")

# (cohort 标题, 协议标注, [(方法, ATE, 是否 ours, 显著性标记)])
PANELS = [
    ("simtest92 (dev, 92 seq)", "uniform ≤64f, post-hoc τ", [
        ("MD-VGGT-R", 3.49, True, ""),
        ("Stage 6", 5.49, False, ""),
        ("DROID-SLAM", 6.04, False, ""),
        ("pi3", 6.23, False, ""),
        ("Reloc3r-512", 7.63, False, ""),
        ("Mesh-RTS", 9.61, False, ""),
        ("ORB-E", 10.40, False, ""),
        ("CUT3R", 10.31, False, ""),
        ("VGGT zero-shot", 12.17, False, ""),
        ("Identity", 23.59, False, ""),
    ]),
    ("Frozen core (265 seq)", "locked, no low-ref", [
        ("MD-VGGT-R", 2.695, True, ""),
        ("DROID-SLAM", 3.262, False, "NS"),
        ("Stage 6", 4.428, False, ""),
        ("pi3", 4.648, False, ""),
        ("Reloc3r-512", 5.974, False, ""),
        ("ORB-E", 9.863, False, ""),
        ("CUT3R", 11.413, False, ""),
    ]),
    ("Frozen extension (93 seq)", "locked, all strata", [
        ("Reloc3r-512", 22.765, False, "lower mean"),
        ("MD-VGGT-R", 23.151, True, "lower med+task"),
        ("DROID-SLAM", 24.662, False, ""),
        ("pi3", 28.255, False, ""),
        ("ORB-E", 29.892, False, ""),
        ("Stage 6", 30.250, False, ""),
        ("CUT3R", 30.762, False, ""),
    ]),
    ("StereoMIS 64f (indep., 11 vid)", "in-vivo, independent", [
        ("MD-VGGT-G", 13.438, True, "p=0.019 vs DROID"),
        ("Stage 6", 13.772, False, ""),
        ("DROID-SLAM", 27.677, False, ""),
        ("ORB-E", 30.116, False, ""),
    ]),
    ("EGO-Mo gtraj (18 seg)", "global-GT real capture", [
        ("MD-VGGT-R", 13.54, True, "p=0.003 vs VGGT"),
        ("pi3", 13.66, False, ""),
        ("Reloc3r-512", 13.98, False, ""),
        ("CUT3R", 14.04, False, ""),
        ("VGGT zero-shot", 14.90, False, ""),
    ]),
]


def main() -> None:
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9.5, 3.1 * len(PANELS)))
    for ax, (title, sub, rows) in zip(axes, PANELS):
        rows = sorted(rows, key=lambda r: r[1])
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        colors = ["#00b4d8" if r[2] else "#c9d4de" for r in rows]
        y = np.arange(len(rows))
        ax.barh(y, vals, color=colors, edgecolor="#33404d", height=0.72)
        for i, (v, r) in enumerate(zip(vals, rows)):
            note = f"  {v:.2f}" + (f"  ({r[3]})" if r[3] else "")
            ax.text(v + max(vals) * 0.01, i, note, va="center", fontsize=8.5,
                    fontweight="bold" if r[2] else "normal",
                    color="#0077a3" if r[2] else "#33404d")
        ax.set_yticks(y)
        ax.set_yticklabels([n + ("  ★" if r[2] else "") for n, r in zip(names, rows)], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Sim(3) ATE (mm, lower is better)", fontsize=9)
        ax.set_title(f"{title}   [{sub}]", fontsize=10, loc="left")
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, max(vals) * 1.28)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("MD-VGGT vs reran baselines across cohorts (★ = ours; per-cohort protocol noted)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(OUT)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
