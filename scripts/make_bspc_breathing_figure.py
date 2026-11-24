"""Create the real near-static-window respiratory-leakage figure (`fig_breathing.png`).

Uses all automatically detected non-overlapping 20-s near-static camera windows (GT spread < 1.0 mm,
yielding n=23 windows across 7 StereoMIS sequences; identical at < 0.5 mm) from
results/revision_vrih/r2_1_r1_2_stereomis_static_and_scale.json across all 7 monocular estimators
plus kinematic ground truth. Spreads are measured in GT millimetres after whole-sequence Sim(3) alignment.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "results/revision_vrih/r2_1_r1_2_stereomis_static_and_scale.json").read_text(encoding="utf-8"))
wins = D["decision1_windows_lt1mm"]
summ = D["decision1_all_near_static_lt1mm"]

specs = [
    ("gt", "Kinematic GT", "#333333"),
    ("EgoLite", "EgoLite", "#2878B5"),
    ("Endo-FASt3r", "Endo-FASt3r", "#E67E22"),
    ("AF-SfMLearner", "AF-SfMLearner", "#B07AA1"),
    ("EndoDAC", "EndoDAC", "#59A14F"),
    ("Same-corpus PoseNet", "Same-corpus\nPoseNet", "#17A2B8"),
    ("DROID-SLAM", "DROID-SLAM", "#9C755F"),
    ("ORB-SLAM3", "ORB-SLAM3", "#E15759"),
]

vals: dict[str, list[float]] = {k: [] for k, _, _ in specs}
for w in wins:
    vals["gt"].append(w["gt_spread_mm"])
    for k, _, _ in specs[1:]:
        if k in w:
            vals[k].append(w[k]["spread_mm"])

fig, ax = plt.subplots(figsize=(11.2, 4.6), dpi=240)
rng = np.random.default_rng(42)
for j, (k, lbl, col) in enumerate(specs):
    arr = np.array(vals[k], dtype=float)
    jitter = np.linspace(-0.16, 0.16, len(arr)) + rng.uniform(-0.02, 0.02, len(arr))
    med = float(np.median(arr))
    p_str = ""
    if k not in ("gt", "EgoLite"):
        pw = summ["methods_window_level"][k]["wilcoxon_p_vs_egolite"]
        if pw is not None and pw < 0.05:
            p_str = f"*"
    ax.scatter(
        np.full(len(arr), j) + jitter,
        arr,
        s=22,
        color=col,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.35,
        label=f"{lbl.replace(chr(10), ' ')} (med {med:.2f} mm{p_str})",
        zorder=3,
    )
    ax.plot([j - 0.25, j + 0.25], [med, med], color=col, lw=2.2, zorder=4)

ax.set_yscale("log")
ax.set_xticks(range(len(specs)), [lbl for _, lbl, _ in specs], fontsize=8.5)
ax.set_ylabel("Maximum camera-position spread in 20-s window (mm)", fontsize=8.8)
ax.set_title(
    f"Automatically detected near-static StereoMIS windows (n={len(wins)} across {summ['n_sequences']} sequences; "
    "whole-sequence Sim(3)-aligned)",
    fontsize=9.5,
)
ax.set_ylim(0.008, 120.0)
ax.grid(axis="y", which="both", alpha=0.25)
ax.legend(fontsize=7.1, loc="upper left", ncol=3, frameon=True, framealpha=0.94)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.text(
    0.01,
    0.01,
    "*Window-level paired Wilcoxon p < 0.05 vs EgoLite (DROID-SLAM p=0.0049 [seq-clustered p=0.047]; Same-corpus PoseNet p=0.008; ORB-SLAM3 p=0.030). Medians shown.",
    fontsize=7.0,
    color="#555555",
)
out = ROOT / "bspc_paper/figs/fig_breathing.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=240)
plt.close(fig)
print("Wrote", out)
