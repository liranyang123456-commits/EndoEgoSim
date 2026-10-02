"""Generate the protocol-separated simtest92 leaderboard.

The primary pool uses uniform sampling of at most 64 frames (actual 40--64)
and complete 92/92 coverage.  Methods evaluated with a separately selected
uniform 32-frame protocol are listed but not ranked with that pool.  Saved
64-frame predictions cannot be post-hoc sliced into the 32-frame protocol
because the selected source-frame indices differ.

用法：python scripts/make_leaderboard.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SOTA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "sota")

# (tag, 显示名, 类别)
METHODS = [
    ("ours_v7_final", "MD-VGGT-R, tau=.12*", "at-most-64"),
    ("ours_v6_decouple_v3", "Stage 6 + route + instrument suppression", "at-most-64"),
    ("ours_v6_infer_simtest", "Stage 6 + route", "at-most-64"),
    ("ours_v6_simtest", "MD-VGGT Stage 6 (global)", "at-most-64"),
    ("droid_simtest", "DROID-SLAM", "at-most-64"),
    ("pi3_simtest", "pi3", "at-most-64"),
    ("reloc3r_simtest", "Reloc3r-512", "at-most-64"),
    ("meshrts_v3_c2w", "Mesh-RTS", "at-most-64"),
    ("cut3r_simtest", "CUT3R", "at-most-64"),
    ("orb_e_c2w_simtest", "ORB-E", "at-most-64"),
    ("vggt_ft2_simtest", "VGGT global+heads FT", "at-most-64"),
    ("vggt_ft_simtest", "VGGT heads FT", "at-most-64"),
    ("vggt_simtest", "VGGT zero-shot", "at-most-64"),
    ("identity_simtest", "Identity", "at-most-64"),
    ("mast3r_simtest", "MASt3R", "trunc32"),
    ("dust3r_simtest", "DUSt3R", "trunc32"),
    ("endo3r_simtest", "Endo3R", "trunc32"),
    ("dust3r_ft_simtest", "DUSt3R metric FT", "trunc32"),
]


def eval_native(tag):
    """Read the metric-revised native-protocol summary."""
    path = os.path.join(SOTA, tag, "summary.json")
    if not os.path.isfile(path):
        return None
    payload = json.load(open(path, encoding="utf-8"))
    summary = payload["summary"]
    records = [r for r in payload.get("records", []) if "error" not in r]
    lengths = [r.get("n_frames_used", r.get("n_frames", 0)) for r in records]
    return dict(
        ate=float(summary["ate_sim3_rmse_mean"]),
        med=float(summary["ate_sim3_rmse_median"]),
        medlen=int(np.median(lengths)) if lengths else 0,
        n=int(summary["n_seq"]),
    )


def main():
    native = {}
    for tag, name, cat in METHODS:
        r = eval_native(tag)
        if r:
            native[tag] = (name, cat, r)

    lines = []
    lines.append("# simtest92 排行榜（协议分层）\n")
    lines.append("ATE = Sim3 RMSE (mm)，越低越好。口径见文末说明。\n")

    lines.append("\n## 统一最多 64 帧协议（实际 40~64，中位 61；92/92 完整覆盖）\n")
    lines.append("| 排名 | 方法 | ATE_sim3 | median | 帧数 |")
    lines.append("|---|---|---|---|---|")
    full = [(t, n_, c, r) for t, (n_, c, r) in native.items()
            if c == "at-most-64" and r["n"] == 92]
    full.sort(key=lambda x: x[3]["ate"])
    for i, (t, name, cat, r) in enumerate(full, 1):
        star = " ★" if i == 1 else ""
        lines.append(f"| {i} | {name}{star} | **{r['ate']:.3f}** | "
                     f"{r['med']:.3f} | {r['medlen']} |")

    lines.append("\n## 独立 32 帧协议（仅供对照，与主池不可直接比）\n")
    lines.append("| 方法 | ATE_sim3 | median | 帧数 |")
    lines.append("|---|---|---|---|")
    tr = [(t, n_, c, r) for t, (n_, c, r) in native.items() if c == "trunc32"]
    tr.sort(key=lambda x: x[3]["ate"])
    for t, name, cat, r in tr:
        lines.append(f"| - | {name} | {r['ate']:.3f} | {r['med']:.3f} | {r['medlen']} |")

    lines.append("\n## 口径说明\n")
    lines.append("- MASt3R/DUSt3R/Endo3R 使用独立选择的 uniform 32 帧，"
                 "与最多 64 帧主池分开列示。")
    lines.append("- 保存的最多 64 帧轨迹不能事后截成 32 帧协议：两种 uniform "
                 "采样选择的原始帧索引不同，必须重跑模型才能形成严格同帧对照。")
    lines.append("- MD-VGGT-R 是主池最低值（3.495），但 hop_cut=0.12 是在 simtest92 "
                 "上后验选择的；论文中标为 benchmark-tuned，需独立留出集确认。\n")

    out = os.path.join(SOTA, "LEADERBOARD.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
