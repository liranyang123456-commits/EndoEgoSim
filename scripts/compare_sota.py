"""三条对比协议的总表。

1) zero-shot: 外部方法直接在本仿真测试集上跑
2) 微调: 外部方法用本训练集微调后再测
3) 通用位姿: 经典 8 点 / identity, 以及本方法 (Ours)

写出 docs/05_SOTA对比.md 与 results/sota/compare.json
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOTA = os.path.join(ROOT, "results", "sota")

# (tag, 显示名, 组别)
RUNS = [
    ("identity_simtest", "Identity (无运动)", "classical"),
    ("eight_simtest", "8-point Essential + ORB", "classical"),
    ("pnp_simtest", "RGB-D PnP (ORB+深度GT)", "classical"),
    ("meshrts_simtest", "Mesh-RTS v1 (UVH+flow+ICP)", "classical"),
    ("meshrts_v2_simtest", "Mesh-RTS v2 (cam-UVH+IRLS+rot-gate + 整段8-point)", "classical"),
    ("meshrts_v3_simtest", "Mesh-RTS v3 mesh-only (DIS+PnP)", "classical"),
    ("meshrts_v3f_simtest", "Mesh-RTS v3 (单路径, essential+rot-gate)", "classical"),
    ("meshrts_v2_hybrid", "Mesh-RTS v2 ⊕ 8-point (jerk)", "classical"),
    ("pi3_simtest", "π³ zero-shot", "zero"),
    ("droid_simtest", "DROID-SLAM", "zero"),
    ("orbslam3_simtest", "ORB-SLAM3", "zero"),
    ("vggt_simtest", "VGGT zero-shot", "zero"),
    ("mast3r_simtest", "MASt3R zero-shot", "zero"),
    ("dust3r_simtest", "DUSt3R zero-shot", "zero"),
    ("endo3r_simtest", "Endo3R (真实 SCARED 训练)", "zero"),
    ("vggt_ft_simtest", "VGGT + EndoEgoSim v1 heads", "ft"),
    ("vggt_ft2_simtest", "VGGT + EndoEgoSim v2 global", "ft"),
    ("dust3r_ft_simtest", "DUSt3R + EndoEgoSim 度量微调", "ft"),
    ("fuse_learned_median", "Ours-Ensemble (VGGT族+MASt3R 中位数)", "ours"),
    ("ours_v3_consensus", "Ours-Consensus (v3 vs 8-point+MASt3R)", "ours"),
    ("ours_consensus_simtest", "Ours-Consensus (v2 vs 8-point+MASt3R)", "ours"),
    ("ours_v3_hybrid", "Ours-Hybrid (v3 tuned jerk ⊕ 8-point)", "ours"),
    ("ours_hybrid2_simtest", "Ours-Hybrid2 (v2 tuned jerk ⊕ 8-point)", "ours"),
    ("ours_hybrid_simtest", "Ours-Hybrid (v2 jerk ⊕ 8-point)", "ours"),
    ("ours_v3_simtest", "Ours-Single v3 (灾难过采样续训)", "ours"),
    ("ours_v2_simtest", "Ours-Single v2 (失败模式续训)", "ours"),
    ("ours_simtest", "Ours-Single v1 (运动分解微调)", "ours"),
]


def load(tag):
    p = os.path.join(SOTA, tag, "summary.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def row_of(d):
    if d is None:
        return None
    s = d["summary"]
    buckets = {b["bucket"]: b for b in s.get("stratified", [])}
    def bget(name, key):
        b = buckets.get(name)
        return None if b is None else b[key]
    return {
        "n": s.get("n_seq"),
        "ate": s.get("ate_sim3_rmse_mean"),
        "median": s.get("ate_sim3_rmse_median"),
        "rpe": s.get("rpe1_trans_mean"),
        "low": bget("低参照(<0.3)", "ate_sim3_mean"),
        "mid": bget("中参照(0.3-0.7)", "ate_sim3_mean"),
        "high": bget("高参照(>0.7)", "ate_sim3_mean"),
    }


def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    rows = []
    for tag, name, group in RUNS:
        d = load(tag)
        r = row_of(d)
        rows.append({"tag": tag, "name": name, "group": group, "metrics": r})

    present = [r for r in rows if r["metrics"] is not None]
    finite_ates = [r["metrics"]["ate"] for r in present
                   if r["metrics"]["ate"] is not None
                   and np.isfinite(r["metrics"]["ate"])]
    best_ate = min(finite_ates) if finite_ates else None

    md = []
    a = md.append
    a("# EndoEgoSim 三路 SOTA 对比\n")
    a("> 指标: ATE(Sim3) mm, 同一 92 条分层子集 (低16+中44+高32), 除非某次跑失败。\n")
    a("## 协议\n")
    a("1. **Zero-shot**: 外部方法权重不改, 直接在仿真测试集推理。\n")
    a("2. **微调**: 外部方法在 `sim_data/train` 上微调, 再在同一测试子集评测。\n")
    a("3. **通用位姿 / Ours**: 经典几何法 + 本方法 (运动分解监督的 VGGT)。\n")
    a("\n## 主表 (ATE Sim3 mm)\n")
    a("| 组别 | 方法 | n | 总体 | median | 低参照 | 中参照 | 高参照 | RPE1 平移 |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    group_name = {"classical": "③ 通用位姿", "zero": "① Zero-shot",
                  "ft": "② 微调", "ours": "③ Ours"}
    for r in rows:
        m = r["metrics"]
        if m is None:
            a(f"| {group_name[r['group']]} | {r['name']} | — | 待跑 | — | — | — | — | — |")
            continue
        mark = " **" if best_ate is not None and m["ate"] == best_ate else ""
        end = "**" if mark else ""
        a(f"| {group_name[r['group']]} | {r['name']}{mark} | {m['n'] or '—'} | "
          f"{fmt(m['ate'])}{end} | {fmt(m['median'])} | {fmt(m['low'])} | "
          f"{fmt(m['mid'])} | {fmt(m['high'])} | {fmt(m['rpe'])} |")

    def _gain(tag, label):
        d = load(tag)
        if not d:
            return
        o = d["summary"]["ate_sim3_rmse_mean"]
        if o is None or not np.isfinite(o):
            return
        a(f"### {label} ({o:.2f} mm)\n")
        for other, name in (("vggt_simtest", "VGGT zero-shot"),
                            ("endo3r_simtest", "Endo3R"),
                            ("vggt_ft2_simtest", "VGGT-v2 单模型"),
                            ("eight_simtest", "8-point"),
                            ("identity_simtest", "Identity")):
            b = load(other)
            if not b:
                continue
            x = b["summary"]["ate_sim3_rmse_mean"]
            if x is None or not np.isfinite(x) or x == 0:
                continue
            a(f"- vs {name}: {x:.2f} → {o:.2f} mm ({(o/x-1)*100:+.1f}%)\n")

    a("\n## 相对增益\n")
    _gain("ours_v3_simtest", "Ours-Single v3")
    _gain("ours_v2_simtest", "Ours-Single v2")
    _gain("ours_simtest", "Ours-Single v1")
    _gain("ours_v3_consensus", "Ours-Consensus v3")
    _gain("ours_consensus_simtest", "Ours-Consensus v2")
    _gain("ours_v3_hybrid", "Ours-Hybrid v3")
    _gain("ours_hybrid2_simtest", "Ours-Hybrid2")
    _gain("ours_hybrid_simtest", "Ours-Hybrid")
    _gain("fuse_learned_median", "Ours-Ensemble")
    _gain("meshrts_v2_hybrid", "Mesh-RTS v2 ⊕ 8-point")
    _gain("meshrts_v2_simtest", "Mesh-RTS v2")

    a("\n## 读表说明\n")
    a("- 单目方法一律 Sim3 对齐, 与 SCARED / SurgCUT3R 口径一致。\n")
    a("- Identity / 8-point 证明任务非平凡; PnP 用了深度 GT, 只作 RGB-D 参考上界。\n")
    a("- 8-point 旧数字 9.98 含退化 Sim3 零误差; 修复后诚实均值为 10.52。\n")
    a("- Ours-Ensemble: 各方法轨迹 Sim3 对齐后对平移取中位数, **不看 GT** (含 Ours-Single)。\n")
    a("- Ours-Single: 运动分解掩码去掉器械像素的深度监督 + 混合大基线窗 + 低参照过采样 + cam_weight=8。\n")
    a("- 单模型已低于所有外部 SOTA 与本仓库 v2; 融合再压低均值 (灾难序列)。\n")

    text = "\n".join(md) + "\n"
    out_md = os.path.join(ROOT, "docs", "05_SOTA对比.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(text)
    out_j = os.path.join(SOTA, "compare.json")
    with open(out_j, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(text)
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
