"""汇总所有 SOTA 基线结果为对比报告（real_test 分源 + sim test 分层）。"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = "results/sota"


def load(tag):
    p = os.path.join(ROOT, tag, "summary.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def agg(records, key="ate_sim3", sub="rmse"):
    vals = [r[key][sub] for r in records if "error" not in r]
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def main():
    print("=" * 78)
    print("EndoEgoSim SOTA 基线评测汇总 (VGGT / MASt3R / DUSt3R, zero-shot)")
    print("=" * 78)

    # ---- real_test 分源 ----
    print("\n[real_test 真实数据锚定]  ATE(Sim3) mm / RPE1 平移 mm")
    print(f"  {'方法':<8}{'C3VD(平滑视频)':>22}{'SCARED(关键帧)':>22}")
    for method in ("vggt", "mast3r", "dust3r", "endo3r"):
        d = load(f"{method}_real")
        if d is None:
            continue
        recs = [r for r in d["records"] if "error" not in r]
        c3vd = [r for r in recs if r["seq_id"].startswith(("c1_", "c2_", "cecum"))]
        scared = [r for r in recs if r["seq_id"].startswith("scared")]
        a1, n1 = agg(c3vd)
        a2, n2 = agg(scared)
        print(f"  {method.upper():<8}{a1:>14.1f} (n={n1}){a2:>14.1f} (n={n2})")

    # ---- sim test 分层 ----
    print("\n[sim test 仿真分层]  ATE(Sim3) mm / RPE1 平移 mm / RPE1 旋转 °")
    print(f"  {'方法':<8}{'低参照<0.3':>24}{'中参照0.3-0.7':>24}{'高参照>0.7':>24}")
    rows = {}
    for method in ("vggt", "mast3r", "dust3r", "endo3r", "vggt_ft", "vggt_ft2", "dust3r_ft"):
        d = load(f"{method}_simtest")
        if d is None:
            continue
        recs = [r for r in d["records"] if "error" not in r]
        cells = []
        for lo, hi in ((0.0, 0.3), (0.3, 0.7), (0.7, 1.01)):
            rs = [r for r in recs
                  if r.get("reference_fraction") is not None
                  and lo <= r["reference_fraction"] < hi]
            a, n = agg(rs)
            rt = float(np.mean([r["rpe_1"]["trans_mm_mean"] for r in rs])) if rs else float("nan")
            cells.append((a, rt, n))
        rows[method] = cells
        print(f"  {method.upper():<8}" + "".join(
            f"{a:>10.1f}/{rt:>5.1f} (n={n})" for a, rt, n in cells))
    print("  (格式: ATE(Sim3)/RPE1_t)")

    # ---- 总体 ----
    print("\n[总体均值]")
    for method in ("vggt", "mast3r", "dust3r", "endo3r", "vggt_ft", "vggt_ft2", "dust3r_ft"):
        d = load(f"{method}_simtest")
        if d is None:
            continue
        s = d["summary"]
        print(f"  {method.upper():<8} ATE(Sim3) {s['ate_sim3_rmse_mean']:>8.2f}mm  "
              f"median {s['ate_sim3_rmse_median']:>8.2f}mm  "
              f"RPE1 t {s['rpe1_trans_mean']:>6.2f}mm r {s['rpe1_rot_mean']:>5.2f}°  "
              f"({s['total_time_sec']/max(s['n_seq'],1):.0f}s/seq)")


if __name__ == "__main__":
    main()
