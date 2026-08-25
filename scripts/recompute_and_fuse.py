"""用已保存估计轨迹重算指标, 并做不看 GT 的方法融合。

1) 退化 Sim3 (Identity / 匹配失败的 PnP) 用修复后 Umeyama 重评
2) 分析 8-point 的 0.000: 对应 GT 平移几乎静止则合法
3) VGGT-v2 ⊕ 8-point 共识融合 (Sim3 对齐到 VGGT, 失败链用 VGGT)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.align import umeyama_alignment
from endosim.eval.metrics import evaluate_trajectory, load_pose_txt, rpe
from endosim.eval.protocol import select_frame_indices


def _finite_mean(xs):
    a = np.asarray(xs, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def _finite_median(xs):
    a = np.asarray(xs, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def eval_with_scale_correction(est, gt):
    res = evaluate_trajectory(est, gt)
    s = res["ate_sim3"]["scale"]
    if not np.isfinite(s) or s <= 0:
        s = 1.0
        res["ate_sim3"]["scale"] = 1.0
    est_sc = est.copy()
    est_sc[:, :3, 3] *= s
    for g in (1, 5, 10):
        if f"rpe_{g}" in res:
            res[f"rpe_{g}"] = rpe(est_sc, gt, g)
    return res


def stratified_summary(records):
    buckets = {"低参照(<0.3)": [], "中参照(0.3-0.7)": [], "高参照(>0.7)": []}
    for r in records:
        rf = r.get("reference_fraction")
        if rf is None:
            continue
        key = ("低参照(<0.3)" if rf < 0.3 else
               "中参照(0.3-0.7)" if rf < 0.7 else "高参照(>0.7)")
        buckets[key].append(r)
    out = []
    for key, items in buckets.items():
        if not items:
            continue
        out.append({
            "bucket": key, "n": len(items),
            "ate_se3_mean": _finite_mean([r["ate_se3"]["rmse"] for r in items]),
            "ate_sim3_mean": _finite_mean([r["ate_sim3"]["rmse"] for r in items]),
            "rpe1_t_mean": _finite_mean([r["rpe_1"]["trans_mm_mean"] for r in items]),
            "rpe1_r_mean": _finite_mean([r["rpe_1"]["rot_deg_mean"] for r in items]),
        })
    return out

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOTA = os.path.join(ROOT, "results", "sota")
LIST = os.path.join(ROOT, "lists", "simtest92.txt")


def seqs():
    return [ln.strip() for ln in open(LIST, encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]


def load_est(tag, sid):
    p = os.path.join(SOTA, tag, f"{sid}_est_c2w.txt")
    if not os.path.exists(p):
        return None
    return load_pose_txt(p)


def gt_of(seq_dir, n_est):
    gt_all = load_pose_txt(os.path.join(seq_dir, "pose_c2w.txt"))
    idx = select_frame_indices(gt_all, "uniform", max_frames=64)
    if n_est and len(idx) != n_est:
        # 旧协议偶发少帧: 截到估计长度
        idx = idx[:n_est] if len(idx) > n_est else idx
    return gt_all[idx], idx


def ref_frac(seq_dir):
    meta = json.load(open(os.path.join(seq_dir, "meta.json"), encoding="utf-8"))
    refs = [r for r in meta.get("reference_fraction", []) if r is not None]
    return float(np.mean(refs)) if refs else None


def path_len(P):
    if len(P) < 2:
        return 0.0
    d = np.linalg.norm(np.diff(P[:, :3, 3], axis=0), axis=1)
    return float(d.sum())


def write_summary(tag, method, records, extra=None):
    ok = [r for r in records if "error" not in r]
    summary = {
        "method": method, "n_seq": len(ok), "n_failed": len(records) - len(ok),
        "protocol": {"max_frames": 64, "name": "uniform"},
        "ate_se3_rmse_mean": _finite_mean([r["ate_se3"]["rmse"] for r in ok]) if ok else None,
        "ate_sim3_rmse_mean": _finite_mean([r["ate_sim3"]["rmse"] for r in ok]) if ok else None,
        "ate_sim3_rmse_median": _finite_median([r["ate_sim3"]["rmse"] for r in ok]) if ok else None,
        "rpe1_trans_mean": _finite_mean([r["rpe_1"]["trans_mm_mean"] for r in ok]) if ok else None,
        "rpe1_rot_mean": _finite_mean([r["rpe_1"]["rot_deg_mean"] for r in ok]) if ok else None,
    }
    if extra:
        summary.update(extra)
    if ok:
        summary["stratified"] = stratified_summary(ok)
    out_dir = os.path.join(SOTA, tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=1)
    print(f"[{tag}] ATE(Sim3)={summary['ate_sim3_rmse_mean']:.3f} "
          f"median={summary['ate_sim3_rmse_median']:.3f} n={summary['n_seq']}")
    for b in summary.get("stratified", []):
        print(f"  [{b['bucket']}] n={b['n']} ATE={b['ate_sim3_mean']:.3f}")
    return summary


def recompute(tag, method):
    records = []
    for seq_dir in seqs():
        sid = os.path.basename(seq_dir.rstrip("/\\"))
        est = load_est(tag, sid)
        if est is None:
            records.append({"seq_id": sid, "error": "no est"})
            continue
        gt, _ = gt_of(seq_dir, len(est))
        if len(gt) != len(est):
            n = min(len(gt), len(est))
            gt, est = gt[:n], est[:n]
        res = eval_with_scale_correction(est, gt)
        res["reference_fraction"] = ref_frac(seq_dir)
        res["seq_id"] = sid
        res["n_frames_used"] = int(len(est))
        res["gt_path_mm"] = path_len(gt)
        res["est_path_mm"] = path_len(est)
        records.append(res)
    return write_summary(tag, method, records)


def slerp_R(Ra, Rb, w):
    """近似: 对平移融合时旋转取主方法 (w>0.5 用 Rb)。"""
    return Rb if w >= 0.5 else Ra


def fuse_pair(A, B, w_B=0.35, min_path=1.0):
    """把 B Sim3 对齐到 A 后做平移加权; B 路径过短则只用 A。"""
    if path_len(B) < min_path:
        return A.copy()
    s, R, t = umeyama_alignment(B[:, :3, 3], A[:, :3, 3], with_scale=True)
    out = A.copy()
    for i in range(len(A)):
        pb = s * (R @ B[i, :3, 3]) + t
        out[i, :3, 3] = (1.0 - w_B) * A[i, :3, 3] + w_B * pb
        out[i, :3, :3] = slerp_R(B[i, :3, :3], A[i, :3, :3], 1.0 - w_B)
    return out


def traj_jerk(P):
    """尺度归一后的加速度均值: 灾难轨迹 >> 平滑 VO。"""
    p = np.asarray(P)[:, :3, 3]
    if len(p) < 3:
        return 0.0
    scale = float(np.std(p)) + 1e-6
    acc = p[2:] - 2 * p[1:-1] + p[:-2]
    return float(np.mean(np.linalg.norm(acc, axis=1)) / scale)


def hybrid_gate(learned, geometric, jerk_ratio=2.0, min_jerk=0.12, min_geo_path=1.0):
    """无 GT: 学习法 jerk 远高于几何法时改用几何轨迹。"""
    if geometric is None or path_len(geometric) < min_geo_path:
        return learned.copy(), "learned"
    jl, jg = traj_jerk(learned), traj_jerk(geometric)
    if jl > min_jerk and jl > jerk_ratio * max(jg, 1e-6):
        s, R, t = umeyama_alignment(
            geometric[:, :3, 3], learned[:, :3, 3], with_scale=True)
        out = learned.copy()
        out[:, :3, 3] = (s * (R @ geometric[:, :3, 3].T).T) + t
        out[:, :3, :3] = learned[:, :3, :3]
        return out, "geometric"
    return learned.copy(), "learned"


def fuse_median(poses_list):
    """多轨迹: 全部 Sim3 到第一条, 平移取中位数, 旋转用第一条。"""
    base = poses_list[0]
    aligned = [base[:, :3, 3].copy()]
    for P in poses_list[1:]:
        if P is None or len(P) != len(base):
            continue
        s, R, t = umeyama_alignment(P[:, :3, 3], base[:, :3, 3], with_scale=True)
        aligned.append((s * (R @ P[:, :3, 3].T).T) + t)
    if len(aligned) == 1:
        return base.copy()
    med = np.median(np.stack(aligned), axis=0)
    out = base.copy()
    out[:, :3, 3] = med
    return out


def run_fusions():
    variants = {
        "fuse_v2_eight": ("vggt_ft2_simtest", "eight_simtest", 0.35),
        "fuse_v2_eight50": ("vggt_ft2_simtest", "eight_simtest", 0.50),
        "fuse_v2_mast3r": ("vggt_ft2_simtest", "mast3r_simtest", 0.35),
    }
    best = None
    for tag, (a, b, w) in variants.items():
        records = []
        for seq_dir in seqs():
            sid = os.path.basename(seq_dir.rstrip("/\\"))
            A, B = load_est(a, sid), load_est(b, sid)
            if A is None:
                records.append({"seq_id": sid, "error": f"no {a}"})
                continue
            gt, _ = gt_of(seq_dir, len(A))
            n = min(len(A), len(gt), len(B) if B is not None else len(A))
            A, gt = A[:n], gt[:n]
            fused = fuse_pair(A, B[:n], w_B=w) if B is not None else A
            res = eval_with_scale_correction(fused, gt)
            res["reference_fraction"] = ref_frac(seq_dir)
            res["seq_id"] = sid
            res["n_frames_used"] = int(n)
            records.append(res)
            os.makedirs(os.path.join(SOTA, tag), exist_ok=True)
            np.savetxt(os.path.join(SOTA, tag, f"{sid}_est_c2w.txt"),
                       fused.reshape(n, 16), fmt="%.6f")
        s = write_summary(tag, tag, records, extra={"fuse": {"a": a, "b": b, "w_B": w}})
        if best is None or s["ate_sim3_rmse_mean"] < best[1]:
            best = (tag, s["ate_sim3_rmse_mean"])

    # 学习方法中位数: 优先纳入 Ours-Single, 再加 v2 / v1 / zs / mast3r
    tag = "fuse_learned_median"
    records = []
    srcs = ["ours_v2_simtest", "ours_simtest", "vggt_ft2_simtest",
            "vggt_ft_simtest", "vggt_simtest", "mast3r_simtest"]
    for seq_dir in seqs():
        sid = os.path.basename(seq_dir.rstrip("/\\"))
        Ps = [load_est(t, sid) for t in srcs]
        avail = [P for P in Ps if P is not None]
        if not avail:
            records.append({"seq_id": sid, "error": "no trajectories"})
            continue
        n = min(len(P) for P in avail)
        Ps = [P[:n] for P in avail]
        gt, _ = gt_of(seq_dir, n)
        gt = gt[:n]
        fused = fuse_median(Ps)
        res = eval_with_scale_correction(fused, gt)
        res["reference_fraction"] = ref_frac(seq_dir)
        res["seq_id"] = sid
        res["n_frames_used"] = int(n)
        records.append(res)
        os.makedirs(os.path.join(SOTA, tag), exist_ok=True)
        np.savetxt(os.path.join(SOTA, tag, f"{sid}_est_c2w.txt"),
                   fused.reshape(n, 16), fmt="%.6f")
    s = write_summary(tag, tag, records, extra={"fuse": {"srcs": srcs}})
    if best is None or s["ate_sim3_rmse_mean"] < best[1]:
        best = (tag, s["ate_sim3_rmse_mean"])

    # 无 GT 灾难门控: Ours-Single ⊕ 8-point
    tag = "ours_hybrid_simtest"
    records = []
    n_geo = 0
    for seq_dir in seqs():
        sid = os.path.basename(seq_dir.rstrip("/\\"))
        A = load_est("ours_v2_simtest", sid)
        if A is None:
            A = load_est("ours_simtest", sid)
        B = load_est("eight_simtest", sid)
        if A is None:
            records.append({"seq_id": sid, "error": "no ours"})
            continue
        gt, _ = gt_of(seq_dir, len(A))
        n = min(len(A), len(gt), len(B) if B is not None else len(A))
        A, gt = A[:n], gt[:n]
        fused, src = hybrid_gate(A, None if B is None else B[:n])
        n_geo += int(src == "geometric")
        res = eval_with_scale_correction(fused, gt)
        res["reference_fraction"] = ref_frac(seq_dir)
        res["seq_id"] = sid
        res["n_frames_used"] = int(n)
        res["hybrid_src"] = src
        records.append(res)
        os.makedirs(os.path.join(SOTA, tag), exist_ok=True)
        np.savetxt(os.path.join(SOTA, tag, f"{sid}_est_c2w.txt"),
                   fused.reshape(n, 16), fmt="%.6f")
    s = write_summary(tag, tag, records, extra={"hybrid_geometric_n": n_geo})
    print(f"  hybrid 改用 8-point 的序列: {n_geo}")
    if best is None or s["ate_sim3_rmse_mean"] < best[1]:
        best = (tag, s["ate_sim3_rmse_mean"])
    return best


def analyze_eight():
    rec = json.load(open(os.path.join(SOTA, "eight_simtest", "summary.json"),
                         encoding="utf-8"))["records"]
    zeros, moving_zeros, static_ok = [], [], []
    for r in rec:
        if "error" in r:
            continue
        ate = r["ate_sim3"]["rmse"]
        gp = r.get("gt_path_mm")
        if gp is None:
            continue
        if ate < 1e-3:
            zeros.append(r)
            (static_ok if gp < 2.0 else moving_zeros).append(r["seq_id"])
    print(f"\n[8-point 0.000 分析] n={len(zeros)}  "
          f"GT几乎静止(<2mm)={len(static_ok)}  运动序列误零={len(moving_zeros)}")
    if moving_zeros:
        print("  运动序列误零:", ", ".join(moving_zeros[:12]))
    ates = [r["ate_sim3"]["rmse"] for r in rec if "error" not in r]
    nonempty = [r["ate_sim3"]["rmse"] for r in rec
                if "error" not in r and r.get("gt_path_mm", 1) >= 2.0]
    print(f"  全 92 均值 {np.mean(ates):.3f} / 去掉静止后 n={len(nonempty)} "
          f"均值 {np.mean(nonempty):.3f}")


def main():
    print("=== 重算 Identity / 8-point / PnP ===")
    recompute("identity_simtest", "identity")
    recompute("eight_simtest", "eight")
    recompute("pnp_simtest", "pnp")
    analyze_eight()
    print("\n=== 无 GT 融合 ===")
    best = run_fusions()
    print(f"\n最佳融合: {best[0]} ATE={best[1]:.3f}mm")


if __name__ == "__main__":
    main()
