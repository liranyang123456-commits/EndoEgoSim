"""MD-VGGT 推理路由评测：一个 v6 权重，不新训。

  # 仿真 64 帧
  python scripts/eval_mdvggt_infer.py --list lists/simtest92.txt \\
      --ckpt results/finetune/ours_vggt_v6/ckpt_200.pth \\
      --tag ours_v6_infer_simtest --protocol clip --max-frames 64 \\
      --reuse-global results/sota/ours_v6_simtest

  # StereoMIS 全帧滑窗防爆
  python scripts/eval_mdvggt_infer.py --list lists/stereomis.txt \\
      --ckpt results/finetune/ours_vggt_v6/ckpt_200.pth \\
      --tag ours_v6_infer_stereomis_full --protocol sliding
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.metrics import load_pose_txt
from endosim.eval.mdvggt_infer import infer_clip, infer_sliding
from endosim.eval.protocol import (
    chain_window_poses_gated,
    list_color_frames,
    motion_stats_of_indices,
    sanitize_poses,
    select_frame_indices,
)
sys.path.insert(0, os.path.dirname(__file__))
from baseline_sota import (  # noqa: E402
    eval_with_scale_correction,
    load_model,
    run_vggt,
    run_vggt_full,
    stratified_summary,
)


def _unit_selftest() -> None:
    n = 16
    idx0 = np.arange(8)
    idx1 = np.arange(8, 16)
    good = np.stack([np.eye(4) for _ in range(8)])
    for i in range(8):
        good[i, :3, 3] = [float(i), 0, 0]
    boom = good.copy()
    boom[3, :3, 3] = [1e70, 0, 0]
    boom[4:] = np.nan
    est, used, stats = chain_window_poses_gated(
        [(idx0, good), (idx1, boom)], with_scale=True)
    assert np.isfinite(est).all(), "gated chain leaked non-finite"
    assert len(used) >= 8
    clean = sanitize_poses(boom)
    assert np.isfinite(clean).all()
    dead = np.full((8, 4, 4), np.nan)
    est2, used2, stats2 = chain_window_poses_gated(
        [(idx0, dead), (idx1, good)], with_scale=True)
    assert np.isfinite(est2).all() and len(used2) >= 8
    print("unit selftest ok", {"n": len(used), **stats, "dead_n": len(used2), **{f"d_{k}": v for k, v in stats2.items()}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="ours_v6_infer")
    ap.add_argument("--out", default="results/sota")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--protocol", default="clip", choices=["clip", "sliding"])
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--window-stride", type=int, default=8)
    ap.add_argument("--pair-window", type=int, default=8)
    ap.add_argument("--pair-stride", type=int, default=4)
    ap.add_argument("--reuse-global", default=None,
                    help="已有全局窗轨迹目录；路由到 global 时直接复用")
    ap.add_argument("--seg-ckpt", default=None,
                    help="器械分割权重；提供时成对链走解耦版本")
    ap.add_argument("--pgo", action="store_true",
                    help="成对链后接位姿图优化（PGO）精化")
    ap.add_argument("--force-pairwise", action="store_true",
                    help="跳过全局推理，所有序列直接走成对链（用于定向精化重跑）")
    ap.add_argument("--ph-input", action="store_true",
                    help="A 方案变体：推理时拼 pseudo-height 第 4 输入通道")
    ap.add_argument("--hop-cut", type=float, default=0.30,
                    help="全局窗 hop 路由阈值；降低可把更多序列送入成对链")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _unit_selftest()
        return

    seqs = [ln.strip() for ln in open(args.list, encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    seqs = [s for s in seqs if os.path.isdir(s)
            and os.path.isfile(os.path.join(s, "pose_c2w.txt"))]
    if args.limit > 0:
        seqs = seqs[:args.limit]
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[mdvggt-infer] n={len(seqs)} protocol={args.protocol} ckpt={args.ckpt}")

    t0 = time.time()
    model = load_model("vggt", args.device, ckpt=args.ckpt,
                       ph_input=getattr(args, "ph_input", False))
    print(f"模型加载完成 ({time.time() - t0:.0f}s)")

    def _run(paths):
        return run_vggt(model, paths, args.device, args)

    def _run_full(paths):
        return run_vggt_full(model, paths, args.device, args)

    records = []
    for i, seq_dir in enumerate(seqs):
        seq_id = os.path.basename(seq_dir.rstrip("/\\"))
        est_path = os.path.join(out_dir, f"{seq_id}_est_c2w.txt")
        t_seq = time.time()
        try:
            gt_all = load_pose_txt(os.path.join(seq_dir, "pose_c2w.txt"))
            if args.skip_existing and os.path.isfile(est_path):
                est = np.loadtxt(est_path).reshape(-1, 4, 4)
                if args.protocol == "sliding":
                    n_use = min(len(est), len(gt_all))
                    est, idx = est[:n_use], np.arange(n_use)
                    if n_use < len(gt_all) and n_use > 0:
                        # 滑窗可能缺帧：按已写轨迹长度对齐
                        pass
                else:
                    idx = select_frame_indices(
                        gt_all, protocol="uniform", max_frames=args.max_frames)
                    n = min(len(est), len(idx))
                    est, idx = est[:n], idx[:n]
                info = {"mode": "reuse_file"}
            else:
                if args.protocol == "sliding":
                    frames = list_color_frames(seq_dir)
                    n_use = min(len(frames), len(gt_all))
                    frames, gt_all = frames[:n_use], gt_all[:n_use]
                    est, idx, info = infer_sliding(
                        _run, frames, args.window, args.window_stride,
                        args.pair_window, args.pair_stride,
                        seg_ckpt=args.seg_ckpt, device=args.device,
                        run_full=_run_full, pgo=args.pgo,
                        hop_cut=args.hop_cut)
                    if len(est) == 0:
                        raise RuntimeError("gated sliding produced empty trajectory")
                else:
                    idx = select_frame_indices(
                        gt_all, protocol="uniform", max_frames=args.max_frames)
                    frames = list_color_frames(seq_dir, indices=idx)
                    n = min(len(frames), len(idx))
                    frames, idx = frames[:n], idx[:n]
                    global_est = None
                    if args.reuse_global:
                        cand = os.path.join(args.reuse_global, f"{seq_id}_est_c2w.txt")
                        if os.path.isfile(cand):
                            global_est = np.loadtxt(cand).reshape(-1, 4, 4)
                    est, info = infer_clip(
                        _run, frames, args.pair_window, args.pair_stride,
                        global_est=global_est,
                        seg_ckpt=args.seg_ckpt, device=args.device,
                        run_full=_run_full, pgo=args.pgo, hop_cut=args.hop_cut,
                        force_pairwise=args.force_pairwise)
            n = min(len(est), len(idx))
            est, idx = est[:n], idx[:n]
            gt = gt_all[idx]
            hop = motion_stats_of_indices(gt_all, idx)
            res = eval_with_scale_correction(est, gt)
            res["protocol_hop"] = hop
            meta_p = os.path.join(seq_dir, "meta.json")
            if os.path.isfile(meta_p):
                meta = json.load(open(meta_p, encoding="utf-8"))
                refs = [r for r in meta.get("reference_fraction", []) if r is not None]
                res["reference_fraction"] = float(np.mean(refs)) if refs else None
                res["motion_type"] = meta.get("motion_type")
                res["scene_kind"] = meta.get("scene_kind")
            res["seq_id"] = seq_id
            res["n_frames_used"] = int(len(est))
            res["time_sec"] = round(time.time() - t_seq, 1)
            res["infer"] = info
            records.append(res)
            np.savetxt(est_path, est.reshape(len(est), 16), fmt="%.6f")
            finite = bool(np.isfinite(est).all())
            print(f"[{i+1}/{len(seqs)}] {seq_id}: "
                  f"ATE(Sim3)={res['ate_sim3']['rmse']:.3f}mm "
                  f"scale={res['ate_sim3']['scale']:.3f} "
                  f"RPE1={res['rpe_1']['trans_mm_mean']:.3f} "
                  f"mode={info.get('mode','?')} finite={finite} "
                  f"({res['time_sec']}s)", flush=True)
        except Exception:
            print(f"[{i+1}/{len(seqs)}] {seq_id}: FAILED", flush=True)
            traceback.print_exc()
            records.append({"seq_id": seq_id, "error": traceback.format_exc()[-400:]})

    ok = [r for r in records if "error" not in r]
    finite_ok = [r for r in ok if np.isfinite(r["ate_sim3"]["rmse"])
                 and r["ate_sim3"]["rmse"] < 1e6]
    summary = {
        "method": "mdvggt_infer",
        "n_seq": len(ok),
        "n_failed": len(records) - len(ok),
        "n_finite": len(finite_ok),
        "protocol": {"name": args.protocol, "max_frames": args.max_frames,
                     "window": args.window, "window_stride": args.window_stride,
                     "pair_window": args.pair_window, "pair_stride": args.pair_stride},
        "ate_sim3_rmse_mean": (float(np.mean([r["ate_sim3"]["rmse"] for r in finite_ok]))
                               if finite_ok else None),
        "ate_sim3_rmse_median": (float(np.median([r["ate_sim3"]["rmse"] for r in finite_ok]))
                                 if finite_ok else None),
        "rpe1_trans_mean": (float(np.mean([r["rpe_1"]["trans_mm_mean"] for r in finite_ok]))
                            if finite_ok else None),
        "total_time_sec": round(time.time() - t0, 1),
    }
    if finite_ok:
        summary["stratified"] = stratified_summary(finite_ok)
    n_pair = sum(1 for r in ok if (r.get("infer") or {}).get("mode") == "pairwise")
    summary["n_pairwise"] = n_pair
    summary["n_global"] = sum(1 for r in ok if (r.get("infer") or {}).get("mode") == "global")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=1)
    print(f"\n=== mdvggt-infer ({len(finite_ok)} finite / {len(ok)} ok / "
          f"{len(records)-len(ok)} fail) ===")
    print(f"  pairwise={summary.get('n_pairwise')} global={summary.get('n_global')}")
    for k in ("ate_sim3_rmse_mean", "ate_sim3_rmse_median", "rpe1_trans_mean"):
        v = summary[k]
        print(f"  {k}: {v:.3f}" if v is not None else f"  {k}: n/a")
    for b in summary.get("stratified", []):
        print(f"  [{b['bucket']}] n={b['n']}  ATE={b['ate_sim3_mean']:.3f}  "
              f"RPE1={b['rpe1_t_mean']:.3f}")
    print(f"结果: {out_dir}/summary.json  总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
