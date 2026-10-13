"""Evaluate an EgoLite checkpoint under the uniform at-most-64-frame protocol.

  python scripts/eval_egolite.py --list lists/simtest92.txt \
      --ckpt results/egolite/v1/best.pth --tag egolite_v1_simtest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from baseline_sota import eval_with_scale_correction, stratified_summary  # noqa: E402
from endosim.lite.egolite import build_from_args  # noqa: E402
from endosim.lite.infer import infer_sequence, infer_sequence_ba, infer_sequence_kf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/sota")
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skips", default="1",
                    help="comma-separated frame gaps; gaps above 1 add pose-graph edges")
    ap.add_argument("--rho-mode", default="pixel", choices=["pixel", "median"])
    ap.add_argument("--ba-window", type=int, default=0)
    ap.add_argument("--ba-gaps", default="1,2")
    ap.add_argument("--ba-iters", type=int, default=4)
    ap.add_argument("--kf-tau", type=float, default=0.0, help="keyframe-relative tracking threshold (px)")
    ap.add_argument("--ablate", default=None,
                    choices=["nogn", "uniform", "nolayer", "novar", "staticonly"],
                    help="inference-time component ablation")
    ap.add_argument("--gn-iters", type=int, default=None,
                    help="override the number of Gauss-Newton steps at inference (trained with 3)")
    args = ap.parse_args()
    skips = tuple(int(s) for s in args.skips.split(","))

    seqs = [ln.strip() for ln in open(args.list, encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    seqs = [s for s in seqs if os.path.isfile(os.path.join(s, "pose_c2w.txt"))]
    if args.limit:
        seqs = seqs[: args.limit]
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = dict(ck.get("args", {}))
    if args.gn_iters is not None:
        cfg["gn_iters"] = args.gn_iters
    model = build_from_args(cfg)
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()
    model.ablate = args.ablate
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[egolite-eval] n={len(seqs)} ckpt={args.ckpt} params={n_par / 1e6:.2f}M", flush=True)

    records = []
    net_time, net_pairs = 0.0, 0
    t_all = time.time()
    for i, d in enumerate(seqs):
        seq_id = os.path.basename(d.rstrip("/\\"))
        try:
            t0 = time.time()
            if args.kf_tau > 0:
                est, gt, idx, sec, n_pairs = infer_sequence_kf(model, d, max_frames=args.max_frames, tau=args.kf_tau)
            elif args.ba_window > 0:
                est, gt, idx, sec, n_pairs = infer_sequence_ba(model, d, max_frames=args.max_frames,
                    window=args.ba_window, stride=max(1,args.ba_window//2),
                    gaps=tuple(int(x) for x in args.ba_gaps.split(',')), iters=args.ba_iters)
            else:
                est, gt, idx, sec, n_pairs = infer_sequence(model, d, max_frames=args.max_frames,
                                                             skips=skips, rho_mode=args.rho_mode)
            if i > 0:
                net_time += sec
                net_pairs += n_pairs
            res = eval_with_scale_correction(est, gt)
            seq_elapsed = time.time() - t0
            meta_p = os.path.join(d, "meta.json")
            meta_p = os.path.join(d, "meta.json")
            with open(os.path.join(out_dir, 'progress.log'), 'a', encoding='utf-8') as pf:
                pf.write(f"{i+1}/{len(seqs)} {seq_id} {seq_elapsed:.1f}s {res['ate_sim3']['rmse']:.4f} mm\\n")

            if os.path.exists(meta_p):
                meta = json.load(open(meta_p, encoding="utf-8"))
                refs = [r for r in meta.get("reference_fraction", []) or [] if r is not None]
                res["reference_fraction"] = float(np.mean(refs)) if refs else None
                res["motion_type"] = meta.get("motion_type")
                res["scene_kind"] = meta.get("scene_kind")
            res["seq_id"] = seq_id
            res["seq_dir"] = d
            res["n_frames_used"] = int(len(est))
            res["time_sec"] = round(time.time() - t0, 2)
            records.append(res)
            np.savetxt(os.path.join(out_dir, f"{seq_id}_est_c2w.txt"),
                       est.reshape(len(est), 16), fmt="%.6f")
            print(f"[{i + 1}/{len(seqs)}] {seq_id}: ATE(Sim3)={res['ate_sim3']['rmse']:.3f}mm "
                  f"RPE1={res['rpe_1']['trans_mm_mean']:.3f} ({res['time_sec']:.1f}s)", flush=True)
        except Exception:
            print(f"[{i + 1}/{len(seqs)}] {seq_id}: FAILED", flush=True)
            traceback.print_exc()
            records.append({"seq_id": seq_id, "error": traceback.format_exc()[-400:]})

    ok = [r for r in records if "error" not in r]
    ates = np.array([r["ate_sim3"]["rmse"] for r in ok])
    n_frames = sum(r["n_frames_used"] for r in ok[1:])
    summary = {
        "net_ms_per_frame": 1000.0 * net_time / max(1, n_frames),
        "method": "egolite",
        "ckpt": args.ckpt,
        "params_M": n_par / 1e6,
        "n_seq": len(records),
        "n_failed": len(records) - len(ok),
        "protocol": {"name": "uniform", "max_frames": args.max_frames, "skips": list(skips),
                     "rho_mode": args.rho_mode},
        "ate_sim3_rmse_mean": float(ates.mean()) if len(ates) else None,
        "ate_sim3_rmse_median": float(np.median(ates)) if len(ates) else None,
        "rpe1_trans_mean": float(np.mean([r["rpe_1"]["trans_mm_mean"] for r in ok])) if ok else None,
        "net_ms_per_pair": 1000.0 * net_time / max(1, net_pairs),
        "stratified": stratified_summary(ok),
        "total_time_sec": round(time.time() - t_all, 1),
    }
    json.dump({"summary": summary, "records": records},
              open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"), indent=1)
    print(f"\n=== egolite ({len(ok)} ok / {len(records)}) ===")
    for k in ("ate_sim3_rmse_mean", "ate_sim3_rmse_median", "rpe1_trans_mean", "net_ms_per_pair"):
        if summary[k] is None:
            print(f"  {k}: not available (all sequences failed)")
        else:
            print(f"  {k}: {summary[k]:.3f}")
    for b in summary["stratified"]:
        print(f"  [{b['bucket']}] n={b['n']} ATE={b['ate_sim3_mean']:.3f}")


if __name__ == "__main__":
    main()
