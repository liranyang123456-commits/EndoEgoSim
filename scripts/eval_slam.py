"""π³ / DROID-SLAM / ORB-SLAM3 on the same EndoEgoSim protocol.

  python scripts/eval_slam.py --method pi3 --list lists/simtest92.txt \\
      --max-frames 64 --out results/sota --tag pi3_simtest

Missing install/weights fail with a clear message (no fake numbers).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.metrics import evaluate_trajectory, load_pose_txt, rpe
from endosim.eval.protocol import list_color_frames, motion_stats_of_indices, select_frame_indices

PI3_ROOT = r"D:\Pi3"
PI3_CKPT = r"D:\Pi3_checkpoints"
DROID_ROOT = r"D:\DROID-SLAM"
DROID_CKPT = r"D:\DROID-SLAM\droid.pth"
ORB_ROOT = r"D:\ORB_SLAM3"


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


def load_K(seq_dir, img):
    p = os.path.join(seq_dir, "intrinsics.json")
    if not os.path.exists(p):
        h, w = img.shape[:2]
        f = 0.9 * max(h, w)
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]], np.float64)
    intr = json.load(open(p, encoding="utf-8"))
    h, w = img.shape[:2]
    sx, sy = w / float(intr["width"]), h / float(intr["height"])
    return np.array([[intr["fx"] * sx, 0, intr["cx"] * sx],
                     [0, intr["fy"] * sy, intr["cy"] * sy],
                     [0, 0, 1.0]], np.float64)


_PI3_MODEL = None


def _load_pi3(device):
    global _PI3_MODEL
    if _PI3_MODEL is not None:
        return _PI3_MODEL
    if PI3_ROOT not in sys.path:
        sys.path.insert(0, PI3_ROOT)
    import torch
    from pi3.models.pi3 import Pi3
    ckpt = None
    for cand in (os.path.join(PI3_CKPT, "model.safetensors"),
                 os.path.join(PI3_ROOT, "ckpts", "model.safetensors")):
        if os.path.isfile(cand):
            ckpt = cand
            break
    if ckpt is None:
        model = Pi3.from_pretrained("yyfz233/Pi3")
    else:
        model = Pi3()
        if ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file
            model.load_state_dict(load_file(ckpt))
        else:
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) else sd)
    model = model.to(device).eval()
    _PI3_MODEL = model
    return model


def run_pi3(frame_paths):
    import math
    import tempfile
    import torch
    from PIL import Image
    from torchvision import transforms
    if PI3_ROOT not in sys.path:
        sys.path.insert(0, PI3_ROOT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_pi3(device)
    imgs = []
    for p in frame_paths:
        imgs.append(Image.open(p).convert("RGB"))
    w0, h0 = imgs[0].size
    limit = 255000
    scale = math.sqrt(limit / float(w0 * h0)) if w0 * h0 > 0 else 1.0
    wt, ht = w0 * scale, h0 * scale
    k, m = round(wt / 14), round(ht / 14)
    while (k * 14) * (m * 14) > limit:
        if k / m > wt / ht:
            k -= 1
        else:
            m -= 1
    tw, th = max(1, k) * 14, max(1, m) * 14
    to_t = transforms.ToTensor()
    x = torch.stack([to_t(im.resize((tw, th), Image.Resampling.LANCZOS)) for im in imgs])
    x = x.to(device)[None]
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with torch.no_grad():
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            pred = model(x.float())
    poses = pred["camera_poses"][0].detach().float().cpu().numpy()
    if poses.shape[-2:] == (3, 4):
        out = np.repeat(np.eye(4)[None], len(poses), 0)
        out[:, :3, :4] = poses
        poses = out
    return poses.astype(np.float64)


def run_droid(frame_paths, K):
    if not os.path.isfile(DROID_CKPT):
        raise FileNotFoundError(f"DROID weights missing: {DROID_CKPT}")
    sys.path.insert(0, DROID_ROOT)
    from droid_slam.droid import Droid
    args = argparse.Namespace(
        weights=DROID_CKPT, image_size=[240, 320], buffer=512,
        stereo=False, disable_vis=True, upsample=False,
        beta=0.3, filter_thresh=2.4, warmup=8, keyframe_thresh=4.0,
        frontend_thresh=16.0, frontend_window=25, frontend_radius=2,
        frontend_nms=1, backend_thresh=22.0, backend_radius=2, backend_nms=3,
    )
    droid = Droid(args)
    for t, p in enumerate(frame_paths):
        im = cv2.imread(p)
        im = cv2.resize(im, (args.image_size[1], args.image_size[0]))
        droid.track(t, im, intrinsics=np.array(
            [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], np.float32))
    traj = droid.terminate(frame_paths)
    poses = np.repeat(np.eye(4)[None], len(frame_paths), 0)
    # DROID returns (t,q) or 4x4 depending on version
    T = np.asarray(traj)
    if T.ndim == 2 and T.shape[1] == 7:
        from scipy.spatial.transform import Rotation
        for i, row in enumerate(T):
            poses[i, :3, 3] = row[:3]
            poses[i, :3, :3] = Rotation.from_quat(row[3:]).as_matrix()
    elif T.ndim == 3:
        poses[:len(T)] = T
    return poses


def run_orbslam3(frame_paths, K, seq_dir):
    exe = os.path.join(ORB_ROOT, "Examples", "Monocular", "mono_euroc.exe")
    if not os.path.isfile(exe):
        exe = os.path.join(ORB_ROOT, "Examples", "Monocular", "mono_tum")
    if not os.path.isfile(exe):
        raise FileNotFoundError(f"ORB-SLAM3 binary missing under {ORB_ROOT}")
    raise RuntimeError("ORB-SLAM3 binary found but EndoEgoSim runner not wired "
                       "until Vocabulary/ORBvoc.txt and a working mono exe exist")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pi3", "droid", "orbslam3"])
    ap.add_argument("--list", default="lists/simtest92.txt")
    ap.add_argument("--out", default="results/sota")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    tag = args.tag or f"{args.method}_simtest"
    seqs = [ln.strip() for ln in open(args.list, encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    seqs = [s for s in seqs if os.path.isdir(s)]
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{args.method}] {len(seqs)} sequences", flush=True)

    records, t0 = [], time.time()
    for i, seq_dir in enumerate(seqs):
        sid = os.path.basename(seq_dir.rstrip("/\\"))
        est_path = os.path.join(out_dir, f"{sid}_est_c2w.txt")
        if args.skip_existing and os.path.isfile(est_path):
            print(f"[{i+1}/{len(seqs)}] {sid}: skip existing", flush=True)
            continue
        t_seq = time.time()
        try:
            frames = list_color_frames(seq_dir)
            gt_all = load_pose_txt(os.path.join(seq_dir, "pose_c2w.txt"))
            n = min(len(frames), len(gt_all))
            frames, gt_all = frames[:n], gt_all[:n]
            idx = select_frame_indices(gt_all, "uniform", max_frames=args.max_frames)
            frame_paths = [frames[k] for k in idx]
            gt = gt_all[idx]
            im0 = cv2.imread(frame_paths[0])
            K = load_K(seq_dir, im0)
            if args.method == "pi3":
                est = run_pi3(frame_paths)
            elif args.method == "droid":
                est = run_droid(frame_paths, K)
            else:
                est = run_orbslam3(frame_paths, K, seq_dir)
            if len(est) != len(gt):
                n2 = min(len(est), len(gt))
                est, gt = est[:n2], gt[:n2]
            res = eval_with_scale_correction(est, gt)
            meta_p = os.path.join(seq_dir, "meta.json")
            if os.path.exists(meta_p):
                meta = json.load(open(meta_p, encoding="utf-8"))
                refs = [r for r in meta.get("reference_fraction", []) if r is not None]
                res["reference_fraction"] = float(np.mean(refs)) if refs else None
            res["seq_id"] = sid
            res["n_frames_used"] = int(len(est))
            res["time_sec"] = round(time.time() - t_seq, 2)
            records.append(res)
            np.savetxt(os.path.join(out_dir, f"{sid}_est_c2w.txt"),
                       est.reshape(len(est), 16), fmt="%.6f")
            print(f"[{i+1}/{len(seqs)}] {sid}: ATE(Sim3)={res['ate_sim3']['rmse']:.3f}mm",
                  flush=True)
        except Exception:
            print(f"[{i+1}/{len(seqs)}] {sid}: FAILED", flush=True)
            traceback.print_exc()
            records.append({"seq_id": sid, "error": traceback.format_exc()[-400:]})
            if i == 0:
                break

    ok = [r for r in records if "error" not in r]
    summary = {
        "method": args.method, "n_seq": len(ok), "n_failed": len(records) - len(ok),
        "protocol": {"max_frames": args.max_frames, "name": "uniform"},
        "ate_se3_rmse_mean": _finite_mean([r["ate_se3"]["rmse"] for r in ok]) if ok else None,
        "ate_sim3_rmse_mean": _finite_mean([r["ate_sim3"]["rmse"] for r in ok]) if ok else None,
        "ate_sim3_rmse_median": _finite_median([r["ate_sim3"]["rmse"] for r in ok]) if ok else None,
        "rpe1_trans_mean": _finite_mean([r["rpe_1"]["trans_mm_mean"] for r in ok]) if ok else None,
        "total_time_sec": round(time.time() - t0, 1),
    }
    if ok:
        summary["stratified"] = stratified_summary(ok)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=1)
    print(f"=== {args.method} ATE(Sim3)={summary['ate_sim3_rmse_mean']} n={summary['n_seq']} ===")


if __name__ == "__main__":
    main()
