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


def _first_dir(*cands):
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return cands[-1]


def _first_file(*cands):
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return cands[-1]


RELOC3R_ROOT = os.environ.get("RELOC3R_ROOT") or _first_dir(
    r"E:\SOTA_Methods\Reloc3r", r"D:\reloc3r_src", r"D:\Reloc3r_src")
DROID_ROOT = os.environ.get("DROID_ROOT") or _first_dir(
    "/root/DROID-SLAM", r"D:\DROID-SLAM")
DROID_CKPT = os.environ.get("DROID_CKPT") or _first_file(
    os.path.join(DROID_ROOT, "droid.pth"),
    "/root/DROID-SLAM/droid.pth", r"D:\DROID-SLAM\droid.pth")
ORB_ROOT = os.environ.get("ORB_ROOT") or _first_dir(
    "/root/ORB_SLAM3", r"D:\ORB_SLAM3")


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
_PI3_CKPT_OVERRIDE = None


def _load_pi3(device, ckpt_path=None):
    global _PI3_MODEL, _PI3_CKPT_OVERRIDE
    if ckpt_path:
        _PI3_CKPT_OVERRIDE = ckpt_path
    if _PI3_MODEL is not None:
        return _PI3_MODEL
    if PI3_ROOT not in sys.path:
        sys.path.insert(0, PI3_ROOT)
    import torch
    from pi3.models.pi3 import Pi3
    ckpt = None
    if _PI3_CKPT_OVERRIDE and os.path.isfile(_PI3_CKPT_OVERRIDE):
        ckpt = _PI3_CKPT_OVERRIDE
    else:
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


_RELOC3R_MODEL = None


def _load_reloc3r(device):
    global _RELOC3R_MODEL
    if _RELOC3R_MODEL is not None:
        return _RELOC3R_MODEL
    if not os.path.isdir(RELOC3R_ROOT):
        raise FileNotFoundError(
            f"Reloc3r code not found at {RELOC3R_ROOT}. "
            "Clone https://github.com/ffrivera0/reloc3r")
    if RELOC3R_ROOT not in sys.path:
        sys.path.insert(0, RELOC3R_ROOT)
    from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model
    _RELOC3R_MODEL = setup_reloc3r_relpose_model(model_args="512", device=device)
    return _RELOC3R_MODEL


def run_reloc3r(frame_paths):
    """Chain consecutive Reloc3r-512 relative poses. Keep raw translation
    (do not unit-normalize per pair) so Sim(3) can recover a global scale.
    pose2to1 maps camera-2 points into camera-1: c2w_{i+1} = c2w_i @ T_{1<-2}.
    """
    import torch
    if RELOC3R_ROOT not in sys.path:
        sys.path.insert(0, RELOC3R_ROOT)
    from reloc3r.utils.image import load_images, check_images_shape_format
    from reloc3r.reloc3r_relpose import inference_relpose
    from reloc3r.utils.device import to_numpy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_reloc3r(device)
    n = len(frame_paths)
    traj = np.repeat(np.eye(4)[None], n, 0)
    for i in range(n - 1):
        images = load_images([frame_paths[i], frame_paths[i + 1]], size=512)
        images = check_images_shape_format(images, device)
        pose2to1 = to_numpy(inference_relpose([images[0], images[1]], model, device)[0])
        T = np.asarray(pose2to1, dtype=np.float64)
        if T.shape == (3, 4):
            T4 = np.eye(4, dtype=np.float64)
            T4[:3, :4] = T
            T = T4
        traj[i + 1] = traj[i] @ T
    return traj


def _droid_stream(frame_paths, K):
    """Official DROID image stream: BGR CHW tensor, size ~384x512, multiple of 8."""
    import torch
    stream = []
    h0 = w0 = None
    for t, p in enumerate(frame_paths):
        image = cv2.imread(p)
        if image is None:
            raise FileNotFoundError(p)
        if h0 is None:
            h0, w0 = image.shape[:2]
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
            h1, w1 = h1 - h1 % 8, w1 - w1 % 8
            sx, sy = w1 / float(w0), h1 / float(h0)
            fx = float(K[0, 0]) * sx
            fy = float(K[1, 1]) * sy
            cx = float(K[0, 2]) * sx
            cy = float(K[1, 2]) * sy
        image = cv2.resize(image, (w1, h1))
        image = torch.as_tensor(image).permute(2, 0, 1)
        intr = torch.as_tensor([fx, fy, cx, cy], dtype=torch.float32)
        stream.append((t, image[None], intr))
    return stream, [h1, w1]


def run_droid(frame_paths, K):
    if not os.path.isfile(DROID_CKPT):
        raise FileNotFoundError(f"DROID weights missing: {DROID_CKPT}")
    import torch
    if DROID_ROOT not in sys.path:
        sys.path.insert(0, DROID_ROOT)
    slam_dir = os.path.join(DROID_ROOT, "droid_slam")
    if slam_dir not in sys.path:
        sys.path.insert(0, slam_dir)
    from droid import Droid
    stream, image_size = _droid_stream(frame_paths, K)
    args = argparse.Namespace(
        weights=DROID_CKPT, image_size=image_size, buffer=512,
        stereo=False, disable_vis=True, upsample=False,
        beta=0.3, filter_thresh=2.4, warmup=8, keyframe_thresh=4.0,
        frontend_thresh=16.0, frontend_window=25, frontend_radius=2,
        frontend_nms=1, backend_thresh=22.0, backend_radius=2, backend_nms=3,
    )
    torch.multiprocessing.set_start_method("spawn", force=True)
    droid = Droid(args)
    for t, image, intrinsics in stream:
        droid.track(t, image, intrinsics=intrinsics)
    traj = droid.terminate(stream)
    poses = np.repeat(np.eye(4)[None], len(frame_paths), 0)
    T = np.asarray(traj)
    if T.ndim == 2 and T.shape[1] >= 7:
        from scipy.spatial.transform import Rotation
        n = min(len(T), len(poses))
        for i in range(n):
            poses[i, :3, 3] = T[i, :3]
            qw, qx, qy, qz = T[i, 3:7]
            poses[i, :3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    elif T.ndim == 3:
        poses[:len(T)] = T
    return poses


def _write_orb_yaml(path, K, width, height, fps=30.0):
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    path.write_text(
        "%YAML:1.0\n"
        'File.version: "1.0"\n'
        'Camera.type: "PinHole"\n'
        f"Camera1.fx: {fx:.6f}\n"
        f"Camera1.fy: {fy:.6f}\n"
        f"Camera1.cx: {cx:.6f}\n"
        f"Camera1.cy: {cy:.6f}\n"
        "Camera1.k1: 0.0\nCamera1.k2: 0.0\nCamera1.p1: 0.0\n"
        "Camera1.p2: 0.0\nCamera1.k3: 0.0\n"
        f"Camera.width: {int(width)}\n"
        f"Camera.height: {int(height)}\n"
        f"Camera.fps: {int(round(fps))}\n"
        "Camera.RGB: 1\n"
        "ORBextractor.nFeatures: 1000\n"
        "ORBextractor.scaleFactor: 1.2\n"
        "ORBextractor.nLevels: 8\n"
        "ORBextractor.iniThFAST: 20\n"
        "ORBextractor.minThFAST: 7\n"
        "Viewer.KeyFrameSize: 0.05\n"
        "Viewer.KeyFrameLineWidth: 1.0\n"
        "Viewer.GraphLineWidth: 0.9\n"
        "Viewer.PointSize: 2.0\n"
        "Viewer.CameraSize: 0.08\n"
        "Viewer.CameraLineWidth: 3.0\n"
        "Viewer.ViewpointX: 0.0\n"
        "Viewer.ViewpointY: -0.7\n"
        "Viewer.ViewpointZ: -1.8\n"
        "Viewer.ViewpointF: 500.0\n",
        encoding="utf-8",
    )


def _parse_tum_traj(path):
    from scipy.spatial.transform import Rotation
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) < 8:
                continue
            ts, tx, ty, tz, qx, qy, qz, qw = vals[:8]
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = (tx, ty, tz)
            rows.append((ts, T))
    return rows


def run_orbslam3(frame_paths, K, seq_dir):
    """Official monocular ORB-SLAM3. Returns (poses[M], gt_index[M]).

    Missing frames are dropped (TUM associate), not interpolated. Too few
    tracked poses raises, so the sequence is counted failed — not a fake ATE.
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    exe = os.path.join(ORB_ROOT, "Examples", "Monocular", "mono_tum")
    if os.name == "nt":
        exe_win = exe + ".exe"
        if os.path.isfile(exe_win):
            exe = exe_win
    voc = os.path.join(ORB_ROOT, "Vocabulary", "ORBvoc.txt")
    if not os.path.isfile(exe):
        raise FileNotFoundError(f"ORB-SLAM3 binary missing: {exe}")
    if not os.path.isfile(voc):
        raise FileNotFoundError(f"ORBvoc.txt missing: {voc}")

    im0 = cv2.imread(frame_paths[0])
    if im0 is None:
        raise FileNotFoundError(frame_paths[0])
    h, w = im0.shape[:2]
    tmp = tempfile.mkdtemp(prefix="orb3_")
    try:
        rgb_dir = os.path.join(tmp, "rgb")
        os.makedirs(rgb_dir)
        rgb_txt = os.path.join(tmp, "rgb.txt")
        with open(rgb_txt, "w", encoding="utf-8") as f:
            f.write("# timestamp filename\n")
            for i, src in enumerate(frame_paths):
                ext = os.path.splitext(src)[1] or ".png"
                dst = os.path.join(rgb_dir, f"{i:06d}{ext}")
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copy2(src, dst)
                f.write(f"{i:.6f} rgb/{i:06d}{ext}\n")
        yaml_path = os.path.join(tmp, "camera.yaml")
        _write_orb_yaml(Path(yaml_path), K, w, h)
        env = os.environ.copy()
        log_p = os.path.join(tmp, "orb.log")
        persist = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "results", "sota", "orbslam3_last.log")
        os.makedirs(os.path.dirname(persist), exist_ok=True)
        cmd = [exe, voc, yaml_path, tmp]
        xvfb = shutil.which("xvfb-run")
        if xvfb:
            cmd = [xvfb, "-a", "-s", "-screen 0 640x480x24"] + cmd
        elif not env.get("DISPLAY"):
            env["DISPLAY"] = ":0"
        with open(log_p, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd, cwd=tmp, env=env, timeout=600,
                stdout=log, stderr=subprocess.STDOUT,
            )
        shutil.copy2(log_p, persist)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ORB-SLAM3 exit {proc.returncode}; see {persist}")
        cand = [
            os.path.join(tmp, "CameraTrajectory.txt"),
            os.path.join(tmp, "KeyFrameTrajectory.txt"),
            os.path.join(os.getcwd(), "CameraTrajectory.txt"),
            os.path.join(os.getcwd(), "KeyFrameTrajectory.txt"),
        ]
        traj = []
        used = None
        for p in cand:
            if os.path.isfile(p):
                rows = _parse_tum_traj(p)
                if len(rows) > len(traj):
                    traj, used = rows, p
        if len(traj) < 5:
            raise RuntimeError(
                f"ORB-SLAM3 produced {len(traj)} poses ({used}); tracking lost")
        n = len(frame_paths)
        sel, poses, seen = [], [], set()
        for ts, T in traj:
            i = int(round(ts))
            if 0 <= i < n and i not in seen:
                seen.add(i)
                sel.append(i)
                poses.append(T)
        if len(poses) < 5:
            raise RuntimeError(
                f"ORB-SLAM3 associated {len(poses)}/{n} frames from {used}")
        return np.stack(poses), np.asarray(sel, dtype=np.int64)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
    ap.add_argument("--method", required=True,
                    choices=["pi3", "droid", "orbslam3", "reloc3r"])
    ap.add_argument("--list", default="lists/simtest92.txt")
    ap.add_argument("--out", default="results/sota")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--ckpt", default=None,
                    help="Optional π³ checkpoint (MD-Pi3 .pth or official .safetensors)")
    args = ap.parse_args()
    tag = args.tag or f"{args.method}_simtest"
    if args.method == "pi3" and args.ckpt:
        import torch
        _load_pi3("cuda" if torch.cuda.is_available() else "cpu", args.ckpt)
    def _resolve_seq(s):
        if os.path.isdir(s):
            return s
        if len(s) >= 3 and s[1] == ":":
            wsl = f"/mnt/{s[0].lower()}{s[2:].replace(chr(92), '/')}"
            if os.path.isdir(wsl):
                return wsl
        return s

    seqs = [ln.strip() for ln in open(args.list, encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    seqs = [p for s in seqs if os.path.isdir(p := _resolve_seq(s))]
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
            gt_all = load_pose_txt(os.path.join(seq_dir, "pose_c2w.txt"))
            idx = select_frame_indices(gt_all, "uniform", max_frames=args.max_frames)
            frame_paths = list_color_frames(seq_dir, indices=idx)
            n = min(len(frame_paths), len(idx))
            frame_paths, idx = frame_paths[:n], idx[:n]
            gt = gt_all[idx]
            if not frame_paths:
                raise FileNotFoundError(f"no frames materialized for {sid}")
            im0 = cv2.imread(frame_paths[0])
            if im0 is None:
                raise FileNotFoundError(frame_paths[0])
            K = load_K(seq_dir, im0)
            if args.method == "pi3":
                est = run_pi3(frame_paths)
            elif args.method == "reloc3r":
                est = run_reloc3r(frame_paths)
            elif args.method == "droid":
                est = run_droid(frame_paths, K)
            else:
                est, sel = run_orbslam3(frame_paths, K, seq_dir)
                gt = gt[sel]
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
        except Exception as e:
            print(f"[{i+1}/{len(seqs)}] {sid}: FAILED", flush=True)
            traceback.print_exc()
            records.append({"seq_id": sid, "error": traceback.format_exc()[-400:]})
            if i == 0 and isinstance(e, (FileNotFoundError, ImportError)):
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
