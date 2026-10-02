"""CUT3R zero-shot on the EndoEgoSim simtest92 protocol.

Requires the official repo and a published checkpoint:
  E:\\SOTA_Methods\\CUT3R
  E:\\SOTA_Methods\\CUT3R\\src\\cut3r_512_dpt_4_64.pth

  python scripts/eval_cut3r.py --list lists/simtest92.txt --max-frames 64 \\
      --out results/sota --tag cut3r_simtest
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

from endosim.eval.metrics import load_pose_txt
from endosim.eval.protocol import list_color_frames, select_frame_indices

sys.path.insert(0, os.path.dirname(__file__))
from eval_slam import (
    _finite_mean,
    _finite_median,
    eval_with_scale_correction,
    stratified_summary,
)

CUT3R_ROOT = os.environ.get("CUT3R_ROOT") or r"E:\SOTA_Methods\CUT3R"
CUT3R_CKPT = os.environ.get("CUT3R_CKPT") or os.path.join(
    CUT3R_ROOT, "src", "cut3r_512_dpt_4_64.pth")

_CUT3R = None


def _purge_dust3r():
    for k in list(sys.modules):
        if k == "dust3r" or k.startswith("dust3r.") or k in (
                "add_ckpt_path", "croco") or k.startswith("croco."):
            del sys.modules[k]


def _force_pytorch_rope():
    """Hide broken 5090 curope kernels so croco/models/pos_embed.py uses PyTorch RoPE.

    The CUDA extension imports, then device-asserts in rope_2d. An empty
    models.curope module makes `from models.curope import cuRoPE2D` fail and
    triggers the official slow fallback. Do not write CUT3R scores unless 92/92.
    """
    from types import ModuleType
    for k in list(sys.modules):
        if k == "curope" or k.startswith("curope.") or k == "models.curope" \
                or k.startswith("models.curope."):
            del sys.modules[k]
    fake = ModuleType("models.curope")
    sys.modules["models.curope"] = fake
    os.environ.setdefault("CUT3R_FORCE_PYTORCH_ROPE", "1")
    print("[cut3r] forcing PyTorch RoPE (curope hidden)", flush=True)


def _load_cut3r(device, ckpt, size):
    global _CUT3R
    if _CUT3R is not None:
        return _CUT3R
    if not os.path.isdir(CUT3R_ROOT):
        raise FileNotFoundError(f"CUT3R repo missing: {CUT3R_ROOT}")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"CUT3R checkpoint missing: {ckpt}. "
            "Download cut3r_512_dpt_4_64.pth from the official Google Drive.")
    src = os.path.join(CUT3R_ROOT, "src")
    croco = os.path.join(src, "croco")
    for p in (CUT3R_ROOT, src, croco):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    _purge_dust3r()
    _force_pytorch_rope()
    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(ckpt)
    _purge_dust3r()
    _force_pytorch_rope()
    from dust3r.model import ARCroco3DStereo
    model = ARCroco3DStereo.from_pretrained(ckpt).to(device)
    model.eval()
    _patch_rope_clamp(model)
    _CUT3R = (model, size)
    return _CUT3R


def _patch_rope_clamp(model):
    """CUT3R position ids can be negative / OOB; F.embedding then device-asserts."""
    import torch
    patched = 0
    for m in model.modules():
        if m.__class__.__name__ != "RoPE2D" or not hasattr(m, "apply_rope1d"):
            continue
        if getattr(m, "_endo_clamped", False):
            continue

        def apply_rope1d(tokens, pos1d, cos, sin):
            n = int(cos.shape[0])
            pos = pos1d.long().nan_to_num(0).clamp(0, max(n - 1, 0))
            c = torch.nn.functional.embedding(pos, cos)[:, None, :, :]
            s = torch.nn.functional.embedding(pos, sin)[:, None, :, :]
            x1, x2 = tokens[..., : tokens.shape[-1] // 2], tokens[..., tokens.shape[-1] // 2 :]
            rot = torch.cat((-x2, x1), dim=-1)
            return (tokens * c) + (rot * s)

        m.apply_rope1d = apply_rope1d
        m._endo_clamped = True
        patched += 1
    print(f"[cut3r] clamped RoPE2D.apply_rope1d on {patched} modules", flush=True)


def _views_from_paths(frame_paths, size):
    from dust3r.utils.image import load_images
    import torch
    images = load_images(frame_paths, size=size, verbose=False)
    views = []
    for i, im in enumerate(images):
        views.append({
            "img": im["img"],
            "ray_map": torch.full(
                (im["img"].shape[0], 6, im["img"].shape[-2], im["img"].shape[-1]),
                torch.nan),
            "true_shape": torch.from_numpy(im["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        })
    return views


def run_cut3r(frame_paths, ckpt=None, size=512):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = _load_cut3r(device, ckpt or CUT3R_CKPT, size)
    from dust3r.inference import inference
    from dust3r.utils.camera import pose_encoding_to_camera
    views = _views_from_paths(frame_paths, size)
    outputs, _ = inference(views, model, device, verbose=False)
    poses = []
    for pred in outputs["pred"]:
        T = pose_encoding_to_camera(pred["camera_pose"].clone()).cpu().numpy()
        if T.ndim == 3:
            T = T[0]
        if T.shape == (3, 4):
            out = np.eye(4, dtype=np.float64)
            out[:3, :4] = T
            T = out
        poses.append(T.astype(np.float64))
    return np.stack(poses, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="lists/simtest92.txt")
    ap.add_argument("--out", default="results/sota")
    ap.add_argument("--tag", default="cut3r_simtest")
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--ckpt", default=CUT3R_CKPT)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    def _resolve_seq(s):
        if os.path.isdir(s):
            return s
        return s

    seqs = [ln.strip() for ln in open(args.list, encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    seqs = [p for s in seqs if os.path.isdir(p := _resolve_seq(s))]
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cut3r] {len(seqs)} sequences ckpt={args.ckpt}", flush=True)

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
                raise FileNotFoundError(f"no frames for {sid}")
            est = run_cut3r(frame_paths, ckpt=args.ckpt, size=args.size)
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
            np.savetxt(est_path, est.reshape(len(est), 16), fmt="%.6f")
            print(f"[{i+1}/{len(seqs)}] {sid}: ATE(Sim3)={res['ate_sim3']['rmse']:.3f}mm",
                  flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(seqs)}] {sid}: FAILED", flush=True)
            traceback.print_exc()
            records.append({"seq_id": sid, "error": traceback.format_exc()[-400:]})
            if i == 0:
                break

    ok = [r for r in records if "error" not in r]
    summary = {
        "method": "cut3r", "n_seq": len(ok), "n_failed": len(records) - len(ok),
        "protocol": {"max_frames": args.max_frames, "name": "uniform",
                     "size": args.size},
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
    print(f"=== cut3r ATE(Sim3)={summary['ate_sim3_rmse_mean']} n={summary['n_seq']} ===")


if __name__ == "__main__":
    main()
