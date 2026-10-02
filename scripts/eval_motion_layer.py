"""运动层分解头评测：位姿精度 + 运动层分割质量（mIoU）。

对每条序列前向 ml_head，输出 静态/变形/器械 三类概率图，与仿真 GT
motion_mask（1/2/3 -> 0/1/2）算 per-class IoU 与 mIoU；同时评位姿 ATE。
器械类（类 2）IoU 可与外部 DeepLabV3 器械分割对照。

用法：
  python scripts/eval_motion_layer.py --list lists/simtest92.txt \
      --ckpt results/finetune/ours_vggt_v7_ml/vggt_endo_ft.pth \
      --out results/sota/ours_v7ml_simtest_mlq.json --max-frames 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.metrics import evaluate_trajectory, load_pose_txt
from endosim.eval.protocol import list_color_frames, select_frame_indices


def iou_per_class(pred: np.ndarray, gt: np.ndarray, n_classes: int = 3) -> dict:
    """pred/gt: (H,W) int，类别 0..n_classes-1，gt 中 -1 忽略。"""
    out = {}
    valid = gt >= 0
    for c in range(n_classes):
        p = valid & (pred == c)
        g = valid & (gt == c)
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        out[c] = float(inter / union) if union > 0 else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    sys.path.insert(0, r"D:\vggt-main\vggt-main")
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    sys.path.insert(0, os.path.dirname(__file__))
    from baseline_sota import load_model

    seqs = [ln.strip() for ln in open(args.list, encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    seqs = [s for s in seqs if os.path.isdir(s)]
    if args.limit > 0:
        seqs = seqs[:args.limit]

    model = load_model("vggt", args.device, ckpt=args.ckpt, ml_head=True)
    print(f"[ml-eval] n={len(seqs)} ckpt={args.ckpt}")

    records = []
    for si, seq_dir in enumerate(seqs):
        seq_id = os.path.basename(seq_dir.rstrip("/\\"))
        try:
            gt_all = load_pose_txt(os.path.join(seq_dir, "pose_c2w.txt"))
            idx = select_frame_indices(gt_all, protocol="uniform",
                                       max_frames=args.max_frames)
            frames = list_color_frames(seq_dir, indices=idx)
            n = min(len(frames), len(idx))
            frames, idx = frames[:n], idx[:n]

            images = load_and_preprocess_images(frames).to(args.device)
            if images.dim() == 4:
                images = images[None]  # (S,3,H,W) -> (1,S,3,H,W)
            dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                tokens, psi = model.aggregator(images)
                pose_enc_list = model.camera_head(tokens)
                ml_pred, _ = model.ml_head(tokens, images=images,
                                           patch_start_idx=psi)
            ext, _ = pose_encoding_to_extri_intri(pose_enc_list[-1],
                                                  images.shape[-2:])
            ext = ext.reshape(-1, 3, 4).float().cpu().numpy()
            est = np.zeros((len(ext), 4, 4))
            for i, e in enumerate(ext):
                T = np.eye(4)
                T[:3, :] = e
                est[i] = np.linalg.inv(T)
            gt = gt_all[idx]
            res = evaluate_trajectory(est, gt)

            # 运动层分割
            ml_logits = ml_pred.float().cpu().numpy()  # (1,S,H,W,3) 或 (1,S,3,H,W)
            if ml_logits.ndim == 5 and ml_logits.shape[-1] == 3:
                ml_probs = ml_logits[0]                       # (S,H,W,3)
            else:
                ml_probs = ml_logits[0].transpose(0, 2, 3, 1)  # (S,3,H,W)->(S,H,W,3)
            pred_cls = ml_probs.argmax(axis=-1)               # (S,H,W)

            ious = []
            for ki, k in enumerate(idx):
                mp = os.path.join(seq_dir, "motion_mask", f"{int(k):06d}.png")
                if not os.path.isfile(mp):
                    continue
                mm = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
                if mm is None:
                    continue
                mm = cv2.resize(mm, (pred_cls.shape[2], pred_cls.shape[1]),
                                interpolation=cv2.INTER_NEAREST)
                gt_cls = np.where(np.isin(mm, [1, 2, 3]), mm - 1, -1)
                ious.append(iou_per_class(pred_cls[ki], gt_cls))
            if ious:
                per_cls = {c: float(np.nanmean([x[c] for x in ious])) for c in range(3)}
                miou = float(np.nanmean([per_cls[c] for c in range(3)]))
            else:
                per_cls = {c: float("nan") for c in range(3)}
                miou = float("nan")

            records.append({
                "seq_id": seq_id,
                "ate_sim3_mm": res["ate_sim3"]["rmse"],
                "miou": miou,
                "iou_static": per_cls[0],
                "iou_deform": per_cls[1],
                "iou_instrument": per_cls[2],
            })
        except Exception as e:
            records.append({"seq_id": seq_id, "error": str(e)[:300]})
        if (si + 1) % 10 == 0 or si + 1 == len(seqs):
            print(f"[ml-eval] {si + 1}/{len(seqs)}", flush=True)

    ok = [r for r in records if "error" not in r]
    summary = {
        "n": len(ok),
        "ate_sim3_mean": float(np.mean([r["ate_sim3_mm"] for r in ok])) if ok else None,
        "miou_mean": float(np.nanmean([r["miou"] for r in ok])) if ok else None,
        "iou_static": float(np.nanmean([r["iou_static"] for r in ok])) if ok else None,
        "iou_deform": float(np.nanmean([r["iou_deform"] for r in ok])) if ok else None,
        "iou_instrument": float(np.nanmean([r["iou_instrument"] for r in ok])) if ok else None,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
