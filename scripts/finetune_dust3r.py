"""DUSt3R 在 EndoEgoSim 训练集上的配对微调（M7b: 与 Endo3R 同范式的域适配）。

监督: 配对帧 (i,j) 的度量点图（世界系 mm）+ 位姿 + 有效掩码。
损失: dust3r 官方 ConfLoss(Regr3D(L21Loss)) —— norm_mode=None 直接度量监督,
让模型学习内窥镜工作距离的绝对尺度（zero-shot 时尺度偏差 29-42x）。

用法:
  python scripts/finetune_dust3r.py --iters 1500 --lr 1e-5 --out results/finetune/dust3r_endo
  # 评测: python scripts/baseline_sota.py --method dust3r --ckpt <ckpt> ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MAST3R_ROOT = r"D:\mast3r"
DUST3R_CKPT = r"D:\dust3r_checkpoints\DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
import argparse as _argparse  # noqa: F401
import torch

torch.serialization.add_safe_globals([_argparse.Namespace])
# dust3r losses.py 使用 print(..., force=True)(其仓库自定义print); 兼容之
import builtins as _builtins
_orig_print = print
def _print_compat(*args, force=False, **kw):
    _orig_print(*args, **kw)
_builtins.print = _print_compat
sys.path.insert(0, MAST3R_ROOT)
import mast3r.utils.path_to_dust3r  # noqa: F401  内嵌 dust3r 子仓库路径


def load_pair(seq_dir, i, j, size=512):
    """采样一对帧 -> (view1, view2) 含模型输入 + GT(pts3d世界系/camera_pose/valid_mask)。"""
    from dust3r.utils.image import load_images

    intr = json.load(open(os.path.join(seq_dir, "intrinsics.json")))
    poses = np.loadtxt(os.path.join(seq_dir, "pose_c2w.txt")).reshape(-1, 4, 4)
    paths = [os.path.join(seq_dir, "color", f"{k:06d}.png") for k in (i, j)]
    views = load_images(paths, size=size, verbose=False)

    for v, k in zip(views, (i, j)):
        v["true_shape"] = torch.as_tensor(v["true_shape"])
        img = v["img"]                       # (1,3,h,w), 已归一化 [-1,1]
        _, _, h, w = img.shape
        sx, sy = w / intr["width"], h / intr["height"]
        K = np.array([[intr["fx"] * sx, 0, intr["cx"] * sx],
                      [0, intr["fy"] * sy, intr["cy"] * sy], [0, 0, 1.0]])
        dep = cv2.imread(os.path.join(seq_dir, "depth", f"{k:06d}.png"),
                         cv2.IMREAD_UNCHANGED).astype(np.float32)
        dep = cv2.resize(dep, (w, h), interpolation=cv2.INTER_LINEAR)
        u, vv = np.meshgrid(np.arange(w), np.arange(h))
        pts_cam = np.stack([(u - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1],
                            np.ones_like(u, dtype=np.float32)], -1) * dep[..., None]
        T_wc = poses[k]
        pts_w = pts_cam @ T_wc[:3, :3].T + T_wc[:3, 3]
        v["pts3d"] = torch.as_tensor(pts_w, dtype=torch.float32)[None]      # (1,h,w,3)
        v["camera_pose"] = torch.as_tensor(T_wc, dtype=torch.float32)[None]  # (1,4,4) c2w
        v["valid_mask"] = torch.as_tensor(dep > 0)[None]                     # (1,h,w)
    return views[0], views[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-gap", type=int, default=8, help="配对帧距上限(常规)")
    ap.add_argument("--large-gap", type=int, default=24,
                    help="大基线配对帧距上限 (覆盖 SCARED 关键帧 hop)")
    ap.add_argument("--large-gap-frac", type=float, default=0.3,
                    help="抽大基线配对的比例")
    ap.add_argument("--hard-frac", type=float, default=0.35,
                    help="低参照序列过采样比例")
    ap.add_argument("--out", default="results/finetune/dust3r_endo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from dust3r.losses import ConfLoss, L21Loss, Regr3D
    from dust3r.model import AsymmetricCroCo3DStereo

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    model = AsymmetricCroCo3DStereo.from_pretrained(DUST3R_CKPT).to(device).train()
    # 度量监督(norm_mode=None): 直接以 mm 尺度回归, 修复域间尺度偏差
    criterion = ConfLoss(Regr3D(L21Loss(), norm_mode=None), alpha=1.0).to(device)

    seq_dirs = sorted(glob.glob("sim_data/train/seq_*"))
    hard_dirs, easy_dirs = [], []
    for d in seq_dirs:
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        refs = [r for r in meta.get("reference_fraction", []) if r is not None]
        mean_r = float(np.mean(refs)) if refs else 0.6
        (hard_dirs if mean_r < 0.45 else easy_dirs).append(d)
    if not hard_dirs:
        hard_dirs = list(seq_dirs)
    if not easy_dirs:
        easy_dirs = list(seq_dirs)
    print(f"训练序列池: {len(seq_dirs)} (hard={len(hard_dirs)} easy={len(easy_dirs)})")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters)

    def sample_pair():
        pool = hard_dirs if rng.random() < args.hard_frac else easy_dirs
        d = pool[int(rng.integers(0, len(pool)))]
        n = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))["n_frames"]
        gap_hi = (args.large_gap if rng.random() < args.large_gap_frac
                  else args.max_gap)
        gap_hi = max(1, min(int(gap_hi), n - 1))
        i = int(rng.integers(0, n - 1))
        j = min(i + int(rng.integers(1, gap_hi + 1)), n - 1)
        return load_pair(d, i, j)

    t0 = time.time()
    scaler = torch.amp.GradScaler("cuda")
    for it in range(1, args.iters + 1):
        v1, v2 = sample_pair()
        v1 = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in v1.items()}
        v2 = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in v2.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred1, pred2 = model(v1, v2)
            loss, _ = criterion(v1, v2, pred1, pred2)
        if not isinstance(loss, torch.Tensor):
            continue          # 无有效像素的配对(官方损失返回 int 0), 跳过
        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()
        sched.step()
        if it % 20 == 0 or it == 1:
            print(f"[{it}/{args.iters}] loss={float(loss):.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
        if it % 500 == 0:
            p = os.path.join(args.out, f"ckpt_{it}.pth")
            torch.save({"model": model.state_dict()}, p)

    ckpt = os.path.join(args.out, "dust3r_endo_ft.pth")
    torch.save({"model": model.state_dict(), "iters": args.iters,
                "metric": True}, ckpt)
    print(f"已保存: {ckpt}")


if __name__ == "__main__":
    main()
