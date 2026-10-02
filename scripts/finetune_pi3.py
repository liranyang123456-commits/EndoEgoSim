"""π³ 在 EndoEgoSim 上的运动分解微调 (MD-Pi3)。

冻结 DINOv2 encoder, 只训 camera_decoder + camera_head。
监督: 窗口首帧归一的 c2w (与 eval_slam.run_pi3 同一约定)。
深度/点图不监督, 避免器械像素把相对运动解释成场景运动。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PI3_ROOT = r"D:\Pi3"
W_IN, H_IN = 518, 392  # 14 的倍数, 接近 VGGT 窗


class SimPoseWindows:
    def __init__(self, root="sim_data/train", seq_len=8, hard_frac=0.3,
                 fail_frac=0.2, href_frac=0.25):
        import importlib.util
        vggt_ft = os.path.join(os.path.dirname(__file__), "finetune_vggt.py")
        spec = importlib.util.spec_from_file_location("finetune_vggt", vggt_ft)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        SimWindowDataset = mod.SimWindowDataset
        self.inner = SimWindowDataset(root=root, seq_len=seq_len,
                                      sample_mode="mixed", hard_frac=hard_frac,
                                      fail_frac=fail_frac, href_frac=href_frac,
                                      stride_range=(2, 8))
        self.seq_len = seq_len

    def sample(self):
        s = self.inner.sample()
        imgs = []
        for rgb in s["images"]:
            im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
            imgs.append(im)
        w0, h0 = imgs[0].size
        limit = 180000
        scale = math.sqrt(limit / float(w0 * h0)) if w0 * h0 else 1.0
        wt, ht = w0 * scale, h0 * scale
        k, m = max(1, round(wt / 14)), max(1, round(ht / 14))
        while (k * 14) * (m * 14) > limit:
            if k / m > wt / ht:
                k -= 1
            else:
                m -= 1
        tw, th = max(1, k) * 14, max(1, m) * 14
        to_t = transforms.ToTensor()
        x = torch.stack([to_t(im.resize((tw, th), Image.Resampling.LANCZOS))
                         for im in imgs])
        w2c = torch.as_tensor(s["w2c"], dtype=torch.float32)
        c2w = torch.linalg.inv(w2c)
        return x, c2w


def se3_pose_loss(pred, gt):
    t = (pred[:, :3, 3] - gt[:, :3, 3]).abs().mean()
    r = pred[:, :3, :3] @ gt[:, :3, :3].transpose(-1, -2)
    tr = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    ang = torch.acos(((tr - 1) * 0.5).clamp(-1 + 1e-5, 1 - 1e-5))
    rel_p = pred[1:] @ torch.linalg.inv(pred[:-1])
    rel_g = gt[1:] @ torch.linalg.inv(gt[:-1])
    rt = (rel_p[:, :3, 3] - rel_g[:, :3, 3]).abs().mean()
    return t + ang.mean() + 2.0 * rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", default="results/finetune/pi3_endo_v1")
    ap.add_argument("--save-every", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if PI3_ROOT not in sys.path:
        sys.path.insert(0, PI3_ROOT)
    from pi3.models.pi3 import Pi3
    device = "cuda"
    model = Pi3()
    ckpt = os.path.join(r"D:\Pi3_checkpoints", "model.safetensors")
    if os.path.isfile(ckpt):
        from safetensors.torch import load_file
        model.load_state_dict(load_file(ckpt))
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3")
    model = model.to(device).train()
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(("camera_decoder", "camera_head")))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MD-Pi3 可训练 {n_train/1e6:.1f}M", flush=True)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.05)
    ds = SimPoseWindows(seq_len=args.seq_len)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        x, c2w = ds.sample()
        x = x.to(device)[None]
        c2w = c2w.to(device)
        pred = model(x.float())["camera_poses"][0]
        loss = se3_pose_loss(pred.float(), c2w.float())
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optim.step()
        if it % 5 == 0 or it == 1:
            print(f"[{it}/{args.iters}] loss={float(loss):.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
        if args.save_every and it % args.save_every == 0:
            torch.save({"model": model.state_dict(), "iters": it},
                       os.path.join(args.out, f"ckpt_{it}.pth"))
    out = os.path.join(args.out, "pi3_endo_ft.pth")
    torch.save({"model": model.state_dict(), "iters": args.iters,
                "recipe": "md_pi3_camhead"}, out)
    print(f"已保存 {out}", flush=True)


if __name__ == "__main__":
    main()
