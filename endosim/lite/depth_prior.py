"""Frozen monocular depth prior (MiDaS DPT-SwinV2-Tiny-256) for EgoLite.

Returns per-frame relative disparity normalised to [0, 1] (2nd-98th percentile), used in
place of the shading-based pseudo-height maps.
"""
from __future__ import annotations

import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

MIDAS_HUB = os.environ.get("MIDAS_HUB", os.path.join(torch.hub.get_dir(), "intel-isl_MiDaS_master"))
MIDAS_CKPT = os.environ.get("MIDAS_CKPT", os.path.join(torch.hub.get_dir(), "checkpoints", "dpt_swin2_tiny_256.pt"))


def load_midas_swin2t():
    m = torch.hub.load(MIDAS_HUB, "DPT_SwinV2_T_256", source="local", pretrained=False)
    sd = torch.load(MIDAS_CKPT, map_location="cpu")
    out = {}
    for k, v in sd.items():
        if k.endswith("attn_mask") or k.startswith("pretrained.model.head."):
            continue
        # newer timm puts each stage's patch merging at the start of the next stage
        mm = re.match(r"(pretrained\.model\.layers\.)(\d+)(\.downsample\..*)", k)
        out[f"{mm.group(1)}{int(mm.group(2)) + 1}{mm.group(3)}" if mm else k] = v
    missing, unexpected = m.load_state_dict(out, strict=False)
    missing = [k for k in missing if "head" not in k]
    if missing or unexpected:
        raise RuntimeError(f"MiDaS weights mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    for i in range(1, 5):                     # newer timm Swin blocks emit (B, H, W, C)
        name = f"act_postprocess{i}"
        setattr(m.pretrained, name, _NHWCPost(getattr(m.pretrained, name)))
    return m


class _NHWCPost(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.orig = orig

    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous() if x.dim() == 4 else self.orig(x)


class DepthPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = load_midas_swin2t()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.eval()
        return self

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img (N,3,H,W) in [0,1] -> (N,1,H,W) relative disparity in [0,1]."""
        H, W = img.shape[-2:]
        x = F.interpolate(img.float(), size=(256, 256), mode="bilinear", align_corners=False)
        with torch.autocast(device_type=img.device.type, dtype=torch.bfloat16):
            d = self.net((x - self.mean) / self.std).float()
        d = F.interpolate(d[:, None], size=(H, W), mode="bilinear", align_corners=False)
        flat = d.flatten(1)
        lo = flat.quantile(0.02, dim=1)[:, None, None, None]
        hi = flat.quantile(0.98, dim=1)[:, None, None, None]
        return ((d - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1)
