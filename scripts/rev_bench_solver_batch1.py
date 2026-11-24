"""Exact CUDA-event-synchronized batch-1 solver & occlusion-mask timings on the local GPU.

Times the exact code paths used in Table R3 / Table S3 (stride-8 GN, stride-4 GN,
full-resolution GN, out-of-bounds mask, and reverse optical flow + forward-backward
consistency check) at batch size 1 in 32-bit eager mode, matching Section 4 / Table R8.
Updates results/revision_vrih/r1_3_design_choices_lossless.json in place.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from endosim.lite.data import H_IN, W_IN  # noqa: E402
from endosim.lite.egolite import build_from_args  # noqa: E402

STRIDE = 8


def bench_cuda_events(fn, warmup: int = 20, iters: int = 100) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        times.append(float(ev0.elapsed_time(ev1)))
    arr = np.array(times, dtype=np.float64)
    return {
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "min_ms": round(float(np.min(arr)), 3),
    }


@torch.no_grad()
def main() -> None:
    ck = torch.load(ROOT / "results/egolite/r_v2/last.pth", map_location="cpu", weights_only=False)
    model = build_from_args(ck["args"], pretrained=False).cuda().eval()
    model.load_state_dict(ck["model"])

    torch.manual_seed(0)
    i0 = torch.rand(1, 3, H_IN, W_IN, device="cuda", dtype=torch.float32)
    i1 = torch.rand(1, 3, H_IN, W_IN, device="cuda", dtype=torch.float32)
    p0 = torch.rand(1, 1, H_IN, W_IN, device="cuda", dtype=torch.float32)
    p1 = torch.rand(1, 1, H_IN, W_IN, device="cuda", dtype=torch.float32)
    Kb = torch.tensor([[[240.0, 0.0, W_IN / 2.0], [0.0, 240.0, H_IN / 2.0], [0.0, 0.0, 1.0]]],
                      device="cuda", dtype=torch.float32)

    fl_full = model.observed_flow(i0, i1)
    o = model(i0, i1, Kb, p0, p1, flow=fl_full)
    Z, u, v, f_obs = o["Z"].float(), o["u"].float(), o["v"].float(), o["f_obs"].float()
    R0, t0 = o["R0"].float(), o["t0"].float()
    prob = torch.softmax(o["logits"].float(), 1)
    share = torch.sigmoid(model.deform_logit.float())
    inv_var = torch.exp(-o["logvar"].float())
    w_full = (prob[:, 0] + share * prob[:, 1]) * inv_var

    up, vp = u[None] + f_obs[:, 0], v[None] + f_obs[:, 1]
    in_b = ((up >= 0) & (up <= W_IN - 1) & (vp >= 0) & (vp <= H_IN - 1)).float()

    f_s4 = F.avg_pool2d(fl_full, 4, 4).float()
    Z_s4 = F.interpolate(Z[:, None], size=(64, 80), mode="bilinear", align_corners=False)[:, 0]
    w_s4 = F.interpolate(w_full[:, None], size=(64, 80), mode="bilinear", align_corners=False)[:, 0]
    v4, u4 = torch.meshgrid(torch.arange(64, device="cuda", dtype=torch.float32) * 4 + 1.5,
                            torch.arange(80, device="cuda", dtype=torch.float32) * 4 + 1.5, indexing="ij")

    Z_s1 = F.interpolate(Z[:, None], size=(H_IN, W_IN), mode="bilinear", align_corners=False)[:, 0]
    w_s1 = F.interpolate(w_full[:, None], size=(H_IN, W_IN), mode="bilinear", align_corners=False)[:, 0]
    v1, u1 = torch.meshgrid(torch.arange(H_IN, device="cuda", dtype=torch.float32),
                            torch.arange(W_IN, device="cuda", dtype=torch.float32), indexing="ij")

    def run_s8():
        model.solve_pose(Z, u, v, Kb, f_obs, w_full, R0, t0)

    def run_oob():
        up_, vp_ = u[None] + f_obs[:, 0], v[None] + f_obs[:, 1]
        ib = ((up_ >= 0) & (up_ <= W_IN - 1) & (vp_ >= 0) & (vp_ <= H_IN - 1)).float()
        model.solve_pose(Z, u, v, Kb, f_obs, w_full * ib, R0, t0)

    def run_s4():
        model.solve_pose(Z_s4, u4, v4, Kb, f_s4, w_s4, R0, t0)

    def run_s1():
        model.solve_pose(Z_s1, u1, v1, Kb, fl_full.float(), w_s1, R0, t0)

    def run_fb_extra():
        fl_bwd = model.observed_flow(i1, i0)
        f_bwd_s8 = F.avg_pool2d(fl_bwd, STRIDE, STRIDE).float()
        f_bwd_w = model._warp(f_bwd_s8, u, v, f_obs, H_IN, W_IN)
        _ = (((f_obs + f_bwd_w).norm(dim=1) < 1.5) & (in_b > 0.5)).float()

    res = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "batch_size": 1,
        "precision": "fp32_eager",
        "stride8_gn": bench_cuda_events(run_s8),
        "stride8_gn_with_oob_mask": bench_cuda_events(run_oob),
        "stride4_gn": bench_cuda_events(run_s4),
        "fullres_gn": bench_cuda_events(run_s1),
        "fb_occ_extra_reverse_flow": bench_cuda_events(run_fb_extra, warmup=10, iters=60),
    }
    print(json.dumps(res, indent=2))

    p = ROOT / "results/revision_vrih/r1_3_design_choices_lossless.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["batch1_cuda_event_timing_fp32_eager"] = res
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print("Updated", p)


if __name__ == "__main__":
    main()
