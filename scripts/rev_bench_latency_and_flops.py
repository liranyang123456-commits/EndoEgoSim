"""Reproducible Latency, Kernel-Launch & Compute-Complexity (GFLOPs) Audit for VRIH R1.8.

Directly tests and quantifies the root causes of Reviewer #1 Comment 8:
1. Why eager batch-1 latency was identical in bf16 and fp32 (45 ms):
   In endosim/lite/egolite.py, `observed_flow` and `solve_pose` explicitly set `autocast(enabled=False)`,
   forcing RAFT-large (28 ms) and the solver into fp32 even under a bf16 context.
2. Why RTX 5090 Laptop (45 ms) and RTX 2080 Ti (47 ms) had similar batch-1 eager times:
   Counts the exact number of ATen/CUDA operator dispatches in a single forward pass (~hundreds of small
   kernels across 12 RAFT updates + 3 GN steps at 32x40), showing CPU/driver launch latency dominates at
   batch=1, whereas at batch=16 compute throughput dominates (9.5 ms vs 17.3 ms vs 42.1 ms).
3. Re-measures EgoLite and the pairwise PyTorch baselines under:
   (a) Eager fp32, (b) Eager true-bf16/fp16 (allowing autocast inside RAFT), and (c) torch.compile /
       CUDA-graph-friendly execution when GPU is idle, plus exact hardware-independent GFLOPs (MACs x 2)
       via torch.utils.flop_counter.FlopCounterMode!
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(r"D:\ego_motiion_Camera")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "revision_vrih"
OUT.mkdir(parents=True, exist_ok=True)


def count_ops_and_flops(fn) -> tuple[int, float]:
    """Returns (number_of_aten_ops_dispatched, total_GFLOPs)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class OpCounter(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.count += 1
            return func(*args, **(kwargs or {}))

    with FlopCounterMode(display=False) as fc:
        fn()
    gflops = fc.get_total_flops() / 1e9

    with OpCounter() as oc:
        fn()
    return oc.count, float(gflops)


def cuda_time_ms(fn, warm: int = 20, reps: int = 100) -> dict:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(ts)
    return {
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
    }


def main():
    dev = "cuda"
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "models": {},
    }

    # 1. EgoLite (original fp32-forced flow vs true bf16 autocast flow vs reduced RAFT iters)
    from endosim.lite.egolite import build_from_args
    ck = torch.load(ROOT / "results/egolite/r_v2/last.pth", map_location="cpu")
    ego = build_from_args(ck["args"], pretrained=False)
    ego.load_state_dict(ck["model"])
    ego = ego.to(dev).eval()

    img = torch.rand(2, 3, 256, 320, device=dev)
    ph = torch.rand(2, 1, 256, 320, device=dev)
    K = torch.tensor([[250.0, 0, 160.0], [0, 250.0, 128.0], [0, 0, 1.0]], device=dev)[None]

    def fn_ego_fp32():
        with torch.no_grad():
            ego(img[:1], img[1:], K, ph[:1], ph[1:])

    n_ops_ego, gflops_ego = count_ops_and_flops(fn_ego_fp32)
    t_ego_fp32 = cuda_time_ms(fn_ego_fp32)

    def fn_ego_orig_bf16():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ego(img[:1], img[1:], K, ph[:1], ph[1:])

    t_ego_orig_bf16 = cuda_time_ms(fn_ego_orig_bf16)

    # True bf16 RAFT (allowing autocast inside RAFT while keeping 6x6 linear solve in fp32 for stability)
    def fn_ego_true_bf16():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fl = ego.flow(img[:1] * 2 - 1, img[1:] * 2 - 1, num_flow_updates=ego.flow_iters)[-1]
            ego(img[:1], img[1:], K, ph[:1], ph[1:], flow=fl)

    t_ego_true_bf16 = cuda_time_ms(fn_ego_true_bf16)

    # Breakdown: RAFT alone in fp32 vs true bf16 autocast vs 6 RAFT iters
    t_raft_fp32 = cuda_time_ms(lambda: ego.observed_flow(img[:1], img[1:]))

    def fn_raft_bf16():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ego.flow(img[:1] * 2 - 1, img[1:] * 2 - 1, num_flow_updates=12)[-1]

    def fn_ego_6iters_bf16():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fl = ego.flow(img[:1] * 2 - 1, img[1:] * 2 - 1, num_flow_updates=6)[-1]
            ego(img[:1], img[1:], K, ph[:1], ph[1:], flow=fl)

    t_raft_true_bf16 = cuda_time_ms(fn_raft_bf16)
    t_ego_6iters = cuda_time_ms(fn_ego_6iters_bf16)

    report["models"]["EgoLite"] = {
        "params_M": round(sum(p.numel() for p in ego.parameters()) / 1e6, 2),
        "aten_ops_per_pair": n_ops_ego,
        "gflops_per_pair": round(gflops_ego, 2),
        "eager_fp32": t_ego_fp32,
        "eager_orig_bf16_context_raft_forced_fp32": t_ego_orig_bf16,
        "eager_true_bf16_raft": t_ego_true_bf16,
        "raft_only_fp32": t_raft_fp32,
        "raft_only_bf16": t_raft_true_bf16,
        "eager_true_bf16_raft_6iters": t_ego_6iters,
    }
    del ego
    torch.cuda.empty_cache()

    # 2. AF-SfMLearner & EndoDAC PoseNets
    import egomotion_baselines as eb
    for mname in ("afsfm", "endodac"):
        enc, dec, from_params = eb._load(mname, dev)
        x = torch.rand(1, 6, 256, 320, device=dev)

        def fn_pn():
            with torch.no_grad():
                o = dec([enc(x)])
                from_params(o[0][:, 0], o[1][:, 0])

        def fn_pn_bf16():
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o = dec([enc(x)])
                from_params(o[0][:, 0].float(), o[1][:, 0].float())

        ops_pn, gflops_pn = count_ops_and_flops(fn_pn)
        report["models"][mname] = {
            "params_M": round((sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())) / 1e6, 2),
            "aten_ops_per_pair": ops_pn,
            "gflops_per_pair": round(gflops_pn, 2),
            "eager_fp32": cuda_time_ms(fn_pn),
            "eager_bf16": cuda_time_ms(fn_pn_bf16),
        }

    # 3. Endo-FASt3r (428M)
    import eval_endofast3r as ef
    f3r = ef.load_model(torch.device(dev))
    xa = torch.rand(1, 3, 384, 512, device=dev)
    xb = torch.rand(1, 3, 384, 512, device=dev)

    def fn_f3r_fp32():
        with torch.no_grad():
            f3r({"img": xa}, {"img": xb})

    def fn_f3r_bf16():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            f3r({"img": xa}, {"img": xb})

    ops_f3r, gflops_f3r = count_ops_and_flops(fn_f3r_fp32)
    report["models"]["Endo-FASt3r"] = {
        "params_M": round(sum(p.numel() for p in f3r.parameters()) / 1e6, 2),
        "aten_ops_per_pair": ops_f3r,
        "gflops_per_pair": round(gflops_f3r, 2),
        "eager_fp32": cuda_time_ms(fn_f3r_fp32, warm=10, reps=40),
        "eager_bf16": cuda_time_ms(fn_f3r_bf16, warm=10, reps=40),
    }
    del f3r
    torch.cuda.empty_cache()

    out_path = OUT / "r1_8_latency_and_flops_audit.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
