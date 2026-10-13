"""Per-pair latency of the reported EgoLite checkpoint at batch size 1 (streaming) and 16.

Times the complete forward pass (optical flow, pixel head, pair head, GN solver) with CUDA events,
fp32 and bfloat16, and records peak GPU memory.  Output: results/egolite/latency_r_v2.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from endosim.lite.egolite import build_from_args  # noqa: E402

ckpt = sys.argv[1] if len(sys.argv) > 1 else "results/egolite/r_v2/last.pth"
out_path = sys.argv[2] if len(sys.argv) > 2 else "results/egolite/latency_r_v2.json"
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
model = build_from_args(ck["args"])
model.load_state_dict(ck["model"])
model = model.cuda().eval()
n_params = sum(p.numel() for p in model.parameters())
H, W = 256, 320
K1 = torch.tensor([[250.0, 0, W / 2], [0, 250.0, H / 2], [0, 0, 1]], device="cuda")


def run(bs, dtype, warm, reps):
    img = torch.rand(bs + 1, 3, H, W, device="cuda")
    ph = torch.rand(bs + 1, 1, H, W, device="cuda")
    K = K1[None].expand(bs, 3, 3)
    torch.cuda.reset_peak_memory_stats()
    ts = []
    with torch.no_grad():
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if dtype == "bf16" else torch.autocast("cuda", enabled=False)
        with ctx:
            for _ in range(warm):
                model(img[:-1], img[1:], K, ph[:-1], ph[1:])
            torch.cuda.synchronize()
            for _ in range(reps):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                model(img[:-1], img[1:], K, ph[:-1], ph[1:])
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) / bs)
    ts = np.asarray(ts)
    return {"batch": bs, "dtype": dtype, "median_ms_per_pair": float(np.median(ts)),
            "p95_ms_per_pair": float(np.percentile(ts, 95)), "min_ms_per_pair": float(ts.min()),
            "peak_mem_MB": float(torch.cuda.max_memory_allocated() / 2 ** 20), "reps": reps}


def run_split(bs, dtype, warm, reps):
    """RAFT alone vs. everything else (flow supplied), per pair."""
    img = torch.rand(bs + 1, 3, H, W, device="cuda")
    ph = torch.rand(bs + 1, 1, H, W, device="cuda")
    K = K1[None].expand(bs, 3, 3)
    t_raft, t_rest = [], []
    with torch.no_grad():
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if dtype == "bf16" else torch.autocast("cuda", enabled=False)
        with ctx:
            flow = model.observed_flow(img[:-1], img[1:])
            for i in range(warm + reps):
                e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
                e0.record()
                model.observed_flow(img[:-1], img[1:])
                e1.record()
                model(img[:-1], img[1:], K, ph[:-1], ph[1:], flow=flow)
                e2.record()
                torch.cuda.synchronize()
                if i >= warm:
                    t_raft.append(e0.elapsed_time(e1) / bs)
                    t_rest.append(e1.elapsed_time(e2) / bs)
    return {"batch": bs, "dtype": dtype, "raft_median_ms": float(np.median(t_raft)),
            "rest_median_ms": float(np.median(t_rest))}


rows = []
# bf16 is only timed on GPUs with native support (Ampere or newer); older cards report fp32 only
dtypes = ("bf16", "fp32") if torch.cuda.get_device_capability(0)[0] >= 8 else ("fp32",)
for bs, reps in ((1, 500), (16, 40)):
    for dtype in dtypes:
        try:
            r = run(bs, dtype, warm=30 if bs == 1 else 5, reps=reps)
            r.update(run_split(bs, dtype, warm=20 if bs == 1 else 3, reps=100 if bs == 1 else 15))
        except Exception as e:  # noqa: BLE001
            r = {"batch": bs, "dtype": dtype, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        rows.append(r)
        print(r, flush=True)
res = {"checkpoint": ckpt, "params_M": n_params / 1e6, "input": [H, W],
       "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": rows,
       "date": time.strftime("%Y-%m-%d %H:%M")}
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1)
print("params_M", round(n_params / 1e6, 2), res["gpu"], torch.__version__)
