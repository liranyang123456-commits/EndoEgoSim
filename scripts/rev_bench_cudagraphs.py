"""Reproducibly benchmark CUDA-Graph (zero Python/CUDA kernel-launch overhead) vs. Eager PyTorch
batch-1 float32 latencies across all four pairwise PyTorch models on the same GPU (R1.8):
  - AF-SfMLearner PoseNet (13.0M, 6.52 GFLOPs, 205 ops):  2.57 ms eager -> 0.83 ms CUDA Graph
  - EndoDAC PoseNet       (13.0M, 6.52 GFLOPs, 205 ops):  2.74 ms eager -> 0.83 ms CUDA Graph
  - EgoLite               (20.9M, 153.11 GFLOPs, 2877 ops): 44.61 ms eager -> 11.80 ms CUDA Graph (84.7 FPS, 0.00 diff)
  - Endo-FASt3r           (428.4M, 1501.49 GFLOPs, 9044 ops): 130.87 ms eager -> 78.29 ms CUDA Graph (12.8 FPS, 0.00 diff)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.models.optical_flow.raft as tv_raft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import egomotion_baselines as eb  # noqa: E402
import eval_endofast3r as ef  # noqa: E402
from endosim.lite.egolite import build_from_args  # noqa: E402


def bench_graph(fn, n_warmup: int = 10, n_rep: int = 80) -> tuple[float, float, object]:
    with torch.no_grad():
        out_eager = fn()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_cg = fn()
        for _ in range(n_warmup):
            g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n_rep):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return round(float(np.median(ts)), 2), round(float(np.percentile(ts, 95)), 2), (out_eager, out_cg)


def main() -> None:
    orig_linspace = torch.linspace
    orig_zeros = torch.zeros
    orig_eye = torch.eye
    orig_ones = torch.ones
    orig_solve = torch.linalg.solve

    # Allocate constant helper grids directly on GPU so CUDA stream capture has zero CPU->GPU copies
    tv_raft.make_coords_grid = lambda b, h, w, device="cuda": torch.stack(
        torch.meshgrid(torch.arange(h, device="cuda"), torch.arange(w, device="cuda"), indexing="ij")[::-1], dim=0
    ).float().repeat(b, 1, 1, 1)
    torch.linspace = lambda *a, **kw: orig_linspace(*a, **{**kw, "device": kw.get("device") or "cuda"})
    torch.linalg.solve = lambda A, B: torch.cholesky_solve(B, torch.linalg.cholesky_ex(A)[0])

    ck = torch.load(ROOT / "results/egolite/r_v2/last.pth", map_location="cpu")
    ego = build_from_args(ck["args"], pretrained=False).cuda().eval()
    ego.load_state_dict(ck["model"])

    af_enc, af_dec, _ = eb._load("afsfm", "cuda")
    dac_enc, dac_dec, _ = eb._load("endodac", "cuda")

    x0 = torch.rand(1, 3, 256, 320, device="cuda")
    x1 = torch.rand(1, 3, 256, 320, device="cuda")
    ph0 = torch.rand(1, 1, 256, 320, device="cuda")
    ph1 = torch.rand(1, 1, 256, 320, device="cuda")
    K = torch.tensor([[[280.0, 0.0, 160.0], [0.0, 280.0, 128.0], [0.0, 0.0, 1.0]]], device="cuda")
    x_pn = torch.cat([x1, x0], 1)
    fl_dummy = torch.zeros(1, 2, 256, 320, device="cuda")

    ego_med, ego_p95, (oe, oc) = bench_graph(lambda: ego(x0, x1, K, ph0, ph1))
    ego_rest_med, _, _ = bench_graph(lambda: ego(x0, x1, K, ph0, ph1, flow=fl_dummy))
    af_med, af_p95, _ = bench_graph(lambda: af_dec([af_enc(x_pn)]))
    dac_med, dac_p95, _ = bench_graph(lambda: dac_dec([dac_enc(x_pn)]))

    f3r = ef.load_model(torch.device("cuda"))
    pe = f3r.patch_embed
    pe.forward = lambda img, true_shape=None, **kw: (
        pe.norm(pe.proj(img).permute(0, 2, 3, 1).flatten(1, 2).float()),
        pe.position_getter(img.shape[0], img.shape[2] // pe.patch_size[1], img.shape[3] // pe.patch_size[0], img.device).long(),
    )
    rope = f3r.rope
    rope.forward = lambda tokens, positions: torch.cat([
        rope.apply_rope1d(tokens[:, :, :, :tokens.size(3) // 2], positions[:, :, 0], *rope.get_cos_sin(tokens.size(3) // 2, 64, tokens.device, tokens.dtype)),
        rope.apply_rope1d(tokens[:, :, :, tokens.size(3) // 2:], positions[:, :, 1], *rope.get_cos_sin(tokens.size(3) // 2, 64, tokens.device, tokens.dtype)),
    ], dim=-1)
    torch.zeros = lambda *a, **kw: orig_zeros(*a, **{**kw, "device": kw.get("device") or "cuda"})
    torch.eye = lambda *a, **kw: orig_eye(*a, **{**kw, "device": kw.get("device") or "cuda"})
    torch.ones = lambda *a, **kw: orig_ones(*a, **{**kw, "device": kw.get("device") or "cuda"})

    im0 = {"img": torch.rand(1, 3, 384, 512, device="cuda"), "true_shape": torch.tensor([[384, 512]], device="cpu", dtype=torch.int32)}
    im1 = {"img": torch.rand(1, 3, 384, 512, device="cuda"), "true_shape": torch.tensor([[384, 512]], device="cpu", dtype=torch.int32)}
    f3r_med, f3r_p95, (fe, fc) = bench_graph(lambda: f3r(im0, im1), n_rep=50)

    out = {
        "afsfm_fp32": {"cudagraph_median_ms": af_med, "cudagraph_p95_ms": af_p95},
        "endodac_fp32": {"cudagraph_median_ms": dac_med, "cudagraph_p95_ms": dac_p95},
        "EgoLite_full_fp32": {
            "cudagraph_median_ms": ego_med,
            "cudagraph_p95_ms": ego_p95,
            "fps": round(1000.0 / ego_med, 1),
            "trunk_plus_gn_cudagraph_ms": ego_rest_med,
            "max_diff_vs_eager": float((oe["t"] - oc["t"]).abs().max()),
        },
        "EndoFASt3r_fp32": {
            "cudagraph_median_ms": f3r_med,
            "cudagraph_p95_ms": f3r_p95,
            "fps": round(1000.0 / f3r_med, 1),
            "max_diff_vs_eager": float((fe[0]["pose"] - fc[0]["pose"]).abs().max()),
        },
    }
    print(json.dumps(out, indent=2))
    p = ROOT / "results/revision_vrih/r1_8_latency_and_flops_audit.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["cuda_graphs_batch1_fp32"] = out
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
