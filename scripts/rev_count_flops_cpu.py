"""Pure-CPU exact GFLOPs (MACs x 2) and ATen operator-dispatch count for VRIH R1.8.

Runs on CPU (`device='cpu'`, 0 MB GPU memory) so it is 100% deterministic and unaffected by GPU load.
Measures EgoLite (full, RAFT-only, trunk+GN-only), AF-SfMLearner PoseNet, EndoDAC PoseNet,
Reloc3r-512, and Endo-FASt3r (428M).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

torch.set_num_threads(4)
ROOT = Path(r"D:\ego_motiion_Camera")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


class OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += 1
        return func(*args, **(kwargs or {}))


def measure(fn) -> dict:
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            fn()
        flops = fc.get_total_flops()
        with OpCounter() as oc:
            fn()
    return {"gflops": round(flops / 1e9, 2), "gmacs": round(flops / 2e9, 2), "aten_ops": int(oc.count)}


def main():
    out = {}

    # 1. EgoLite
    from endosim.lite.egolite import build_from_args
    ck = torch.load(ROOT / "results/egolite/r_v2/last.pth", map_location="cpu")
    ego = build_from_args(ck["args"], pretrained=False).eval()
    ego.load_state_dict(ck["model"])
    img = torch.rand(2, 3, 256, 320)
    ph = torch.rand(2, 1, 256, 320)
    K = torch.tensor([[250.0, 0, 160.0], [0, 250.0, 128.0], [0, 0, 1.0]])[None]

    m_full = measure(lambda: ego(img[:1], img[1:], K, ph[:1], ph[1:]))
    fl = ego.observed_flow(img[:1], img[1:])
    m_raft = measure(lambda: ego.observed_flow(img[:1], img[1:]))
    m_rest = measure(lambda: ego(img[:1], img[1:], K, ph[:1], ph[1:], flow=fl))
    out["EgoLite"] = {
        "params_M": round(sum(p.numel() for p in ego.parameters()) / 1e6, 2),
        "input_res": "256x320",
        "full": m_full,
        "raft_12iters": m_raft,
        "encoder_head_gn3": m_rest,
    }
    del ego

    # 2. AF-SfMLearner & EndoDAC PoseNets
    import egomotion_baselines as eb
    x_pn = torch.rand(1, 6, 256, 320)
    for name in ("afsfm", "endodac"):
        enc, dec, from_params = eb._load(name, "cpu")
        m_pn = measure(lambda: (lambda o: from_params(o[0][:, 0], o[1][:, 0]))(dec([enc(x_pn)])))
        out[name] = {
            "params_M": round((sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())) / 1e6, 2),
            "input_res": "256x320",
            "full": m_pn,
        }

    # 3. Endo-FASt3r (428M) & Reloc3r-512 (427M)
    import eval_endofast3r as ef
    f3r = ef.load_model(torch.device("cpu"))
    xa = torch.rand(1, 3, 384, 512)
    xb = torch.rand(1, 3, 384, 512)
    m_f3r = measure(lambda: f3r({"img": xa}, {"img": xb}))
    out["Endo-FASt3r"] = {
        "params_M": round(sum(p.numel() for p in f3r.parameters()) / 1e6, 2),
        "input_res": "384x512",
        "full": m_f3r,
    }
    del f3r

    dest = ROOT / "results/revision_vrih/r1_8_flops_and_ops_cpu.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
