"""Smoke tests for the EgoLite release: the model builds from the released config and runs one frame pair on CPU.

The trained weights are a release asset (egolite_r_v2.pth), so the forward test uses random weights; the
checkpoint test runs only if the file is present (set EGOLITE_CKPT or put it at results/egolite/r_v2/last.pth).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from endosim.lite.egolite import build_from_args  # noqa: E402

CFG = ROOT / "configs" / "egolite_r_v2_args.json"


def _pair(h=256, w=320):
    g = torch.Generator().manual_seed(0)
    img0 = torch.rand(1, 3, h, w, generator=g)
    img1 = torch.roll(img0, shifts=(2, 3), dims=(2, 3))
    K = torch.tensor([[[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1.0]]])
    ph = torch.rand(1, 1, h, w, generator=g)
    return img0, img1, K, ph, torch.roll(ph, shifts=(2, 3), dims=(2, 3))


def _check(out):
    R, t = out["R"], out["t"]
    assert R.shape == (1, 3, 3) and t.shape == (1, 3)
    assert torch.isfinite(R).all() and torch.isfinite(t).all()
    eye = R[0] @ R[0].T
    assert torch.allclose(eye, torch.eye(3), atol=1e-3), "R is not orthonormal"
    assert abs(torch.det(R[0]).item() - 1.0) < 1e-3


def test_config_matches_reported_model():
    cfg = json.load(open(CFG, encoding="utf-8"))
    assert cfg["flow_net"] == "large" and cfg["gn_iters"] == 3 and cfg["head_extra"] == 6
    assert abs(cfg["params_M"] - 20.87) < 0.01


def test_forward_cpu_random_weights():
    cfg = json.load(open(CFG, encoding="utf-8"))
    model = build_from_args(cfg, pretrained=False).eval()
    n = sum(p.numel() for p in model.parameters())
    assert abs(n / 1e6 - cfg["params_M"]) < 0.01
    with torch.no_grad():
        out = model(*_pair())
    _check(out)


def test_released_checkpoint_loads_and_runs():
    ck = os.environ.get("EGOLITE_CKPT") or str(ROOT / "results" / "egolite" / "r_v2" / "last.pth")
    if not os.path.isfile(ck):
        pytest.skip("checkpoint not downloaded")
    state = torch.load(ck, map_location="cpu")
    model = build_from_args(dict(state["args"]), pretrained=False).eval()
    model.load_state_dict(state["model"])
    with torch.no_grad():
        out = model(*_pair())
    _check(out)
    # a pure translation-like shift of random noise has no meaningful pose; only sanity of the output is checked
    assert np.isfinite(out["t"].numpy()).all()
