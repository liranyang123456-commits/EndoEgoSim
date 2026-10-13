"""Published self-supervised endoscopic ego-motion networks (official SCARED-trained weights).

  afsfm    AF-SfMLearner (Shao et al., MedIA 2022), stage-wise model
  endodac  EndoDAC (Cui et al., MICCAI 2024)

Both use the Monodepth2 PoseNet (ResNet-18 on two stacked frames, 320x256). Following the
authors' evaluate_pose.py, the network sees [frame_{i+1}, frame_i] and
transformation_from_parameters(axisangle, translation) is the camera-(i+1)-to-camera-i
transform, so c2w_{i+1} = c2w_i @ T.
"""
from __future__ import annotations

import importlib
import os
import sys

import cv2
import numpy as np

ROOTS = {
    "afsfm": (r"D:\baselines\af_sfmlearner\code", r"D:\baselines\af_sfmlearner\stagewise\Model_MIA"),
    "endodac": (r"D:\baselines\endodac\code", r"D:\baselines\endodac\weights"),
}
W, H = 320, 256
_MODELS = {}


def _import_isolated(code_root, names):
    """Import modules from code_root without clashing with same-named packages of other repos."""
    clash = [k for k in list(sys.modules) if k.split(".")[0] in ("networks", "layers", "models", "utils")]
    saved = {k: sys.modules.pop(k) for k in clash}
    sys.path.insert(0, code_root)
    try:
        return [importlib.import_module(n) for n in names]
    finally:
        sys.path.remove(code_root)
        for k in [k for k in list(sys.modules) if k.split(".")[0] in ("networks", "layers", "models", "utils")]:
            sys.modules.pop(k)
        sys.modules.update(saved)


def _load(which, device):
    if which in _MODELS:
        return _MODELS[which]
    import torch
    code, wdir = ROOTS[which]
    if which == "afsfm":
        networks, layers = _import_isolated(code, ["networks", "layers"])
        enc = networks.ResnetEncoder(18, False, 2)
        dec = networks.PoseDecoder(enc.num_ch_enc, 1, 2)
    else:
        encoders, decoders, layers = _import_isolated(
            code, ["models.encoders", "models.decoders", "utils.layers"])
        enc = encoders.ResnetEncoder(18, False, 2)
        dec = decoders.PoseDecoder(enc.num_ch_enc, 1, 2)
    enc.load_state_dict(torch.load(os.path.join(wdir, "pose_encoder.pth"), map_location="cpu"))
    dec.load_state_dict(torch.load(os.path.join(wdir, "pose.pth"), map_location="cpu"))
    enc.to(device).eval()
    dec.to(device).eval()
    _MODELS[which] = (enc, dec, layers.transformation_from_parameters)
    return _MODELS[which]


def _read(p):
    bgr = cv2.imread(p)
    if bgr is None:
        raise FileNotFoundError(p)
    rgb = cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def run_posenet(frame_paths, which, batch=32):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc, dec, from_params = _load(which, device)
    imgs = torch.from_numpy(np.stack([_read(p) for p in frame_paths])).permute(0, 3, 1, 2)
    rel = []
    with torch.no_grad():
        for a in range(0, len(frame_paths) - 1, batch):
            b = min(a + batch, len(frame_paths) - 1)
            x = torch.cat([imgs[a + 1:b + 1], imgs[a:b]], 1).to(device)
            out = dec([enc(x)])
            axisangle, translation = out[0], out[1]
            rel.append(from_params(axisangle[:, 0], translation[:, 0]).double().cpu().numpy())
    rel = np.concatenate(rel) if rel else np.zeros((0, 4, 4))
    traj = np.repeat(np.eye(4)[None], len(frame_paths), 0)
    for i, T in enumerate(rel):
        traj[i + 1] = traj[i] @ T
    return traj
