"""Clip loading for EgoLite: images, pseudo-height, DIS flow, and simulator labels."""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from endosim.pseudoheight import pseudo_height_label

H_IN, W_IN = 256, 320
STRIDE = 8


def scaled_K(intr: dict, width: int, height: int) -> np.ndarray:
    sx, sy = W_IN / float(intr.get("width", width)), H_IN / float(intr.get("height", height))
    return np.array([[intr["fx"] * sx, 0, intr["cx"] * sx],
                     [0, intr["fy"] * sy, intr["cy"] * sy],
                     [0, 0, 1.0]], dtype=np.float32)


_DIS = None


def dis_flow(g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _DIS.calc(g0, g1, None).astype(np.float32)


def load_frames(paths: list[str], with_dis: bool = False):
    imgs, grays, phs = [], [], []
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            raise FileNotFoundError(p)
        bgr = cv2.resize(bgr, (W_IN, H_IN), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append(rgb)
        grays.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        phs.append(pseudo_height_label(rgb))
    if not with_dis:
        return np.stack(imgs), np.stack(phs), None
    flows = [dis_flow(grays[k], grays[k + 1]) for k in range(len(paths) - 1)]
    return np.stack(imgs), np.stack(phs), np.stack(flows) if flows else np.zeros((0, H_IN, W_IN, 2), np.float32)


def _barrel_free(d: str) -> bool:
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    return abs(float(meta.get("appearance", {}).get("barrel_k1", 0.0) or 0.0)) <= 1e-8


ABS_EPS_MM = 0.2

# CholecSeg8k ids -> anatomy motion groups: 0 anchor (abdominal wall, connective tissue,
# liver ligament), 1 large (liver, gallbladder, hepatic vein), 2 medium, 3 instrument
SEM_GROUP = np.full(256, -100, np.int64)
SEM_GROUP[[1, 6, 12]] = 0
SEM_GROUP[[2, 10, 11]] = 1
SEM_GROUP[[0, 3, 4, 7, 8]] = 2
SEM_GROUP[[5, 9, 13]] = 3


def anatomy_groups(d: str, k: int, h: int, w: int) -> np.ndarray:
    p = os.path.join(d, "anatomy", f"{k:06d}.png")
    a = cv2.imread(p, cv2.IMREAD_UNCHANGED) if os.path.exists(p) else None
    if a is None:
        return np.full((h, w), -100, np.int64)
    return SEM_GROUP[cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST)]


def abs_static_labels(d: str, k: int, inst: np.ndarray | None, h: int, w: int):
    """Motion layers on the (h, w) grid from the whole-sequence displacement of each point.

    0 = absolutely static (max displacement over the sequence < ABS_EPS_MM), 1 = moving
    tissue, 2 = moving object (instance id >= 2), -100 = unknown. None if frame k has no file.
    """
    p = os.path.join(d, "abs_motion", f"{k:06d}.npz")
    if not os.path.exists(p):
        return None
    md = np.load(p)["max_disp"].astype(np.float32)
    md = np.where(np.isfinite(md), md, np.inf)
    fh, fw = md.shape[0] // h, md.shape[1] // w
    md = md[: h * fh, : w * fw].reshape(h, fh, w, fw).max(axis=(1, 3))
    lab = np.where(md < ABS_EPS_MM, 0, 1).astype(np.int64)
    lab[~np.isfinite(md)] = -100
    if inst is not None:
        lab[(inst >= 2) & (lab == 1)] = 2
    return lab


class SimClips(Dataset):
    """Random clips from simulated sequences with depth, pose, and motion layers.

    roots: list of (directory, sampling weight). Sequences with abs_motion/ are
    labelled by whole-sequence (absolute) stationarity; the others keep the
    per-pair motion masks.
    """

    def __init__(self, root: str = "sim_data/train", clip: int = 6, length: int = 200000,
                 seed: int = 0, stride_p=(0.7, 0.2, 0.1), roots=None, use_abs_labels: bool = True):
        # use_abs_labels=False ignores abs_motion/ and uses the per-pair motion masks everywhere
        # (label-definition ablation of the revision; the reported checkpoint uses True).
        self.use_abs_labels = use_abs_labels
        roots = roots or [(root, 1.0)]
        self.groups = []
        for r, wgt in roots:
            dirs = sorted(glob.glob(os.path.join(r, "seq_*")))
            dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "pose_c2w.txt"))
                    and _barrel_free(d)]
            if dirs:
                self.groups.append((dirs, float(wgt)))
        self.dirs = [d for g, _ in self.groups for d in g]
        gw = np.array([wgt for _, wgt in self.groups])
        self.group_p = gw / gw.sum()
        self.clip = clip
        self.length = length
        self.seed = seed
        self.stride_p = np.asarray(stride_p) / np.sum(stride_p)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        for _ in range(20):
            dirs = self.groups[int(rng.choice(len(self.groups), p=self.group_p))][0]
            d = dirs[int(rng.integers(len(dirs)))]
            poses = np.loadtxt(os.path.join(d, "pose_c2w.txt")).reshape(-1, 4, 4)
            n = len(poses)
            stride = int(rng.choice(len(self.stride_p), p=self.stride_p)) + 1
            span = (self.clip - 1) * stride + 1
            if n >= span:
                break
        else:
            raise RuntimeError("no sequence long enough for the requested clip")
        a = int(rng.integers(0, n - span + 1))
        idx = a + np.arange(self.clip) * stride
        intr = json.load(open(os.path.join(d, "intrinsics.json"), encoding="utf-8"))
        K = scaled_K(intr, 640, 512)
        paths = [os.path.join(d, "color", f"{k:06d}.png") for k in idx]
        imgs, phs, _ = load_frames(paths)
        h, w = H_IN // STRIDE, W_IN // STRIDE
        depth, layer = [], []
        has_abs = self.use_abs_labels and os.path.isdir(os.path.join(d, "abs_motion"))
        sem = np.stack([anatomy_groups(d, int(k), h, w) for k in idx])
        for k in idx:
            z = cv2.imread(os.path.join(d, "depth", f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
            z = cv2.resize(z.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            depth.append(z)
            if has_abs:
                im = cv2.imread(os.path.join(d, "object_mask", f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
                inst = (cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)
                        if im is not None else None)
                lab = abs_static_labels(d, int(k), inst, h, w)
                layer.append(lab if lab is not None else np.full((h, w), -100, np.int64))
                continue
            mp = os.path.join(d, "motion_mask", f"{k:06d}.png")
            m = cv2.imread(mp, cv2.IMREAD_UNCHANGED) if os.path.exists(mp) else None
            if m is None:
                lab = np.full((h, w), -100, np.int64)
            else:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                lab = np.where(np.isin(m, [1, 2, 3]), m.astype(np.int64) - 1, -100)
            layer.append(lab)
        T = poses[idx]
        T = np.linalg.inv(T[0])[None] @ T
        return {
            "img": torch.from_numpy(imgs).permute(0, 3, 1, 2),
            "ph": torch.from_numpy(phs)[:, None],
            "K": torch.from_numpy(K),
            "depth": torch.from_numpy(np.stack(depth)),
            "layer": torch.from_numpy(np.stack(layer)),
            "sem": torch.from_numpy(sem),
            "T_wc": torch.from_numpy(T.astype(np.float32)),
        }


HAMLYN_ROOT = os.environ.get("HAMLYN_ROOT", "datasets/Hamlyn")  # optional, not used by the reported chain


def _hamlyn_K(seq_name: str, w0: int, h0: int) -> np.ndarray:
    """Left intrinsics of a rectified Hamlyn sequence at its image resolution (w0, h0).

    The calibration files of some sequences are given at half the stored image size;
    the scale is chosen so that the principal point falls near the image centre.
    """
    num = seq_name.replace("rectified", "")
    M = np.loadtxt(os.path.join(HAMLYN_ROOT, "calibration", num, "intrinsics.txt"))[:3, :3]

    def off(s):
        return max(abs(M[0, 2] * s - w0 / 2) / w0, abs(M[1, 2] * s - h0 / 2) / h0)
    s = min((1.0, 2.0), key=off)
    K = np.array([[M[0, 0] * s, 0, M[0, 2] * s], [0, M[1, 1] * s, M[1, 2] * s], [0, 0, 1.0]])
    return K if off(s) < 0.15 else None             # calibration inconsistent with the images


class RealClips(Dataset):
    """Unlabelled real laparoscopic video clips (Hamlyn rectified, left camera) with stereo depth.

    Returns images, pseudo-height, intrinsics at the network input size and stereo depth
    (mm) on the 1/8 grid (0 = invalid). No camera poses: used for self-supervision.
    """

    def __init__(self, root: str = HAMLYN_ROOT, clip: int = 3, length: int = 200000, seed: int = 0,
                 exclude=()):
        self.seqs = []
        for d in sorted(glob.glob(os.path.join(root, "rectified*", "rectified*"))):
            name = os.path.basename(d)
            if name in exclude or not os.path.isdir(os.path.join(d, "image01")):
                continue
            frames = sorted(glob.glob(os.path.join(d, "image01", "*")))
            if len(frames) > 10:
                h0, w0 = cv2.imread(frames[0]).shape[:2]
                K = _hamlyn_K(name, w0, h0)
                if K is not None:
                    self.seqs.append((name, d, frames, K, (w0, h0)))
        self.clip, self.length, self.seed = clip, length, seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, 7, i))
        name, d, frames, K0, (w0, h0) = self.seqs[int(rng.integers(len(self.seqs)))]
        stride = int(rng.integers(1, 4))
        a = int(rng.integers(0, len(frames) - (self.clip - 1) * stride))
        paths = [frames[a + k * stride] for k in range(self.clip)]
        imgs, phs, _ = load_frames(paths)
        K = np.array([[K0[0, 0] * W_IN / w0, 0, K0[0, 2] * W_IN / w0],
                      [0, K0[1, 1] * H_IN / h0, K0[1, 2] * H_IN / h0], [0, 0, 1]], np.float32)
        h, w = H_IN // STRIDE, W_IN // STRIDE
        dep = []
        for p in paths:
            dp = p.replace(os.sep + "image01" + os.sep, os.sep + "depth01" + os.sep)
            dp = os.path.splitext(dp)[0] + ".png"
            z = cv2.imread(dp, cv2.IMREAD_UNCHANGED) if os.path.exists(dp) else None
            dep.append(cv2.resize(z.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                       if z is not None else np.zeros((h, w), np.float32))
        return {"img": torch.from_numpy(imgs).permute(0, 3, 1, 2), "ph": torch.from_numpy(phs)[:, None],
                "K": torch.from_numpy(K), "depth": torch.from_numpy(np.stack(dep))}


class PseudoPoseClips(Dataset):
    """Hamlyn clips supervised by π³ window poses; stereo fixes the otherwise free scale.

    The labeler writes one NPZ per real video with frame indices, scaled camera-to-world poses,
    and a validity mask from comparison to stereo depth. Poses are expressed relative to the
    first frame of that window and translations are metric millimetres.
    """
    def __init__(self, label_dir: str, clip: int = 6, length: int = 200000, seed: int = 0):
        rc = RealClips(length=1)
        self.items = []
        for name, d, frames, K0, wh in rc.seqs:
            p = os.path.join(label_dir, f"{name}.npz")
            if not os.path.isfile(p):
                continue
            z = np.load(p)
            for wi in np.flatnonzero(z["valid"]):
                self.items.append((frames, K0, wh, z["idx"][wi].astype(np.int64),
                                   z["poses"][wi].astype(np.float32)))
        if not self.items:
            raise RuntimeError(f"no valid π³ pseudo-label windows in {label_dir}")
        self.clip, self.length, self.seed = int(clip), int(length), int(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, 11, int(i)))
        frames, K0, (w0, h0), idx, poses = self.items[int(rng.integers(len(self.items)))]
        max_s = max(1, (len(idx) - 1) // max(1, self.clip - 1))
        stride = int(rng.integers(1, min(3, max_s) + 1))
        span = (self.clip - 1) * stride + 1
        a = int(rng.integers(0, len(idx) - span + 1))
        sel = a + np.arange(self.clip) * stride
        raw = idx[sel]
        paths = [frames[int(k)] for k in raw]
        imgs, phs, _ = load_frames(paths)
        K = np.array([[K0[0, 0] * W_IN / w0, 0, K0[0, 2] * W_IN / w0],
                      [0, K0[1, 1] * H_IN / h0, K0[1, 2] * H_IN / h0], [0, 0, 1]], np.float32)
        h, w = H_IN // STRIDE, W_IN // STRIDE
        dep = []
        for p in paths:
            dp = p.replace(os.sep + "image01" + os.sep, os.sep + "depth01" + os.sep)
            dp = os.path.splitext(dp)[0] + ".png"
            z = cv2.imread(dp, cv2.IMREAD_UNCHANGED) if os.path.isfile(dp) else None
            dep.append(cv2.resize(z.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                       if z is not None else np.zeros((h, w), np.float32))
        T = poses[sel]
        T = np.linalg.inv(T[0])[None] @ T
        return {"img": torch.from_numpy(imgs).permute(0, 3, 1, 2), "ph": torch.from_numpy(phs)[:, None],
                "K": torch.from_numpy(K), "depth": torch.from_numpy(np.stack(dep)),
                "T_wc": torch.from_numpy(T.astype(np.float32))}


def worker_init(_):
    cv2.setNumThreads(1)
