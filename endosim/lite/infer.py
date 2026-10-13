from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from endosim.eval.metrics import load_pose_txt
from endosim.eval.protocol import list_color_frames, select_frame_indices
from endosim.geometry.pose_graph import optimize_pose_graph
from endosim.lite.data import load_frames, scaled_K
from endosim.lite.egolite import chain, pixel_rho


def _intrinsics(seq_dir: str) -> np.ndarray:
    intr = json.load(open(os.path.join(seq_dir, "intrinsics.json"), encoding="utf-8"))
    if "fx" not in intr and "K" in intr:
        K = np.asarray(intr["K"], dtype=np.float64).reshape(3, 3)
        intr = dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                    width=intr.get("width", 640), height=intr.get("height", 512))
    return scaled_K(intr, int(intr.get("width", 640)), int(intr.get("height", 512)))


def _run_pairs(model, img, ph, K, ii, jj, batch=16, extra=()):
    """Run arbitrary pair indices in chunks and concatenate pair-batched outputs."""
    outs = []
    for st in range(0, len(ii), batch):
        a, b = ii[st:st + batch], jj[st:st + batch]
        Kb = K[None].expand(len(a), -1, -1) if K.ndim == 2 else K[a]
        outs.append(model(img[a], img[b], Kb, ph[a], ph[b]))
    if not outs:
        raise ValueError("no pairs to process")
    merged = {}
    for key in outs[0]:
        vals = []
        ok = True
        for st, o in zip(range(0, len(ii), batch), outs):
            v = o[key]
            n_chunk = min(batch, len(ii) - st)
            if torch.is_tensor(v) and v.ndim and v.shape[0] == n_chunk:
                vals.append(v)
            else:
                ok = False
                break
        if ok and vals:
            merged[key] = torch.cat(vals, 0)
        else:
            # shared (non pair-batched) outputs such as the u/v grid are
            # identical across chunks; keep the first chunk's value
            merged[key] = outs[0][key]
    return merged


def chained_rho(o, mode):
    if mode == "one":
        return torch.ones_like(o["rho"])
    if mode == "median" or o["R"].shape[0] < 2:
        return o["rho"]
    sub = {k: o[k][:-1] for k in ("zp", "f_cam", "logits", "rho")}
    sub["u"], sub["v"] = o["u"], o["v"]
    return torch.cat([pixel_rho(sub, o["Z"][1:]), o["rho"][-1:]])


@torch.no_grad()
def infer_sequence(model, seq_dir, device="cuda", max_frames=64, batch=16, skips=(1,), rho_mode="pixel"):
    gt_all = load_pose_txt(os.path.join(seq_dir, "pose_c2w.txt"))
    idx = select_frame_indices(gt_all, "uniform", max_frames=max_frames)
    paths = list_color_frames(seq_dir, indices=idx)
    imgs, phs, _ = load_frames(paths)
    K = torch.from_numpy(_intrinsics(seq_dir)).to(device)
    img = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
    ph = torch.from_numpy(phs)[:, None].to(device)
    n = len(paths)
    if n < 2:
        raise ValueError(f"need >=2 frames, got {n}")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    ii = torch.arange(n - 1, device=device)
    o1 = _run_pairs(model, img, ph, K, ii, ii + 1, batch)
    R1, t1 = o1["R"], o1["t"]
    rho1 = chained_rho(o1, "one" if getattr(model, "abs_depth", False) else rho_mode)
    T = chain(R1, t1, rho1).cpu().numpy().astype(np.float64)
    n_pairs = n - 1
    extra = [gap for gap in skips if gap > 1 and gap < n]
    if extra:
        scale = np.concatenate([[1.0], np.cumprod(rho1.cpu().numpy().astype(np.float64))])
        edges = {}
        for k in range(n - 1):
            M = np.eye(4)
            M[:3, :3] = R1[k].cpu().numpy()
            M[:3, 3] = t1[k].cpu().numpy() * scale[k]
            edges[(k, k + 1)] = np.linalg.inv(M)
        for gap in extra:
            a = torch.arange(n - gap, device=device)
            oo = _run_pairs(model, img, ph, K, a, a + gap, batch)
            n_pairs += n - gap
            for k, (R, t) in enumerate(zip(oo["R"].cpu().numpy(), oo["t"].cpu().numpy())):
                M = np.eye(4)
                M[:3, :3] = R
                M[:3, 3] = t * scale[k]
                edges[(k, k + gap)] = np.linalg.inv(M)
        T, _ = optimize_pose_graph(n, edges, init=T, iters=20)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return T, gt_all[idx], idx, time.time() - t0, n_pairs


@torch.no_grad()
def infer_sequence_ba(model, seq_dir, device="cuda", max_frames=64, window=12, stride=6,
                      gaps=(1, 2), iters=4, batch=16):
    """Offline dense BA variant, reported separately from the real-time pair tracker."""
    from endosim.lite.window_ba import dense_ba
    gt_all = load_pose_txt(os.path.join(seq_dir, "pose_c2w.txt"))
    idx = select_frame_indices(gt_all, "uniform", max_frames=max_frames)
    paths = list_color_frames(seq_dir, indices=idx)
    imgs, phs, _ = load_frames(paths)
    n = len(paths)
    if n < 2:
        raise ValueError("need at least two frames")
    K = torch.from_numpy(_intrinsics(seq_dir)).to(device)
    img = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
    ph = torch.from_numpy(phs)[:, None].to(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    ii = torch.arange(n - 1, device=device)
    o1 = _run_pairs(model, img, ph, K, ii, ii + 1, batch)
    rho = chained_rho(o1, "one" if getattr(model, "abs_depth", False) else "pixel")
    T0 = chain(o1["R"], o1["t"], rho)
    N = len(T0)
    scale = torch.cat([torch.ones(1, device=device), torch.cumprod(rho, 0)])
    T0m = T0.clone()
    T0m[:, :3, 3] *= 50.0
    Tcw = torch.linalg.inv(T0m)
    Z = o1["Z"].float() * 50.0
    dinv = torch.cat([1 / Z, 1 / Z[-1:]], 0)
    edges, flows, weights, used = [], [], [], 0
    for gap in sorted(set(int(x) for x in gaps)):
        if gap <= 0 or gap > N:
            continue
        if gap == 1:
            oo = o1
        else:
            a = torch.arange(N - gap, device=device)
            oo = _run_pairs(model, img, ph, K, a, a + gap, batch)
        for k in range(N - gap):
            edges.append((k, k + gap))
        flows.append(oo["f_obs"].float())
        weights.append(oo["w"].float())
        used += N - gap
    u, v = o1["u"].float(), o1["v"].float()
    Tb, _ = dense_ba(Tcw, dinv, dinv, edges, torch.cat(flows), torch.cat(weights), K, u, v,
                     iters=iters, prior_sigma=0.15, huber=1.5, damping=1e-3, prior_weight=1.0)
    est = torch.linalg.inv(Tb).cpu().numpy().astype(np.float64)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return est, gt_all[idx], idx, time.time() - t0, used


@torch.no_grad()
def infer_sequence_kf(model, seq_dir, device="cuda", max_frames=64, tau=6.0, chunk=8):
    """Keyframe-relative flow-threshold route; experimental."""
    gt_all = load_pose_txt(os.path.join(seq_dir, "pose_c2w.txt"))
    idx = select_frame_indices(gt_all, "uniform", max_frames=max_frames)
    paths = list_color_frames(seq_dir, indices=idx)
    imgs, phs, _ = load_frames(paths)
    K = torch.from_numpy(_intrinsics(seq_dir)).to(device)
    img = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
    ph = torch.from_numpy(phs)[:, None].to(device)
    n = len(paths)
    if n < 2:
        raise ValueError("need at least two frames")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    T = np.repeat(np.eye(4)[None], n, axis=0)
    kf, i, used = 0, 1, 0
    while i < n:
        end = min(n, i + chunk)
        ii = torch.full((end - i,), kf, device=device, dtype=torch.long)
        jj = torch.arange(i, end, device=device)
        o = _run_pairs(model, img, ph, K, ii, jj, chunk, extra=("f_obs",))
        used += end - i
        mag = o["f_obs"].norm(dim=1).flatten(1).median(1).values.cpu().numpy()
        over = np.flatnonzero(mag > tau)
        last = int(over[0]) if len(over) else end - i - 1
        for j in range(last + 1):
            M = np.eye(4)
            M[:3, :3] = o["R"][j].cpu().numpy()
            M[:3, 3] = o["t"][j].cpu().numpy()
            T[i + j] = T[kf] @ np.linalg.inv(M)
        if len(over):
            kf = i + last
        i = i + last + 1
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return T, gt_all[idx], idx, time.time() - t0, used


@torch.no_grad()
def _infer_pair_diagnostics(model, seq_dir, device="cuda", max_frames=0, batch=16):
    """Save actual per-pair network/solver outputs with correctly directed GT transform."""
    gt = load_pose_txt(os.path.join(seq_dir, "pose_c2w.txt"))
    idx = np.arange(len(gt)) if max_frames <= 0 else select_frame_indices(gt, "uniform", max_frames=max_frames)
    paths = list_color_frames(seq_dir, indices=idx)
    imgs, phs, _ = load_frames(paths)
    K = torch.from_numpy(_intrinsics(seq_dir)).to(device)
    img = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
    ph = torch.from_numpy(phs)[:, None].to(device)
    gt_t = torch.from_numpy(gt).to(device=device, dtype=torch.float64)
    records, st = [], 0
    while st < len(idx) - 1:
        end = min(st + batch, len(idx) - 1)
        pair_ids = torch.arange(st, end + 1, device=device)
        Kb = K[None].expand(len(pair_ids), -1, -1) if K.ndim == 2 else K[pair_ids]
        o = model(img[pair_ids], img[pair_ids + 1], Kb, ph[pair_ids], ph[pair_ids + 1])
        if len(pair_ids) > 1:
            sub = {k: o[k][:-1] for k in ("zp", "f_cam", "logits", "rho")}
            sub["u"], sub["v"] = o["u"], o["v"]
            rho_pix = pixel_rho(sub, o["Z"][1:]).detach().cpu().numpy()
        else:
            rho_pix = np.zeros(0, dtype=np.float32)
        for q in range(end - st):
            k = st + q
            Hi = o["H"][q].detach().float()
            ev = torch.linalg.eigvalsh(Hi).clamp(min=0)
            pos = ev[ev > ev[-1] * 1e-7]
            cond = float(ev[-1] / pos.min()) if len(pos) else float("inf")
            Grel = torch.linalg.inv(gt_t[idx[k + 1]]) @ gt_t[idx[k]]
            gtvec = Grel[:3, 3]
            rho_model = float(o["rho"][q])
            rho_pixel = float(rho_pix[q]) if q < len(rho_pix) else rho_model
            records.append({
                "pair_index": int(k), "frame_i": int(idx[k]), "frame_j": int(idx[k + 1]),
                "pred_R_source_to_target": o["R"][q].detach().float().cpu().numpy().tolist(),
                "pred_t_source_to_target_50mm": (o["t"][q].detach().float().cpu().numpy() * 50).tolist(),
                "gt_R_source_to_target": Grel[:3, :3].cpu().numpy().tolist(),
                "gt_t_source_to_target_mm": gtvec.cpu().numpy().tolist(),
                "gt_pair_step_mm": float(torch.linalg.norm(gtvec)),
                "rho_model_median_z": rho_model, "rho_chain_pixel_correspondence": rho_pixel,
                "depth_median_mm": float(o["Z"][q].median() * 50),
                "depth_p10_mm": float(o["Z"][q].flatten().quantile(.1) * 50),
                "depth_p90_mm": float(o["Z"][q].flatten().quantile(.9) * 50),
                "anchor_weight_mean": float(o["w"][q].mean()),
                "anchor_support_fraction": float((o["w"][q] > o["w"][q].median() * .25).float().mean()),
                "logvar_median": float(o["logvar"][q].median()),
                "flow_median_px": float(o["f_obs"][q].norm(dim=0).median()),
                "scene_flow_median_px": float(o["f_scene"][q].norm(dim=0).median()),
                "hessian_eigenvalues": ev.cpu().numpy().tolist(),
                "hessian_condition_thresholded": cond,
            })
        st = end
    return {"sequence": Path(seq_dir).name, "n_frames": len(idx), "frames": idx.tolist(),
            "records": records,
            "unit": "translation multiplied by fixed 50-mm unit; GT millimetres",
            "note": "forward source-camera-to-target-camera; pixel rho is diagnostic only; actual r_v2 abs_depth chain rho is unity."}
