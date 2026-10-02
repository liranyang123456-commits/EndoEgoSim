"""MD-VGGT 推理路由（一个权重，不新训）。

A  滑窗尺度门 + 重置（chain_window_poses_gated）
B  重叠窗投票的多跨度相对链（skip 1/2/4）
C  插入/大 hop 代理 → 强制成对链
D  低参照代理 → 成对链；高/中参照 → 全局窗

路由不用 GT：只看光流刚性和径向分量。
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..geometry.se3 import compose, relative, so3_exp, so3_log
from .protocol import (
    chain_window_poses_gated,
    sanitize_poses,
    sliding_windows,
    window_is_usable,
)
from .decouple import infer_pairwise_decoupled

RunFn = Callable  # (paths: list[str]) -> (N,4,4) c2w


def fuse_rotations(Rs: list[np.ndarray]) -> np.ndarray:
    logs = [so3_log(R) for R in Rs if np.isfinite(R).all()]
    if not logs:
        return np.eye(3)
    return so3_exp(np.mean(np.stack(logs), axis=0))


def fuse_poses(Ts: list[np.ndarray]) -> np.ndarray:
    Ts = [np.asarray(T, dtype=np.float64) for T in Ts if np.isfinite(T).all()]
    if not Ts:
        return np.eye(4)
    if len(Ts) == 1:
        return Ts[0]
    R = fuse_rotations([T[:3, :3] for T in Ts])
    t = np.median(np.stack([T[:3, 3] for T in Ts]), axis=0)
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = t
    return out


def flow_proxies(frame_paths: list[str], n_pairs: int = 5, side: int = 256) -> dict:
    """无 GT 的参照/插入/大位移代理。"""
    import cv2

    n = len(frame_paths)
    if n < 2:
        return {"ref_proxy": 0.5, "radial_frac": 0.0, "flow_mag": 0.0, "n_pairs": 0}
    picks = np.unique(np.linspace(0, n - 2, min(n_pairs, n - 1)).round().astype(int))
    mags, rads, resid = [], [], []
    for i in picks:
        a = cv2.imread(frame_paths[int(i)], cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(frame_paths[int(i) + 1], cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            continue
        a = cv2.resize(a, (side, side), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (side, side), interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        u, v = flow[..., 0], flow[..., 1]
        mag = np.sqrt(u * u + v * v)
        yy, xx = np.indices((side, side))
        cx, cy = (side - 1) / 2.0, (side - 1) / 2.0
        rx, ry = xx - cx, yy - cy
        rnorm = np.sqrt(rx * rx + ry * ry) + 1e-6
        radial = (u * rx + v * ry) / rnorm
        med_u, med_v = float(np.median(u)), float(np.median(v))
        res = np.sqrt((u - med_u) ** 2 + (v - med_v) ** 2)
        mags.append(float(np.median(mag)))
        rads.append(float(np.mean(np.abs(radial)) / (np.mean(mag) + 1e-6)))
        resid.append(float(np.median(res) / (np.median(mag) + 1e-6)))
    if not mags:
        return {"ref_proxy": 0.5, "radial_frac": 0.0, "flow_mag": 0.0, "n_pairs": 0}
    residual = float(np.median(resid))
    ref = float(np.clip(1.0 - residual, 0.0, 1.0))
    return {
        "ref_proxy": ref,
        "radial_frac": float(np.median(rads)),
        "flow_mag": float(np.median(mags)),
        "n_pairs": int(len(mags)),
    }


def decide_route(proxies: dict, global_est: np.ndarray | None = None,
                 hop_cut: float = 0.30) -> dict:
    """C+D：无 GT。全局窗单步 hop 过大或退化 → pairwise；否则 global。

    光流径向在本数据上不分桶，只作日志，不当开关。
    hop_cut 对 v6 全局窗：0.30 能抓住 8 条 ATE≥20 且只多切 5 条中高参照。
    """
    reasons = []
    mode = "global"
    hop_max = None
    if global_est is not None:
        est = np.asarray(global_est, dtype=np.float64)
        if not window_is_usable(est):
            mode = "pairwise"
            reasons.append("global_degenerate")
        elif len(est) >= 2:
            hops = np.linalg.norm(np.diff(est[:, :3, 3], axis=0), axis=1)
            hop_max = float(np.max(hops)) if np.isfinite(hops).all() else None
            if hop_max is not None and hop_max > hop_cut:
                mode = "pairwise"
                reasons.append("global_large_hop")
    if mode == "global" and not reasons:
        reasons.append("keep_global")
    out = {"mode": mode, "reasons": reasons, **proxies}
    if hop_max is not None:
        out["global_hop_max"] = hop_max
    return out


def _extract_skip_edges(est: np.ndarray, idx: np.ndarray,
                        skips: tuple[int, ...] = (1, 2, 4)) -> dict:
    edges: dict[tuple[int, int], list[np.ndarray]] = {}
    idx = np.asarray(idx, dtype=int)
    est = np.asarray(est, dtype=np.float64)
    pos = {int(fi): j for j, fi in enumerate(idx)}
    for skip in skips:
        for a in range(len(idx) - skip):
            i, j = int(idx[a]), int(idx[a + skip])
            if j - i != skip:
                # 下标不连续时仍用窗口内相邻 skip 步
                ia, ib = pos[int(idx[a])], pos[int(idx[a + skip])]
                T = relative(est[ia], est[ib])
            else:
                T = relative(est[a], est[a + skip])
            if np.isfinite(T).all():
                edges.setdefault((i, j), []).append(T)
    return edges


def compose_edge_graph(n: int, edges: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """从 0 出发，用 skip 1/2/4 边投票拼轨迹。"""
    P = np.zeros((n, 4, 4))
    P[0] = np.eye(4)
    filled = {0}
    for t in range(1, n):
        cands = []
        for skip in (1, 2, 4):
            s = t - skip
            if s < 0 or s not in filled:
                continue
            T = edges.get((s, t))
            if T is None:
                continue
            cands.append(compose(P[s], T))
        if not cands:
            # 没有边：复制上一已填帧（零运动）
            prev = max(i for i in filled if i < t)
            P[t] = P[prev].copy()
        else:
            P[t] = fuse_poses(cands)
        filled.add(t)
    return sanitize_poses(P)


def infer_pairwise(run: RunFn, frame_paths: list[str],
                   window: int = 8, stride: int = 4) -> tuple[np.ndarray, dict]:
    """B+C+D 成对链：重叠短窗独立估计，skip 1/2/4 边投票。"""
    n = len(frame_paths)
    if n < 2:
        return np.eye(4)[None], {"n_windows": 0, "n_edges": 0}
    wins = sliding_windows(n, min(window, n), max(1, stride))
    bucket: dict[tuple[int, int], list[np.ndarray]] = {}
    n_ok = 0
    for w in wins:
        paths = [frame_paths[int(i)] for i in w]
        try:
            est = sanitize_poses(np.asarray(run(paths), dtype=np.float64))
        except Exception:
            continue
        if not window_is_usable(est):
            continue
        n_ok += 1
        for (i, j), ts in _extract_skip_edges(est, w).items():
            bucket.setdefault((i, j), []).extend(ts)
    edges = {k: fuse_poses(v) for k, v in bucket.items() if v}
    if not edges:
        # 退回相邻 2 帧
        for i in range(n - 1):
            try:
                est = sanitize_poses(np.asarray(run([frame_paths[i], frame_paths[i + 1]]),
                                                dtype=np.float64))
            except Exception:
                continue
            if len(est) >= 2 and np.isfinite(est).all():
                edges[(i, i + 1)] = relative(est[0], est[1])
    traj = compose_edge_graph(n, edges)
    return traj, {"n_windows": int(n_ok), "n_edges": int(len(edges))}


def infer_global(run: RunFn, frame_paths: list[str]) -> tuple[np.ndarray, dict]:
    est = sanitize_poses(np.asarray(run(frame_paths), dtype=np.float64))
    return est, {"n_windows": 1, "usable": bool(window_is_usable(est))}


def infer_clip(run: RunFn, frame_paths: list[str],
               pair_window: int = 8, pair_stride: int = 4,
               global_est: np.ndarray | None = None,
               seg_ckpt: str | None = None,
               device: str = "cuda",
               run_full=None,
               pgo: bool = False,
               hop_cut: float = 0.30,
               force_pairwise: bool = False) -> tuple[np.ndarray, dict]:
    """单段 clip：先全局窗（或复用），大 hop / 退化再走多跨度相对链。

    force_pairwise=True 时跳过全局推理与路由判定，直接走成对链
    （用于只对已确认 pairwise 的序列做精化重跑）。"""
    if force_pairwise:
        est, extra = infer_pairwise_decoupled(
            run, frame_paths, seg_ckpt=seg_ckpt,
            window=pair_window, stride=pair_stride, device=device,
            run_full=run_full, pgo=pgo)
        info = {"mode": "pairwise", "forced": True, **extra}
        return sanitize_poses(est), info
    proxies = flow_proxies(frame_paths)
    extra_g = {}
    if global_est is None:
        global_est, extra_g = infer_global(run, frame_paths)
    else:
        global_est = sanitize_poses(np.asarray(global_est, dtype=np.float64))
        extra_g = {"n_windows": 1, "usable": bool(window_is_usable(global_est)),
                   "reused_global": True}
    route = decide_route(proxies, global_est, hop_cut=hop_cut)
    if route["mode"] == "pairwise":
        est, extra = infer_pairwise_decoupled(
            run, frame_paths, seg_ckpt=seg_ckpt,
            window=pair_window, stride=pair_stride, device=device,
            run_full=run_full, pgo=pgo)
        extra = {**extra_g, **extra}
    else:
        est, extra = global_est, extra_g
    info = {**route, **extra}
    return sanitize_poses(est), info


def infer_sliding(run: RunFn, frame_paths: list[str],
                  window: int = 16, stride: int = 8,
                  pair_window: int = 8, pair_stride: int = 4,
                  seg_ckpt: str | None = None,
                  device: str = "cuda",
                  run_full=None,
                  pgo: bool = False,
                  hop_cut: float = 0.30) -> tuple[np.ndarray, np.ndarray, dict]:
    """长视频：每窗先全局，退化则该窗改 pairwise，再尺度门拼接。"""
    n = len(frame_paths)
    wins = sliding_windows(n, window, stride)
    packed = []
    n_pair = 0
    n_drop = 0
    for w in wins:
        paths = [frame_paths[int(i)] for i in w]
        try:
            est = sanitize_poses(np.asarray(run(paths), dtype=np.float64))
        except Exception:
            est = None
        hops = None
        if est is not None and len(est) >= 2 and np.isfinite(est[:, :3, 3]).all():
            hops = np.linalg.norm(np.diff(est[:, :3, 3], axis=0), axis=1)
        large = hops is not None and float(hops.max()) > hop_cut
        if est is None or not window_is_usable(est) or large:
            est, _ = infer_pairwise_decoupled(
                run, paths, seg_ckpt=seg_ckpt,
                window=pair_window, stride=pair_stride, device=device,
                run_full=run_full, pgo=pgo)
            n_pair += 1
            if not window_is_usable(est):
                n_drop += 1
                continue
        packed.append((w, est))
    est, idx, stats = chain_window_poses_gated(packed, with_scale=True)
    stats["n_window_pairwise"] = n_pair
    stats["n_window_unusable"] = n_drop
    return est, idx, stats
