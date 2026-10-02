"""评测帧选择与滑窗位姿拼接。

SCARED 关键帧崩溃的主因之一是协议: 对稀疏/长轨迹做均匀 32 帧子采样,
会把 1–5mm 的相邻帧 hop 放大到 55–266mm, 前馈匹配全部失败。
本模块提供:
- consecutive / stride / adaptive / uniform 选帧
- sliding: 重叠窗口独立估计, Sim(3) 拼回全局轨迹
- 路径索引序列: color_index.json 指向原始磁盘帧 (不拷贝)
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from ..geometry.se3 import relative, so3_log
from .align import umeyama_alignment


def list_color_frames(seq_dir: str, indices=None) -> list[str]:
    """列出序列帧路径: 优先本地 color/, 否则读 color_index.json (原始路径引用)。

    indices: 若给定且源是视频, 只解码这些下标, 返回与 indices 对齐的路径。
    """
    color_dir = os.path.join(seq_dir, "color")
    frames = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    if not frames:
        frames = sorted(glob.glob(os.path.join(color_dir, "*.jpg")))
    if frames:
        if indices is None:
            return frames
        return [frames[int(i)] for i in indices if 0 <= int(i) < len(frames)]
    idx_p = os.path.join(seq_dir, "color_index.json")
    if not os.path.exists(idx_p):
        return []
    with open(idx_p, encoding="utf-8") as f:
        idx = json.load(f)
    if isinstance(idx, dict) and idx.get("type") == "video":
        from ..dataset.real_access import (
            materialize_video_frames, materialize_video_indices)
        if indices is not None:
            return materialize_video_indices(seq_dir, idx, indices)
        return materialize_video_frames(seq_dir, idx)
    if isinstance(idx, list):
        paths = [p for p in idx if isinstance(p, str) and os.path.exists(p)]
        if indices is None:
            return paths
        return [paths[int(i)] for i in indices if 0 <= int(i) < len(paths)]
    keys = sorted(idx, key=lambda k: int(k) if str(k).isdigit() else str(k))
    return [idx[k] for k in keys if os.path.exists(idx[k])]


def _steps_mm(poses: np.ndarray) -> np.ndarray:
    rel = np.stack([relative(poses[i], poses[i + 1]) for i in range(len(poses) - 1)])
    return np.linalg.norm(rel[:, :3, 3], axis=1)


def select_frame_indices(poses: np.ndarray, protocol: str = "uniform",
                         max_frames: int = 0, stride: int = 1,
                         max_step_mm: float = 12.0) -> np.ndarray:
    """按协议选帧下标 (与 GT 位姿对齐)。

    uniform:     全序列均匀抽 max_frames (旧协议, SCARED 稀疏序列上会放大基线)
    consecutive: 取开头连续 max_frames 帧
    stride:      固定步长抽样
    adaptive:    贪心: 下帧累计平移尽量落在 (0.3*max_step, max_step] mm
    """
    n = len(poses)
    idx = np.arange(n)
    if n == 0:
        return idx
    protocol = (protocol or "uniform").lower()
    if protocol == "consecutive":
        k = n if max_frames <= 0 else min(max_frames, n)
        return idx[:k]
    if protocol == "stride":
        s = max(int(stride), 1)
        out = idx[::s]
        if max_frames > 0 and len(out) > max_frames:
            out = out[:max_frames]
        return out
    if protocol == "adaptive":
        return _adaptive_indices(poses, max_frames=max_frames or n,
                                 max_step_mm=max_step_mm)
    # uniform (default)
    if max_frames <= 0 or n <= max_frames:
        return idx
    return np.unique(np.linspace(0, n - 1, max_frames).round().astype(int))


def _adaptive_indices(poses: np.ndarray, max_frames: int,
                      max_step_mm: float) -> np.ndarray:
    """贪心选帧, 相邻选中帧的平移 ≤ max_step_mm, 尽量用满预算。"""
    n = len(poses)
    if n <= 2:
        return np.arange(n)
    chosen = [0]
    acc = 0.0
    min_step = 0.3 * max_step_mm
    for i in range(1, n):
        rel = relative(poses[chosen[-1]], poses[i])
        d = float(np.linalg.norm(rel[:3, 3]))
        rot = float(np.rad2deg(np.linalg.norm(so3_log(rel[:3, :3]))))
        if d < min_step and i < n - 1:
            acc = d
            continue
        if d > max_step_mm and i > chosen[-1] + 1:
            # 超限: 退回到上一合法候选 (i-1)
            if i - 1 not in chosen:
                chosen.append(i - 1)
            else:
                chosen.append(i)
        else:
            chosen.append(i)
        if len(chosen) >= max_frames:
            break
        _ = rot, acc
    if chosen[-1] != n - 1 and len(chosen) < max_frames:
        chosen.append(n - 1)
    # 去重保序
    out, seen = [], set()
    for i in chosen:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return np.asarray(out[:max_frames], dtype=int)


def sliding_windows(n: int, window: int, stride: int) -> list[np.ndarray]:
    """重叠窗口下标列表。最后一窗强制贴尾, 保证覆盖末帧。"""
    window = max(int(window), 2)
    stride = max(int(stride), 1)
    if n <= window:
        return [np.arange(n)]
    wins = []
    start = 0
    while start + window <= n:
        wins.append(np.arange(start, start + window))
        start += stride
    if wins[-1][-1] != n - 1:
        wins.append(np.arange(n - window, n))
    # 去完全重复窗
    uniq, keys = [], set()
    for w in wins:
        k = (int(w[0]), int(w[-1]))
        if k not in keys:
            uniq.append(w)
            keys.add(k)
    return uniq


def chain_window_poses(windows: list[tuple[np.ndarray, np.ndarray]],
                       with_scale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """把各窗独立估计的 c2w 用重叠帧 Sim(3)/SE(3) 拼成全局轨迹。

    windows: [(frame_idx (W,), est_c2w (W,4,4)), ...]
    返回 (est_global (M,4,4), used_idx (M,)) 按帧号升序, 重叠处保留先到者。
    """
    if not windows:
        return np.zeros((0, 4, 4)), np.zeros((0,), dtype=int)
    acc: dict[int, np.ndarray] = {}
    for k, (idx, est) in enumerate(windows):
        idx = np.asarray(idx, dtype=int)
        est = np.asarray(est, dtype=np.float64)
        if k == 0:
            for j, fi in enumerate(idx):
                acc[int(fi)] = est[j].copy()
            continue
        overlap = [int(fi) for fi in idx if int(fi) in acc]
        if len(overlap) >= 2:
            src = np.stack([est[int(np.where(idx == fi)[0][0]), :3, 3] for fi in overlap])
            dst = np.stack([acc[fi][:3, 3] for fi in overlap])
            s, R, t = umeyama_alignment(src, dst, with_scale=with_scale)
        elif len(overlap) == 1:
            fi = overlap[0]
            j = int(np.where(idx == fi)[0][0])
            # 单点: 用该帧相对位姿把窗对齐到已有轨迹 (不缩放)
            T_acc = acc[fi]
            T_est = est[j]
            T_align = T_acc @ np.linalg.inv(T_est)
            s, R, t = 1.0, T_align[:3, :3], T_align[:3, 3]
        else:
            # 无重叠: 接到最近已有帧
            nearest = min(acc, key=lambda f: abs(f - int(idx[0])))
            T_align = acc[nearest] @ np.linalg.inv(est[0])
            s, R, t = 1.0, T_align[:3, :3], T_align[:3, 3]
        for j, fi in enumerate(idx):
            fi = int(fi)
            if fi in acc:
                continue
            T = np.eye(4)
            T[:3, :3] = R @ est[j, :3, :3]
            T[:3, 3] = s * (R @ est[j, :3, 3]) + t
            acc[fi] = T
    used = np.array(sorted(acc), dtype=int)
    out = np.stack([acc[i] for i in used])
    return out, used


SCALE_LO = 1e-3
SCALE_HI = 1e3
HOP_CAP_MM = 1.0e4


def sanitize_poses(est: np.ndarray, hop_cap_mm: float = HOP_CAP_MM) -> np.ndarray:
    """把非有限 / 单步爆炸的 c2w 收成可用轨迹。爆炸步改复制上一帧（零运动）。"""
    est = np.asarray(est, dtype=np.float64).copy()
    if len(est) == 0:
        return est
    if not np.isfinite(est[0]).all():
        est[0] = np.eye(4)
    for i in range(1, len(est)):
        if not np.isfinite(est[i]).all():
            est[i] = est[i - 1].copy()
            continue
        hop = float(np.linalg.norm(est[i, :3, 3] - est[i - 1, :3, 3]))
        if not np.isfinite(hop) or hop > hop_cap_mm:
            est[i] = est[i - 1].copy()
    return est


def window_is_usable(est: np.ndarray, hop_cap_mm: float = HOP_CAP_MM) -> bool:
    est = np.asarray(est, dtype=np.float64)
    if est.ndim != 3 or est.shape[-2:] != (4, 4) or len(est) < 2:
        return False
    if not np.isfinite(est).all():
        return False
    hops = np.linalg.norm(np.diff(est[:, :3, 3], axis=0), axis=1)
    if not np.isfinite(hops).all():
        return False
    if float(hops.max()) > hop_cap_mm:
        return False
    return True


def chain_window_poses_gated(
    windows: list[tuple[np.ndarray, np.ndarray]],
    with_scale: bool = True,
    scale_lo: float = SCALE_LO,
    scale_hi: float = SCALE_HI,
    hop_cap_mm: float = HOP_CAP_MM,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """滑窗拼接，带尺度门与重置。爆炸窗不写入全局轨迹。

    返回 (est_global, used_idx, stats)。
    """
    stats = {"n_windows": len(windows), "n_used": 0, "n_scale_clamp": 0,
             "n_reset": 0, "n_dropped": 0}
    if not windows:
        return np.zeros((0, 4, 4)), np.zeros((0,), dtype=int), stats
    acc: dict[int, np.ndarray] = {}
    for k, (idx, est) in enumerate(windows):
        idx = np.asarray(idx, dtype=int)
        est = sanitize_poses(np.asarray(est, dtype=np.float64), hop_cap_mm)
        if not window_is_usable(est, hop_cap_mm):
            stats["n_dropped"] += 1
            continue
        if not acc:
            for j, fi in enumerate(idx):
                acc[int(fi)] = est[j].copy()
            stats["n_used"] += 1
            continue
        overlap = [int(fi) for fi in idx if int(fi) in acc]
        reset = False
        if len(overlap) >= 2:
            src = np.stack([est[int(np.where(idx == fi)[0][0]), :3, 3] for fi in overlap])
            dst = np.stack([acc[fi][:3, 3] for fi in overlap])
            s, R, t = umeyama_alignment(src, dst, with_scale=with_scale)
            if (not np.isfinite(s)) or s < scale_lo or s > scale_hi:
                s, R, t = umeyama_alignment(src, dst, with_scale=False)
                stats["n_scale_clamp"] += 1
            aligned = s * (src @ R.T) + t
            resid = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
            if not np.isfinite(resid) or resid > hop_cap_mm:
                reset = True
        elif len(overlap) == 1:
            fi = overlap[0]
            j = int(np.where(idx == fi)[0][0])
            T_align = acc[fi] @ np.linalg.inv(est[j])
            s, R, t = 1.0, T_align[:3, :3], T_align[:3, 3]
            if not (np.isfinite(R).all() and np.isfinite(t).all()):
                reset = True
        else:
            nearest = min(acc, key=lambda f: abs(f - int(idx[0])))
            T_align = acc[nearest] @ np.linalg.inv(est[0])
            s, R, t = 1.0, T_align[:3, :3], T_align[:3, 3]
            if not (np.isfinite(R).all() and np.isfinite(t).all()):
                reset = True
        if reset:
            stats["n_reset"] += 1
            fi = int(idx[0]) if int(idx[0]) in acc else (
                overlap[0] if overlap else min(acc, key=lambda f: abs(f - int(idx[0]))))
            j = int(np.where(idx == fi)[0][0]) if fi in set(int(x) for x in idx) else 0
            if fi not in acc:
                fi = min(acc, key=lambda f: abs(f - int(idx[0])))
                j = 0
            T_align = acc[fi] @ np.linalg.inv(est[j])
            s, R, t = 1.0, T_align[:3, :3], T_align[:3, 3]
            if not (np.isfinite(R).all() and np.isfinite(t).all()):
                stats["n_dropped"] += 1
                continue
        wrote = False
        for j, fi in enumerate(idx):
            fi = int(fi)
            if fi in acc:
                continue
            T = np.eye(4)
            T[:3, :3] = R @ est[j, :3, :3]
            T[:3, 3] = s * (R @ est[j, :3, 3]) + t
            if not np.isfinite(T).all():
                continue
            if acc:
                prev = acc[max(i for i in acc if i <= fi)] if any(i <= fi for i in acc) else acc[min(acc)]
                hop = float(np.linalg.norm(T[:3, 3] - prev[:3, 3]))
                if not np.isfinite(hop) or hop > hop_cap_mm:
                    continue
            acc[fi] = T
            wrote = True
        if wrote:
            stats["n_used"] += 1
        else:
            stats["n_dropped"] += 1
    if not acc:
        return np.zeros((0, 4, 4)), np.zeros((0,), dtype=int), stats
    used = np.array(sorted(acc), dtype=int)
    out = sanitize_poses(np.stack([acc[i] for i in used]), hop_cap_mm)
    return out, used, stats


def motion_stats_of_indices(poses: np.ndarray, idx: np.ndarray) -> dict:
    """选中帧的相邻平移/旋转统计 (诊断协议是否把基线放大)。"""
    if len(idx) < 2:
        return {"step_mm_mean": 0.0, "step_mm_max": 0.0,
                "rot_deg_mean": 0.0, "n": int(len(idx))}
    P = poses[idx]
    rel = np.stack([relative(P[i], P[i + 1]) for i in range(len(P) - 1)])
    trans = np.linalg.norm(rel[:, :3, 3], axis=1)
    rots = np.array([np.rad2deg(np.linalg.norm(so3_log(r[:3, :3]))) for r in rel])
    return {"step_mm_mean": float(trans.mean()), "step_mm_max": float(trans.max()),
            "rot_deg_mean": float(rots.mean()), "rot_deg_max": float(rots.max()),
            "n": int(len(idx))}
