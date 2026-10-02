"""MD-VGGT 的可选器械抑制与重叠窗口相对链。

论文计分路径只启用两项：原图/器械抑制图双跑，以及无 GT 的 hop 一致性选择。
形变组织始终保留。文件中保留的深度--光流精炼与静态锚定函数是未接入计分
路径的研究原型；不得把它们描述成当前 MD-VGGT 的已启用模块。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from typing import Callable

import cv2
import numpy as np

from ..geometry.se3 import compose, relative, so3_exp, so3_log
from ..geometry.pose_graph import optimize_pose_graph
from .protocol import (
    chain_window_poses_gated,
    sanitize_poses,
    sliding_windows,
    window_is_usable,
)

RunFn = Callable  # (paths: list[str]) -> (N,4,4) c2w


# ---------------------------------------------------------------------------
# 器械分割
# ---------------------------------------------------------------------------

_SEG_MODEL = None
_SEG_DEVICE = "cuda"


def load_instrument_seg(ckpt: str, device: str = "cuda"):
    """加载 DeepLabV3 器械分割。"""
    global _SEG_MODEL, _SEG_DEVICE
    if _SEG_MODEL is not None:
        return _SEG_MODEL
    import torch
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    m = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=2)
    sd = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(sd.get("model", sd))
    m.eval().to(device)
    _SEG_MODEL = m
    _SEG_DEVICE = device
    return m


def instrument_prob(frame_paths: list[str], ckpt: str, device: str = "cuda",
                    batch: int = 8) -> list[np.ndarray]:
    """返回每帧 instrument 概率图 (H,W) float32 [0,1]。"""
    import torch
    model = load_instrument_seg(ckpt, device)
    probs = []
    for i in range(0, len(frame_paths), batch):
        chunk = frame_paths[i:i + batch]
        xs = []
        for p in chunk:
            img = cv2.imread(p)
            img = cv2.resize(img, (518, 420))
            x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            xs.append(torch.from_numpy(x).permute(2, 0, 1))
        x = torch.stack(xs).to(device)
        with torch.no_grad():
            logits = model(x)["out"][:, 1]
            p = torch.sigmoid(logits).cpu().numpy()
        probs.extend([p[j] for j in range(len(chunk))])
    return probs


# ---------------------------------------------------------------------------
# 器械区域输入抑制
# ---------------------------------------------------------------------------

def _suppress_frame(img_bgr: np.ndarray, inst_prob: np.ndarray,
                    hard_thr: float = 0.5, soft_thr: float = 0.15,
                    blur_ksize: int = 41) -> np.ndarray:
    """对器械区域做软抑制：高置信 inpaint，中置信混合模糊。

    只抑制第 3 层（器械/独立运动物体）；第 2 层（形变组织）不动。
    """
    h, w = img_bgr.shape[:2]
    prob = cv2.resize(inst_prob, (w, h), interpolation=cv2.INTER_LINEAR)

    hard_mask = (prob > hard_thr).astype(np.uint8)
    if hard_mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        hard_mask = cv2.dilate(hard_mask, kernel, iterations=2)
        img_bgr = cv2.inpaint(img_bgr, hard_mask, 7, cv2.INPAINT_TELEA)

    soft_alpha = np.clip((prob - soft_thr) / max(hard_thr - soft_thr, 1e-6), 0, 1)
    if soft_alpha.max() > 0.01:
        blurred = cv2.GaussianBlur(img_bgr, (blur_ksize, blur_ksize), 0)
        alpha = (soft_alpha * 0.7)[..., None]  # 最多混合 70%
        img_bgr = (img_bgr * (1 - alpha) + blurred * alpha).astype(np.uint8)

    return img_bgr


def _create_suppressed_frames(frame_paths: list[str],
                              inst_probs: list[np.ndarray],
                              tmp_dir: str) -> list[str]:
    """对每帧生成器械抑制后的临时图像，返回路径列表。"""
    os.makedirs(tmp_dir, exist_ok=True)
    out = []
    for i, (path, prob) in enumerate(zip(frame_paths, inst_probs)):
        img = cv2.imread(path)
        if img is None:
            out.append(path)
            continue
        sup = _suppress_frame(img, prob)
        out_path = os.path.join(tmp_dir, f"{i:06d}.png")
        cv2.imwrite(out_path, sup)
        out.append(out_path)
    return out


# ---------------------------------------------------------------------------
# 光流残差：估第 1 层（静态参照）
# ---------------------------------------------------------------------------

def _flow_residual_mask(img0: np.ndarray, img1: np.ndarray,
                        T_prev_from_cur: np.ndarray, K: np.ndarray,
                        depth0: np.ndarray | None = None,
                        instrument0: np.ndarray | None = None,
                        side: int = 256) -> np.ndarray:
    """用当前 c2w 相对边解释 ego 流，残差大的是非静态。

    返回 (side,side) bool：True = 静态参照候选。
    """
    h0, w0 = img0.shape[:2]
    gray0 = cv2.cvtColor(cv2.resize(img0, (side, side)), cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(cv2.resize(img1, (side, side)), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u, v = flow[..., 0], flow[..., 1]

    yy, xx = np.indices((side, side))
    Ks = np.asarray(K, dtype=np.float64).copy()
    Ks[0, :] *= side / float(w0)
    Ks[1, :] *= side / float(h0)
    fx, fy, cx, cy = Ks[0, 0], Ks[1, 1], Ks[0, 2], Ks[1, 2]
    if depth0 is None:
        mag = np.sqrt(u * u + v * v)
        static = mag < np.median(mag)
    else:
        d0 = cv2.resize(depth0, (side, side), interpolation=cv2.INTER_LINEAR)
        z = np.maximum(d0, 1e-3)
        x3 = (xx - cx) * z / fx
        y3 = (yy - cy) * z / fy
        pts = np.stack([x3, y3, z], axis=-1).reshape(-1, 3)
        T_cur_from_prev = np.linalg.inv(T_prev_from_cur)
        pts1 = (
            T_cur_from_prev[:3, :3] @ pts.T
        ).T + T_cur_from_prev[:3, 3]
        z1 = np.maximum(pts1[:, 2], 1e-3)
        du_exp = (fx * pts1[:, 0] / z1 + cx).reshape(side, side) - xx
        dv_exp = (fy * pts1[:, 1] / z1 + cy).reshape(side, side) - yy
        res = np.sqrt((u - du_exp) ** 2 + (v - dv_exp) ** 2)
        static = res < np.median(res) * 1.5
    if instrument0 is not None:
        inst = cv2.resize(instrument0, (side, side), interpolation=cv2.INTER_NEAREST) > 0.3
        static = static & (~inst)
    return static


# ---------------------------------------------------------------------------
# 深度加权位姿精炼
# ---------------------------------------------------------------------------

def _weighted_kabsch(P: np.ndarray, Q: np.ndarray,
                     w: np.ndarray) -> np.ndarray:
    """加权 Kabsch：求 R,t 使 sum w_i |R P_i + t - Q_i|^2 最小。"""
    w_sum = w.sum()
    if w_sum < 1e-6:
        return np.eye(4)
    cp = (P * w[:, None]).sum(axis=0) / w_sum
    cq = (Q * w[:, None]).sum(axis=0) / w_sum
    Pc = P - cp
    Qc = Q - cq
    H = Pc.T @ (Qc * w[:, None])
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cq - R @ cp
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _refine_pair(T_vggt: np.ndarray,
                 depth0: np.ndarray, depth1: np.ndarray,
                 conf0: np.ndarray, conf1: np.ndarray,
                 K: np.ndarray,
                 inst0: np.ndarray | None, inst1: np.ndarray | None,
                 img0_bgr: np.ndarray, img1_bgr: np.ndarray,
                 min_corr: int = 30) -> np.ndarray:
    """用深度+光流残差+器械分割精炼相邻帧相对位姿。

    流程：
      1. Farneback 光流 → 像素对应
      2. 深度反投影 → 3D-3D 对应
      3. 权重 = depth_conf * 静态掩码 * (1 - 器械概率)
      4. 加权 Kabsch → 精炼相对位姿
      5. 平移尺度对齐到 VGGT 的平移模长
    """
    H, W = depth0.shape
    gray0 = cv2.cvtColor(cv2.resize(img0_bgr, (W, H)), cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(cv2.resize(img1_bgr, (W, H)), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    yy, xx = np.indices((H, W))
    pts0 = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float64)
    pts1 = pts0 + flow.reshape(-1, 2).astype(np.float64)

    # 有效范围
    valid = ((pts1[:, 0] >= 0) & (pts1[:, 0] < W)
             & (pts1[:, 1] >= 0) & (pts1[:, 1] < H))

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 帧0 反投影
    d0 = depth0.ravel().astype(np.float64)
    z0 = np.maximum(d0, 1e-3)
    P0 = np.stack([(pts0[:, 0] - cx) * z0 / fx,
                   (pts0[:, 1] - cy) * z0 / fy,
                   z0], axis=-1)

    # 帧1 在流对应位置采样深度
    pts1_f = pts1.astype(np.float32)
    d1 = cv2.remap(depth1.astype(np.float32),
                   pts1_f[:, 0].reshape(H, W),
                   pts1_f[:, 1].reshape(H, W),
                   cv2.INTER_LINEAR).ravel().astype(np.float64)
    z1 = np.maximum(d1, 1e-3)
    P1 = np.stack([(pts1[:, 0] - cx) * z1 / fx,
                   (pts1[:, 1] - cy) * z1 / fy,
                   z1], axis=-1)

    # 权重：conf * 静态 * 非器械
    c0 = conf0.ravel().astype(np.float64)
    c1 = cv2.remap(conf1.astype(np.float32),
                   pts1_f[:, 0].reshape(H, W),
                   pts1_f[:, 1].reshape(H, W),
                   cv2.INTER_LINEAR).ravel().astype(np.float64)
    mag = np.linalg.norm(flow.reshape(-1, 2), axis=1)
    static = (mag < np.median(mag)).astype(np.float64)

    w = c0 * c1 * static * valid.astype(np.float64)
    if inst0 is not None:
        i0 = cv2.resize(inst0, (W, H), interpolation=cv2.INTER_NEAREST).ravel()
        w *= np.clip(1.0 - i0, 0.05, 1.0)
    if inst1 is not None:
        i1 = cv2.resize(inst1, (W, H), interpolation=cv2.INTER_NEAREST).ravel()
        w *= np.clip(1.0 - i1, 0.05, 1.0)

    # 取 top 50% 权重对应
    pos_w = w[w > 0]
    if len(pos_w) < min_corr:
        return T_vggt
    thr = np.percentile(pos_w, 50)
    mask = w >= thr
    if mask.sum() < min_corr:
        return T_vggt

    # Kabsch above estimates previous-camera -> current-camera.  The VGGT
    # trajectory is c2w, whose compositional edge is current-camera ->
    # previous-camera.
    T_ref = np.linalg.inv(_weighted_kabsch(P0[mask], P1[mask], w[mask]))

    # 平移尺度对齐到 VGGT
    t_vggt_norm = np.linalg.norm(T_vggt[:3, 3])
    t_ref_norm = np.linalg.norm(T_ref[:3, 3])
    if t_ref_norm > 1e-6 and t_vggt_norm > 1e-6:
        T_ref[:3, 3] *= t_vggt_norm / t_ref_norm

    return T_ref


def _refine_window(poses: np.ndarray, depth: np.ndarray,
                   depth_conf: np.ndarray, intr: np.ndarray,
                   inst_probs: list[np.ndarray] | None,
                   frame_paths_win: list[str]) -> np.ndarray:
    """对窗口内所有相邻帧做深度加权精炼，返回精炼后的位姿序列。"""
    n = len(poses)
    if n < 2:
        return poses
    refined = poses.copy()
    imgs = [cv2.imread(p) for p in frame_paths_win]
    for i in range(n - 1):
        if imgs[i] is None or imgs[i + 1] is None:
            continue
        T_vggt = relative(refined[i], refined[i + 1])
        ip0 = inst_probs[i] if inst_probs else None
        ip1 = inst_probs[i + 1] if inst_probs else None
        T_ref = _refine_pair(T_vggt, depth[i], depth[i + 1],
                             depth_conf[i], depth_conf[i + 1],
                             intr[i], ip0, ip1,
                             imgs[i], imgs[i + 1])
        refined[i + 1] = compose(refined[i], T_ref)
    return refined


# ---------------------------------------------------------------------------
# 解耦成对链：双跑 + hop 一致性选择
# ---------------------------------------------------------------------------

RunFullFn = Callable  # (paths) -> (poses, depth, depth_conf, intrinsics)


def _max_hop(est: np.ndarray) -> float:
    if len(est) < 2:
        return 0.0
    hops = np.linalg.norm(np.diff(est[:, :3, 3], axis=0), axis=1)
    return float(np.max(hops)) if len(hops) else 0.0


def _pick_better(est_orig: np.ndarray | None,
                 est_supp: np.ndarray | None) -> tuple[np.ndarray | None, bool]:
    """hop 一致性选更优窗口估计。返回 (est, used_supp)。"""
    if est_orig is None and est_supp is None:
        return None, False
    if est_orig is None:
        return est_supp, True
    if est_supp is None:
        return est_orig, False
    ok_o = window_is_usable(est_orig)
    ok_s = window_is_usable(est_supp)
    if ok_o and not ok_s:
        return est_orig, False
    if ok_s and not ok_o:
        return est_supp, True
    if not ok_o and not ok_s:
        return est_orig, False
    hop_o = _max_hop(est_orig)
    hop_s = _max_hop(est_supp)
    if hop_s < hop_o * 0.8:
        return est_supp, True
    return est_orig, False


# ---------------------------------------------------------------------------
# 静态锚定 8 点法：用光流找静态区域 + ORB 特征 + Essential 矩阵
# ---------------------------------------------------------------------------

def _static_anchored_8pt(img0_bgr: np.ndarray, img1_bgr: np.ndarray,
                          K: np.ndarray,
                          inst0: np.ndarray | None = None,
                          inst1: np.ndarray | None = None,
                          t_scale: float = 1.0,
                          side: int = 320) -> np.ndarray | None:
    """静态锚定 8 点法：只用静态+非器械区域的 ORB 特征估 Essential。

    返回 (4,4) 相对位姿 T_01，或 None（失败）。
    t_scale: VGGT 的平移模长，用于恢复尺度。
    """
    h0, w0 = img0_bgr.shape[:2]
    gray0 = cv2.cvtColor(cv2.resize(img0_bgr, (side, side)), cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(cv2.resize(img1_bgr, (side, side)), cv2.COLOR_BGR2GRAY)

    # 1) 光流 → 静态掩码
    flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    static = (mag < np.median(mag)).astype(np.uint8)

    # 2) 去除器械区域
    if inst0 is not None:
        inst0_r = cv2.resize(inst0, (side, side), interpolation=cv2.INTER_NEAREST)
        static &= (inst0_r < 0.3).astype(np.uint8)
    if inst1 is not None:
        inst1_r = cv2.resize(inst1, (side, side), interpolation=cv2.INTER_NEAREST)
        static &= (inst1_r < 0.3).astype(np.uint8)

    # 3) ORB 特征（只在静态区域检测）
    orb = cv2.ORB_create(2000)
    k0, d0 = orb.detectAndCompute(gray0, static)
    k1, d1 = orb.detectAndCompute(gray1, static)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return None

    # 4) 匹配 + Essential
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = bf.match(d0, d1)
    if len(m) < 8:
        return None
    m = sorted(m, key=lambda x: x.distance)[:400]
    p0 = np.float32([k0[x.queryIdx].pt for x in m])
    p1 = np.float32([k1[x.trainIdx].pt for x in m])

    # 图像被直接缩放到正方形，内参需分别按横纵比例缩放
    sx, sy = side / float(w0), side / float(h0)
    K_s = K.copy()
    K_s[0, :] *= sx
    K_s[1, :] *= sy

    E, mask = cv2.findEssentialMat(p0, p1, K_s, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    if E is None:
        return None
    n_inlier, R, t, _ = cv2.recoverPose(E, p0, p1, K_s, mask=mask)
    if n_inlier < 15:
        return None

    # 5) 尺度对齐到 VGGT
    T_cur_from_prev = np.eye(4)
    T_cur_from_prev[:3, :3] = R
    T_cur_from_prev[:3, 3] = t.reshape(3) * t_scale
    return np.linalg.inv(T_cur_from_prev)


def _static_anchored_chain(frame_paths: list[str],
                            K: np.ndarray,
                            inst_probs: list[np.ndarray] | None,
                            vggt_poses: np.ndarray,
                            blend_thr: float = 0.3) -> np.ndarray:
    """用静态锚定 8 点法精炼 VGGT 的相对位姿链。

    对每个相邻帧对：
      1. 计算 VGGT 的相对位姿 T_vggt
      2. 计算静态锚定 8 点法的相对位姿 T_8pt
      3. 如果 T_8pt 有效且静态覆盖率高，混合两者
    """
    n = len(frame_paths)
    if n < 2:
        return vggt_poses

    refined = vggt_poses.copy()
    imgs = [cv2.imread(p) for p in frame_paths]

    for i in range(n - 1):
        if imgs[i] is None or imgs[i + 1] is None:
            continue
        T_vggt = relative(refined[i], refined[i + 1])
        t_scale = np.linalg.norm(T_vggt[:3, 3])
        if t_scale < 1e-6:
            continue

        ip0 = inst_probs[i] if inst_probs else None
        ip1 = inst_probs[i + 1] if inst_probs else None

        T_8pt = _static_anchored_8pt(imgs[i], imgs[i + 1], K,
                                      ip0, ip1, t_scale)
        if T_8pt is None:
            continue

        # 混合：如果 8 点法的旋转与 VGGT 一致，信任 8 点法更多
        R_diff = np.arccos(np.clip(
            (np.trace(T_vggt[:3, :3].T @ T_8pt[:3, :3]) - 1) / 2, -1, 1))
        if R_diff < blend_thr:
            # 旋转一致，用 8 点法的旋转 + VGGT 的平移方向
            alpha = 0.5  # 混合权重
            R_mix = so3_exp(alpha * so3_log(T_8pt[:3, :3])
                            + (1 - alpha) * so3_log(T_vggt[:3, :3]))
            t_mix = alpha * T_8pt[:3, 3] + (1 - alpha) * T_vggt[:3, 3]
            T_mix = np.eye(4)
            T_mix[:3, :3] = R_mix
            T_mix[:3, 3] = t_mix
            refined[i + 1] = compose(refined[i], T_mix)

    return refined


def infer_pairwise_decoupled(run: RunFn, frame_paths: list[str],
                             seg_ckpt: str | None = None,
                             window: int = 8, stride: int = 4,
                             device: str = "cuda",
                             run_full: RunFullFn | None = None,
                             pgo: bool = False,
                             ) -> tuple[np.ndarray, dict]:
    """重叠相对链，可选器械区域抑制。

    双跑策略：原始 + 器械抑制，hop 一致性选更优。
    ``run_full`` 仅为兼容旧调用保留；当前计分路径不执行深度/光流精炼。
    """
    n = len(frame_paths)
    if n < 2:
        return np.eye(4)[None], {"n_windows": 0, "n_edges": 0, "seg": False}

    inst_probs = None
    if seg_ckpt and os.path.isfile(seg_ckpt):
        inst_probs = instrument_prob(frame_paths, seg_ckpt, device)

    # 无分割权重：退回标准单跑成对链（隔离窗口/步长影响，也更快）
    if inst_probs is None:
        return _infer_pairwise_std(run, frame_paths, window, stride, pgo=pgo)

    _ = run_full

    # 双跑策略：原始 + 器械抑制，hop 一致性选更优
    tmp_dir = tempfile.mkdtemp(prefix="v6dec_")
    supp_paths = _create_suppressed_frames(frame_paths, inst_probs, tmp_dir)

    n_ok = 0
    n_supp = 0
    try:
        wins = sliding_windows(n, min(window, n), max(1, stride))
        bucket: dict[tuple[int, int], list[np.ndarray]] = {}
        for w in wins:
            orig = [frame_paths[int(i)] for i in w]
            supp = [supp_paths[int(i)] for i in w]

            est_o = None
            try:
                est_o = sanitize_poses(np.asarray(run(orig), dtype=np.float64))
            except Exception:
                pass
            est_s = None
            try:
                est_s = sanitize_poses(np.asarray(run(supp), dtype=np.float64))
            except Exception:
                pass

            est, used_s = _pick_better(est_o, est_s)
            if est is None or not window_is_usable(est):
                continue
            n_ok += 1
            if used_s:
                n_supp += 1

            for (i, j), ts in _extract_skip_edges(est, w).items():
                bucket.setdefault((i, j), []).extend(ts)

        edges = {k: fuse_poses(v) for k, v in bucket.items() if v}
        if not edges:
            for i in range(n - 1):
                try:
                    est = sanitize_poses(np.asarray(
                        run([frame_paths[i], frame_paths[i + 1]]),
                        dtype=np.float64))
                except Exception:
                    continue
                if len(est) >= 2 and np.isfinite(est).all():
                    edges[(i, i + 1)] = relative(est[0], est[1])
        traj = compose_edge_graph(n, edges)
        pgo_stats = {}
        if pgo:
            traj, pgo_stats = pgo_gn_refine(n, edges, traj, iters=20)
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    info = {"n_windows": int(n_ok), "n_edges": int(len(edges)),
            "seg": True, "n_supp_used": n_supp, "n_refined": 0,
            **pgo_stats}
    return traj, info


def _infer_pairwise_std(run: RunFn, frame_paths: list[str],
                        window: int = 8, stride: int = 4,
                        pgo: bool = False) -> tuple[np.ndarray, dict]:
    """无分割时的标准成对链（与 infer_pairwise 相同）。"""
    n = len(frame_paths)
    if n < 2:
        return np.eye(4)[None], {"n_windows": 0, "n_edges": 0, "seg": False}
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
        for i in range(n - 1):
            try:
                est = sanitize_poses(np.asarray(
                    run([frame_paths[i], frame_paths[i + 1]]),
                    dtype=np.float64))
            except Exception:
                continue
            if len(est) >= 2 and np.isfinite(est).all():
                edges[(i, i + 1)] = relative(est[0], est[1])
    traj = compose_edge_graph(n, edges)
    pgo_stats = {}
    if pgo:
        traj, pgo_stats = pgo_gn_refine(n, edges, traj, iters=20)
    return traj, {"n_windows": int(n_ok), "n_edges": int(len(edges)),
                  "seg": False, "pgo": bool(pgo), **pgo_stats}


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
                ia, ib = pos[int(idx[a])], pos[int(idx[a + skip])]
                T = relative(est[ia], est[ib])
            else:
                T = relative(est[a], est[a + skip])
            if np.isfinite(T).all():
                edges.setdefault((i, j), []).append(T)
    return edges


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


def compose_edge_graph(n: int, edges: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
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
            prev = max(i for i in filled if i < t)
            P[t] = P[prev].copy()
        else:
            P[t] = fuse_poses(cands)
        filled.add(t)
    return sanitize_poses(P)


def pgo_gn_refine(n: int, edges: dict[tuple[int, int], np.ndarray],
                  init: np.ndarray, iters: int = 20,
                  huber_delta: float | None = None) -> tuple[np.ndarray, dict]:
    """GN/LM 位姿图精化（Huber 鲁棒核），替代高斯-赛德尔松弛版 pgo_refine。

    与 pgo_refine 的差异：显式最小化 Σ ρ(e_ij)，e_ij = Log(Z_ij^{-1} X_i^{-1} X_j)，
    解析雅可比 + 稀疏 GN；离群边由 Huber 核自动降权。
    """
    clean = {k: T for k, T in edges.items() if np.isfinite(T).all()}
    if not clean:
        return sanitize_poses(init), {"pgo": "gn", "n_edges": 0}
    opt, stats = optimize_pose_graph(
        n, clean, init=init, huber_delta=huber_delta, iters=iters)
    stats = {**stats, "pgo": "gn"}
    return sanitize_poses(opt), stats


def pgo_refine(n: int, edges: dict[tuple[int, int], np.ndarray],
               init: np.ndarray, iters: int = 20) -> np.ndarray:
    """位姿图优化：用所有边约束做高斯-赛德尔松弛，分散贪心链累积的误差。

    边 (i,j) 存 T_ij = P_i^-1 @ P_j。对每个节点 t：
      前向边 (i,t): P_t = P_i @ T_it
      后向边 (t,j): P_t = P_j @ T_tj^-1
    多源候选用 fuse_poses（中位平移 + 对数均值旋转）鲁棒融合。
    """
    adj: list[list[tuple[int, np.ndarray, bool]]] = [[] for _ in range(n)]
    for (i, j), T in edges.items():
        if not np.isfinite(T).all():
            continue
        adj[j].append((i, T, True))
        adj[i].append((j, T, False))
    P = init.copy()
    for _ in range(iters):
        for t in range(1, n):
            cands = []
            for (nb, T, fwd) in adj[t]:
                if fwd:
                    cands.append(compose(P[nb], T))
                else:
                    Ti = np.linalg.inv(T)
                    if np.isfinite(Ti).all():
                        cands.append(compose(P[nb], Ti))
            if cands:
                P[t] = fuse_poses(cands)
    return sanitize_poses(P)
