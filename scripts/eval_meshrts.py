"""Mesh-RTS pose proxy on EndoEgoSim (UVH + Farneback + RANSAC Kabsch + ICP).

This is the geometric method in mia_paper.tex, not MD-VGGT.
Height is a truncated-gradient response, not metric depth. Trajectories are
evaluated after Sim(3), same protocol as baseline_sota.py.

  python scripts/eval_meshrts.py --list lists/simtest92.txt --out results/sota --tag meshrts_simtest --max-frames 64
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.metrics import evaluate_trajectory, load_pose_txt, rpe
from endosim.eval.protocol import list_color_frames, motion_stats_of_indices, select_frame_indices


def _finite_mean(xs):
    a = np.asarray(xs, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def _finite_median(xs):
    a = np.asarray(xs, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def eval_with_scale_correction(est, gt):
    res = evaluate_trajectory(est, gt)
    s = res["ate_sim3"]["scale"]
    if not np.isfinite(s) or s <= 0:
        s = 1.0
        res["ate_sim3"]["scale"] = 1.0
    est_sc = est.copy()
    est_sc[:, :3, 3] *= s
    for g in (1, 5, 10):
        if f"rpe_{g}" in res:
            res[f"rpe_{g}"] = rpe(est_sc, gt, g)
    return res


def _resize_max(img, max_side):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img, 1.0
    s = max_side / float(m)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), s


def height_field(bgr, alpha=32.0, s_z=40.0):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bf = cv2.bilateralFilter(gray, 5, 40, 5)
    gx = cv2.Sobel(bf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(bf, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    tau = float(np.percentile(g, alpha))
    mask = (g > tau).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    trunc = mask.astype(np.float32) * g
    h = trunc / 255.0 * s_z
    return h, mask


def sample_uvh(h, mask, max_pts=2500, stride=2):
    ys, xs = np.where(mask[::stride, ::stride] > 0)
    if len(xs) == 0:
        return np.zeros((0, 3), np.float32)
    xs = xs * stride
    ys = ys * stride
    if len(xs) > max_pts:
        sel = np.linspace(0, len(xs) - 1, max_pts).astype(int)
        xs, ys = xs[sel], ys[sel]
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32), h[ys, xs]], axis=1)
    return pts


def kabsch(src, dst):
    if len(src) < 3:
        return np.eye(4, dtype=np.float64)
    c_s = src.mean(0)
    c_d = dst.mean(0)
    x = src - c_s
    y = dst - c_d
    u, _, vt = np.linalg.svd(x.T @ y)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = c_d - r @ c_s
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r
    T[:3, 3] = t
    return T


def _nn_idx(pred, dst):
    try:
        from scipy.spatial import cKDTree
        d, j = cKDTree(dst).query(pred, k=1, workers=-1)
        return np.asarray(d), np.asarray(j)
    except Exception:
        d2 = ((pred[:, None, :] - dst[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        return np.sqrt(d2[np.arange(len(pred)), j]), j


def icp_refine(src, dst, T0, iters=8):
    T = T0.copy()
    if len(src) < 8 or len(dst) < 8:
        return T
    cur = src.copy()
    for _ in range(iters):
        pred = (T[:3, :3] @ cur.T).T + T[:3, 3]
        d, j = _nn_idx(pred, dst)
        keep = d < np.percentile(d, 80)
        if keep.sum() < 8:
            break
        T = kabsch(cur[keep], dst[j[keep]])
    return T


def _rot_angle_deg(R):
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _tukey_weights(res, c=6.0):
    u = np.asarray(res, np.float64).reshape(-1) / max(c, 1e-6)
    w = np.zeros_like(u)
    m = np.abs(u) < 1.0
    w[m] = (1.0 - u[m] ** 2) ** 2
    return w


def kabsch_weighted(src, dst, w=None):
    if len(src) < 3:
        return np.eye(4, dtype=np.float64)
    if w is None:
        w = np.ones(len(src), np.float64)
    w = np.clip(np.asarray(w, np.float64).reshape(-1), 0.0, None)
    sw = float(w.sum())
    if sw <= 1e-12:
        return kabsch(src, dst)
    w = w / sw
    c_s = (src * w[:, None]).sum(0)
    c_d = (dst * w[:, None]).sum(0)
    x = src - c_s
    y = dst - c_d
    S = (x * w[:, None]).T @ y
    u, _, vt = np.linalg.svd(S)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = c_d - r @ c_s
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r
    T[:3, 3] = t
    return T


def irls_refine(src, dst, T0, iters=8, c=6.0):
    T = T0.copy()
    if len(src) < 8:
        return T
    for _ in range(iters):
        pred = (T[:3, :3] @ src.T).T + T[:3, 3]
        r = np.linalg.norm(pred[:, :2] - dst[:, :2], axis=1)
        w = _tukey_weights(r, c)
        if float(w.sum()) <= 1e-9:
            break
        T = kabsch_weighted(src, dst, w)
    return T


def uvh_to_cam(uvh, K):
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    u, v, h = uvh[:, 0], uvh[:, 1], np.maximum(uvh[:, 2], 1e-3)
    x = (u - cx) / fx * h
    y = (v - cy) / fy * h
    return np.stack([x, y, h], axis=1)


def flow_motion_scores(flow, cx, cy, mask=None, step=4):
    H, W = flow.shape[:2]
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    vx = flow[::step, ::step, 0]
    vy = flow[::step, ::step, 1]
    if mask is not None:
        m = mask[::step, ::step] > 0
        xs, ys, vx, vy = xs[m], ys[m], vx[m], vy[m]
    else:
        xs, ys, vx, vy = xs.ravel(), ys.ravel(), vx.ravel(), vy.ravel()
    if len(xs) < 20:
        return 0.0, 0.0, 0.0
    rx, ry = xs.astype(np.float64) - cx, ys.astype(np.float64) - cy
    r = np.hypot(rx, ry) + 1e-6
    radial = (vx * rx + vy * ry) / r
    tang = (-vx * ry + vy * rx) / r
    mag = np.hypot(vx, vy)
    return (float(np.median(np.abs(radial))),
            float(np.median(np.abs(tang))),
            float(np.median(mag)))


def _orb_match(img0, img1, n_feat=2000):
    orb = cv2.ORB_create(n_feat)
    k0, d0 = orb.detectAndCompute(img0, None)
    k1, d1 = orb.detectAndCompute(img1, None)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = bf.match(d0, d1)
    if len(m) < 8:
        return None, None
    m = sorted(m, key=lambda x: x.distance)[:400]
    p0 = np.float32([k0[x.queryIdx].pt for x in m])
    p1 = np.float32([k1[x.trainIdx].pt for x in m])
    return p0, p1


def pairwise_essential(img0, img1, K, return_inliers=False):
    g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) if img0.ndim == 3 else img0
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    p0, p1 = _orb_match(g0, g1)
    if p0 is None:
        return (None, 0) if return_inliers else None
    E, mask = cv2.findEssentialMat(
        p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return (None, 0) if return_inliers else None
    _, R, t, _ = cv2.recoverPose(E, p0, p1, K, mask=mask)
    # OpenCV returns the previous-camera -> current-camera transform.
    # Mesh-RTS composes camera-to-world poses and therefore requires the
    # opposite edge, current-camera -> previous-camera.
    T_cur_from_prev = np.eye(4, dtype=np.float64)
    T_cur_from_prev[:3, :3] = R
    T_cur_from_prev[:3, 3] = t.reshape(3)
    T = np.linalg.inv(T_cur_from_prev)
    n_inl = int(mask.sum()) if mask is not None else 0
    return (T, n_inl) if return_inliers else T


def ransac_kabsch(src, dst, n_iter=256, thresh=6.0, min_inliers=12):
    n = len(src)
    if n < 6:
        return kabsch(src, dst), np.ones(n, bool)
    rng = np.random.default_rng(0)
    best_inl = None
    best_n = -1
    sample = 6 if n >= 6 else 3
    for _ in range(n_iter):
        idx = rng.choice(n, size=sample, replace=False)
        T = kabsch(src[idx], dst[idx])
        pred = (T[:3, :3] @ src.T).T + T[:3, 3]
        err = np.linalg.norm(pred[:, :2] - dst[:, :2], axis=1)
        inl = err < thresh
        c = int(inl.sum())
        if c > best_n:
            best_n = c
            best_inl = inl
    if best_inl is None or best_n < min_inliers:
        T = kabsch(src, dst)
        return T, np.ones(n, bool)
    return kabsch(src[best_inl], dst[best_inl]), best_inl


def pairwise_T(img0, img1, max_side=384):
    im0, _ = _resize_max(img0, max_side)
    im1, _ = _resize_max(img1, max_side)
    h0, m0 = height_field(im0)
    h1, m1 = height_field(im1)
    g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    ys, xs = np.where(m0 > 0)
    if len(xs) < 30:
        return np.eye(4)
    if len(xs) > 4000:
        sel = np.linspace(0, len(xs) - 1, 4000).astype(int)
        xs, ys = xs[sel], ys[sel]
    dx = flow[ys, xs, 0]
    dy = flow[ys, xs, 1]
    x1 = xs.astype(np.float32) + dx
    y1 = ys.astype(np.float32) + dy
    H, W = h1.shape
    ok = (x1 >= 0) & (x1 < W - 1) & (y1 >= 0) & (y1 < H - 1)
    xs, ys, x1, y1 = xs[ok], ys[ok], x1[ok], y1[ok]
    if len(xs) < 20:
        return np.eye(4)
    xi = np.clip(np.round(x1).astype(int), 0, W - 1)
    yi = np.clip(np.round(y1).astype(int), 0, H - 1)
    src = np.stack([x1, y1, h1[yi, xi]], axis=1).astype(np.float64)
    dst = np.stack([xs.astype(np.float64), ys.astype(np.float64), h0[ys, xs]], axis=1)
    T, _ = ransac_kabsch(src, dst)
    pts1 = sample_uvh(h1, m1)
    pts0 = sample_uvh(h0, m0)
    if len(pts1) >= 30 and len(pts0) >= 30:
        try:
            T = icp_refine(pts1.astype(np.float64), pts0.astype(np.float64), T)
        except Exception:
            pass
    return T


def dense_flow(g0, g1):
    try:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return dis.calc(g0, g1, None)
    except Exception:
        return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)


def flow_fb_ok(flow_fw, flow_bw, xs, ys, thresh=1.5):
    H, W = flow_fw.shape[:2]
    x1 = xs.astype(np.float32) + flow_fw[ys, xs, 0]
    y1 = ys.astype(np.float32) + flow_fw[ys, xs, 1]
    xi = np.clip(np.round(x1).astype(int), 0, W - 1)
    yi = np.clip(np.round(y1).astype(int), 0, H - 1)
    x0 = x1 + flow_bw[yi, xi, 0]
    y0 = y1 + flow_bw[yi, xi, 1]
    err = np.hypot(x0 - xs.astype(np.float32), y0 - ys.astype(np.float32))
    return err < thresh


def mesh_pnp(src_cam, dst_uv, K):
    if len(src_cam) < 8:
        return None
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        src_cam.astype(np.float32), dst_uv.astype(np.float32),
        K.astype(np.float64), None, flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=3.0, iterationsCount=200, confidence=0.99)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # solvePnP maps the supplied object points to the observing camera.
    # Here object points are expressed in the current-camera pseudo-3D frame
    # and image points belong to the previous frame, so the returned transform
    # is already current-camera -> previous-camera: the c2w compositional edge.
    T_prev_from_cur = np.eye(4, dtype=np.float64)
    T_prev_from_cur[:3, :3] = R
    T_prev_from_cur[:3, 3] = tvec.reshape(3)
    return T_prev_from_cur


def pairwise_T_v3(img0, img1, K, max_side=384):
    """Single mesh path: DIS flow + cam UVH + IRLS/PnP. Rotation → t=0, no 8-point chain."""
    im0, s = _resize_max(img0, max_side)
    im1, _ = _resize_max(img1, max_side)
    Ks = np.asarray(K, np.float64).copy()
    Ks[0, :] *= s
    Ks[1, :] *= s
    cx, cy = float(Ks[0, 2]), float(Ks[1, 2])
    h0, m0 = height_field(im0, alpha=68.0, s_z=40.0)
    h1, m1 = height_field(im1, alpha=68.0, s_z=40.0)
    g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    flow = dense_flow(g0, g1)
    flow_bw = dense_flow(g1, g0)
    rad, tang, mag = flow_motion_scores(flow, cx, cy, mask=m0)
    rot_like = mag > 0.8 and rad < 0.35 * mag

    ys, xs = np.where(m0 > 0)
    T_mesh = np.eye(4)
    inl_ratio = 0.0
    src_cam = dst_cam = None
    dst_uv = None
    if len(xs) >= 30:
        if len(xs) > 4000:
            sel = np.linspace(0, len(xs) - 1, 4000).astype(int)
            xs, ys = xs[sel], ys[sel]
        keep = flow_fb_ok(flow, flow_bw, xs, ys)
        if int(keep.sum()) >= 20:
            xs, ys = xs[keep], ys[keep]
        dx = flow[ys, xs, 0]
        dy = flow[ys, xs, 1]
        x1 = xs.astype(np.float32) + dx
        y1 = ys.astype(np.float32) + dy
        H, W = h1.shape
        ok = (x1 >= 0) & (x1 < W - 1) & (y1 >= 0) & (y1 < H - 1)
        xs, ys, x1, y1 = xs[ok], ys[ok], x1[ok], y1[ok]
        if len(xs) >= 20:
            xi = np.clip(np.round(x1).astype(int), 0, W - 1)
            yi = np.clip(np.round(y1).astype(int), 0, H - 1)
            src_uvh = np.stack([x1, y1, h1[yi, xi]], axis=1).astype(np.float64)
            dst_uvh = np.stack([xs.astype(np.float64), ys.astype(np.float64),
                               h0[ys, xs]], axis=1)
            src_cam = uvh_to_cam(src_uvh, Ks)
            dst_cam = uvh_to_cam(dst_uvh, Ks)
            dst_uv = dst_uvh[:, :2]
            T_mesh, inl = ransac_kabsch(src_cam, dst_cam, n_iter=256, thresh=6.0)
            T_mesh = irls_refine(src_cam, dst_cam, T_mesh)
            if inl is not None and int(inl.sum()) >= 30:
                T_mesh = irls_refine(src_cam[inl], dst_cam[inl], T_mesh)
                inl_ratio = float(inl.mean())
            T_pnp = mesh_pnp(src_cam, dst_uv, Ks)
            if T_pnp is not None and inl_ratio < 0.35:
                T_mesh = T_pnp

    T_e, n_inl_e = pairwise_essential(img0, img1, K, return_inliers=True)
    if rot_like:
        T = np.eye(4, dtype=np.float64)
        if T_e is not None:
            T[:3, :3] = T_e[:3, :3]
        elif _rot_angle_deg(T_mesh[:3, :3]) > 0.15:
            T[:3, :3] = T_mesh[:3, :3]
        return T, True
    # translation pair: trust essential when it is well supported, else UVH mesh
    if T_e is not None and n_inl_e >= 8:
        return T_e, False
    return T_mesh, False


def pairwise_T_v2(img0, img1, K, max_side=384):
    """Camera-space UVH + IRLS. Rotation-dominated flow → mesh R (or essential if degenerate), t=0."""
    im0, s = _resize_max(img0, max_side)
    im1, _ = _resize_max(img1, max_side)
    Ks = np.asarray(K, np.float64).copy()
    Ks[0, :] *= s
    Ks[1, :] *= s
    cx, cy = float(Ks[0, 2]), float(Ks[1, 2])
    h0, m0 = height_field(im0, alpha=68.0, s_z=40.0)
    h1, m1 = height_field(im1, alpha=68.0, s_z=40.0)
    g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    rad, tang, mag = flow_motion_scores(flow, cx, cy, mask=m0)
    rot_like = mag > 0.8 and rad < 0.35 * mag

    ys, xs = np.where(m0 > 0)
    T_mesh = np.eye(4)
    if len(xs) >= 30:
        if len(xs) > 4000:
            sel = np.linspace(0, len(xs) - 1, 4000).astype(int)
            xs, ys = xs[sel], ys[sel]
        dx = flow[ys, xs, 0]
        dy = flow[ys, xs, 1]
        x1 = xs.astype(np.float32) + dx
        y1 = ys.astype(np.float32) + dy
        H, W = h1.shape
        ok = (x1 >= 0) & (x1 < W - 1) & (y1 >= 0) & (y1 < H - 1)
        xs, ys, x1, y1 = xs[ok], ys[ok], x1[ok], y1[ok]
        if len(xs) >= 20:
            xi = np.clip(np.round(x1).astype(int), 0, W - 1)
            yi = np.clip(np.round(y1).astype(int), 0, H - 1)
            src_uvh = np.stack([x1, y1, h1[yi, xi]], axis=1).astype(np.float64)
            dst_uvh = np.stack([xs.astype(np.float64), ys.astype(np.float64),
                               h0[ys, xs]], axis=1)
            src = uvh_to_cam(src_uvh, Ks)
            dst = uvh_to_cam(dst_uvh, Ks)
            T_mesh, inl = ransac_kabsch(src, dst, n_iter=256, thresh=6.0)
            T_mesh = irls_refine(src, dst, T_mesh)
            if inl is not None and int(inl.sum()) >= 30:
                T_mesh = irls_refine(src[inl], dst[inl], T_mesh)

    T_e = pairwise_essential(im0, im1, Ks)
    if rot_like:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = T_mesh[:3, :3]
        if _rot_angle_deg(T[:3, :3]) < 0.15 and T_e is not None:
            T[:3, :3] = T_e[:3, :3]
        return T, True
    return T_mesh, False


def run_meshrts(frame_paths, max_side=384, K=None, variant="v1"):
    n = len(frame_paths)
    poses = [np.eye(4)]
    prev = cv2.imread(frame_paths[0])
    if prev is None:
        raise RuntimeError("cannot read " + frame_paths[0])
    n_rot = 0
    for i in range(1, n):
        cur = cv2.imread(frame_paths[i])
        if cur is None:
            poses.append(poses[-1].copy())
            continue
        if variant == "v3" and K is not None:
            T, rot_like = pairwise_T_v3(prev, cur, K, max_side=max_side)
            n_rot += int(rot_like)
        elif variant == "v2" and K is not None:
            T, rot_like = pairwise_T_v2(prev, cur, K, max_side=max_side)
            n_rot += int(rot_like)
        else:
            T = pairwise_T(prev, cur, max_side=max_side)
        poses.append(poses[-1] @ T)
        prev = cur
    info = {"n_rot_pairs": n_rot, "n_pairs": n - 1}
    return np.stack(poses), info


def load_K(seq_dir, img):
    p = os.path.join(seq_dir, "intrinsics.json")
    if not os.path.exists(p):
        return None
    intr = json.load(open(p, encoding="utf-8"))
    h, w = img.shape[:2]
    sx, sy = w / float(intr["width"]), h / float(intr["height"])
    return np.array([[intr["fx"] * sx, 0, intr["cx"] * sx],
                     [0, intr["fy"] * sy, intr["cy"] * sy],
                     [0, 0, 1.0]], np.float64)


def run_eight_on(frame_paths, K):
    from endosim.geometry.se3 import rot_trans
    poses = [np.eye(4)]
    gray0 = cv2.cvtColor(cv2.imread(frame_paths[0]), cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frame_paths)):
        gray1 = cv2.cvtColor(cv2.imread(frame_paths[i]), cv2.COLOR_BGR2GRAY)
        T = np.eye(4)
        p0, p1 = _orb_match(gray0, gray1)
        if p0 is not None:
            E, mask = cv2.findEssentialMat(
                p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, p0, p1, K, mask=mask)
                T = np.linalg.inv(rot_trans(R, t.reshape(3)))
        poses.append(poses[-1] @ T)
        gray0 = gray1
    return np.stack(poses)


def stratified_summary(records):
    buckets = {"低参照(<0.3)": [], "中参照(0.3-0.7)": [], "高参照(>0.7)": []}
    for r in records:
        rf = r.get("reference_fraction")
        if rf is None:
            continue
        key = ("低参照(<0.3)" if rf < 0.3 else
               "中参照(0.3-0.7)" if rf < 0.7 else "高参照(>0.7)")
        buckets[key].append(r)
    out = []
    for key, items in buckets.items():
        if not items:
            continue
        out.append({
            "bucket": key, "n": len(items),
            "ate_se3_mean": _finite_mean([r["ate_se3"]["rmse"] for r in items]),
            "ate_sim3_mean": _finite_mean([r["ate_sim3"]["rmse"] for r in items]),
            "rpe1_t_mean": _finite_mean([r["rpe_1"]["trans_mm_mean"] for r in items]),
            "rpe1_r_mean": _finite_mean([r["rpe_1"]["rot_deg_mean"] for r in items]),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="lists/simtest92.txt")
    ap.add_argument("--out", default="results/sota")
    ap.add_argument("--tag", default="meshrts_simtest")
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--max-side", type=int, default=384)
    ap.add_argument("--variant", default="v1", choices=["v1", "v2", "v3"],
                    help="v2: cam-UVH+IRLS+rot-gate + sequence 8-point; "
                         "v3: same pairwise, no sequence 8-point (single mesh path)")
    args = ap.parse_args()

    seqs = [ln.strip() for ln in open(args.list, encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    seqs = [s for s in seqs if os.path.isdir(s)]
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Mesh-RTS {args.variant}] {len(seqs)} sequences max_frames={args.max_frames}",
          flush=True)

    records, t0 = [], time.time()
    for i, seq_dir in enumerate(seqs):
        sid = os.path.basename(seq_dir.rstrip("/\\"))
        t_seq = time.time()
        try:
            frames = list_color_frames(seq_dir)
            gt_all = load_pose_txt(os.path.join(seq_dir, "pose_c2w.txt"))
            n = min(len(frames), len(gt_all))
            frames, gt_all = frames[:n], gt_all[:n]
            idx = select_frame_indices(gt_all, "uniform", max_frames=args.max_frames)
            frame_paths = [frames[k] for k in idx]
            gt = gt_all[idx]
            im0 = cv2.imread(frame_paths[0])
            K = load_K(seq_dir, im0)
            est, info = run_meshrts(frame_paths, max_side=args.max_side,
                                    K=K, variant=args.variant)
            if args.variant == "v2" and K is not None and info["n_pairs"] > 0:
                frac_rot = info["n_rot_pairs"] / float(info["n_pairs"])
                if frac_rot >= 0.40:
                    est = run_eight_on(frame_paths, K)
                    info["fallback"] = "eight"
            # v3: keep the mesh chain (rotation pairs already have t=0)
            res = eval_with_scale_correction(est, gt)
            res["mesh_info"] = info
            res["protocol_hop"] = motion_stats_of_indices(gt_all, idx)
            meta_p = os.path.join(seq_dir, "meta.json")
            if os.path.exists(meta_p):
                meta = json.load(open(meta_p, encoding="utf-8"))
                refs = [r for r in meta.get("reference_fraction", []) if r is not None]
                res["reference_fraction"] = float(np.mean(refs)) if refs else None
                res["motion_type"] = meta.get("motion_type")
                res["scene_kind"] = meta.get("scene_kind")
            res["seq_id"] = sid
            res["n_frames_used"] = int(len(idx))
            res["time_sec"] = round(time.time() - t_seq, 2)
            records.append(res)
            np.savetxt(os.path.join(out_dir, f"{sid}_est_c2w.txt"),
                       est.reshape(len(est), 16), fmt="%.6f")
            print(f"[{i+1}/{len(seqs)}] {sid}: ATE(Sim3)={res['ate_sim3']['rmse']:.3f}mm "
                  f"({res['time_sec']}s)", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(seqs)}] {sid}: FAILED {e}", flush=True)
            records.append({"seq_id": sid, "error": str(e)})

    ok = [r for r in records if "error" not in r]
    summary = {
        "method": f"meshrts_{args.variant}", "n_seq": len(ok),
        "n_failed": len(records) - len(ok),
        "protocol": {"max_frames": args.max_frames, "name": "uniform",
                     "max_side": args.max_side, "variant": args.variant},
        "ate_se3_rmse_mean": _finite_mean([r["ate_se3"]["rmse"] for r in ok]) if ok else None,
        "ate_se3_rmse_median": _finite_median([r["ate_se3"]["rmse"] for r in ok]) if ok else None,
        "ate_sim3_rmse_mean": _finite_mean([r["ate_sim3"]["rmse"] for r in ok]) if ok else None,
        "ate_sim3_rmse_median": _finite_median([r["ate_sim3"]["rmse"] for r in ok]) if ok else None,
        "rpe1_trans_mean": _finite_mean([r["rpe_1"]["trans_mm_mean"] for r in ok]) if ok else None,
        "rpe1_rot_mean": _finite_mean([r["rpe_1"]["rot_deg_mean"] for r in ok]) if ok else None,
        "total_time_sec": round(time.time() - t0, 1),
    }
    if ok:
        summary["stratified"] = stratified_summary(ok)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=1)
    print(f"\n=== Mesh-RTS ATE(Sim3)={summary['ate_sim3_rmse_mean']} "
          f"median={summary['ate_sim3_rmse_median']} ===", flush=True)
    for b in summary.get("stratified", []):
        print(f"  [{b['bucket']}] n={b['n']} ATE={b['ate_sim3_mean']:.3f}mm", flush=True)


if __name__ == "__main__":
    main()
