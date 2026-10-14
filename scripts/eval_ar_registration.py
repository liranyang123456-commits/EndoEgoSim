"""Augmented-reality registration test: does removing the camera ego-motion first make registration to moving tissue
more accurate and more stable?

Sequences are EndoEgoSim-Abs test sequences (not used in training) in which the camera travels and the tissue
breathes. A rigid patch of points on deforming tissue is registered from frame t to frame t+s. The registration
methods differ only in the prior they start from: none, a whole-frame motion model that mixes camera and tissue
motion, a classical ego-motion estimate (ORB + essential matrix, given its true translation magnitude), EgoLite, or
the true camera motion (reference). The error is the distance of the registered patch from its true projection.

  python scripts/eval_ar_registration.py
  python scripts/eval_ar_registration.py --root sim_data_abs/test --ckpt results/egolite/r_v2/last.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from endosim.lite.data import load_frames, scaled_K  # noqa: E402
from endosim.lite.egolite import build_from_args  # noqa: E402

DEPTH_UNIT_MM = 50.0     # EgoLite translation unit (fixed depth unit of the reported checkpoint)
PATCH = 7                # template half-size of the local registration
MIN_CORR = 0.35
LOCAL = ("local_none", "local_homography", "local_orb", "local_egolite", "local_true")
GLOBAL = ("lk_pnp", "flow_pnp", "icp", "homography")


# ----------------------------------------------------------------------------------------- data
def select_sequences(root, min_path=40.0, min_amp=4.0):
    rows = []
    for name in sorted(os.listdir(root)):
        meta = os.path.join(root, name, "meta.json")
        if not os.path.isfile(meta):
            continue
        m = json.load(open(meta, encoding="utf-8"))
        path = (m.get("motion_stats") or {}).get("path_length_mm") or 0.0
        amp = (m.get("deformation") or {}).get("amp_mm") or 0.0
        if path >= min_path and amp >= min_amp:
            rows.append((float(path), float(amp), os.path.join(root, name)))
    rows.sort(reverse=True)
    return rows


def project_w(Xw, T, K):
    R, t = T[:3, :3], T[:3, 3]
    Xc = (R.T @ (Xw - t).T).T
    z = Xc[:, 2]
    return np.c_[K[0, 0] * Xc[:, 0] / z + K[0, 2], K[1, 1] * Xc[:, 1] / z + K[1, 2]], z


def project_c(K, Xc):
    z = Xc[:, 2]
    return np.c_[K[0, 0] * Xc[:, 0] / z + K[0, 2], K[1, 1] * Xc[:, 1] / z + K[1, 2]], z


def load_clip(seq):
    """Camera poses and the world trajectory of a patch on deforming tissue (from the simulator's world displacement)."""
    intr = json.load(open(os.path.join(seq, "intrinsics.json"), encoding="utf-8"))
    K = np.array(intr["K"], float)
    pose = np.loadtxt(os.path.join(seq, "pose_c2w.txt")).reshape(-1, 4, 4)
    depth0 = cv2.imread(os.path.join(seq, "depth", "000000.png"), cv2.IMREAD_UNCHANGED)
    mask0 = cv2.imread(os.path.join(seq, "motion_mask", "000000.png"), cv2.IMREAD_UNCHANGED)
    if depth0 is None or mask0 is None:
        raise ValueError("missing depth or motion mask")
    ys, xs = np.where(mask0 == 2)
    if len(ys) < 20:
        raise ValueError("too little deforming tissue")
    cy, cx = int(np.median(ys)), int(np.median(xs))
    h, w = mask0.shape
    pix = [(cx + dx, cy + dy) for dy in range(-28, 32, 8) for dx in range(-28, 32, 8)
           if 0 <= cx + dx < w and 0 <= cy + dy < h and mask0[cy + dy, cx + dx] == 2 and depth0[cy + dy, cx + dx] > 0]
    if len(pix) < 10:
        raise ValueError("patch too small")
    pix = np.array(pix, float)
    z = depth0[pix[:, 1].astype(int), pix[:, 0].astype(int)].astype(float)
    X = np.c_[(pix[:, 0] - K[0, 2]) * z / K[0, 0], (pix[:, 1] - K[1, 2]) * z / K[1, 1], z]
    traj = [X.copy()]
    for t in range(len(pose) - 1):
        path = os.path.join(seq, "abs_motion", f"{t:06d}.npz")
        if not os.path.isfile(path):
            break
        d = np.load(path)["world_disp"].astype(np.float64)
        uv, zz = project_w(traj[-1], pose[t], K)
        step = np.zeros_like(traj[-1])
        for i, (u, v, zc) in enumerate(zip(uv[:, 0], uv[:, 1], zz)):
            ui, vi = u / 4.0, v / 4.0
            if zc > 1 and 1 <= ui < d.shape[1] - 2 and 1 <= vi < d.shape[0] - 2:
                step[i] = d[int(round(vi)), int(round(ui))]
        traj.append(traj[-1] + step)
    return K, pose[:len(traj)], np.stack(traj)


def cam_static(Ta, Tb):
    """X_b = R X_a + t for a point that is fixed in the world (camera-to-world poses Ta, Tb)."""
    Ra, ta, Rb, tb = Ta[:3, :3], Ta[:3, 3], Tb[:3, :3], Tb[:3, 3]
    return Rb.T @ Ra, Rb.T @ (ta - tb)


def kabsch(Xm, Xc):
    Am, Ac = Xm - Xm.mean(0), Xc - Xc.mean(0)
    U, _, Vt = np.linalg.svd(Ac.T @ Am)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    return R, Xc.mean(0) - R @ Xm.mean(0)


def px(a, b):
    return float(np.median(np.linalg.norm(a - b, axis=1)))


# ----------------------------------------------------------------------------------------- registration
def local_match(prev, nxt, src, pred, win):
    h, w = nxt.shape
    out, ok = np.zeros_like(pred), np.zeros(len(pred), bool)
    for i, ((x0, y0), (xp, yp)) in enumerate(zip(src, pred)):
        x0i, y0i, x1, y1 = int(round(x0)), int(round(y0)), int(round(xp)), int(round(yp))
        if not (PATCH <= x0i < w - PATCH and PATCH <= y0i < h - PATCH):
            continue
        if not (win + PATCH <= x1 < w - win - PATCH and win + PATCH <= y1 < h - win - PATCH):
            continue
        templ = prev[y0i - PATCH:y0i + PATCH + 1, x0i - PATCH:x0i + PATCH + 1]
        search = nxt[y1 - win - PATCH:y1 + win + PATCH + 1, x1 - win - PATCH:x1 + win + PATCH + 1]
        res = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
        _, mv, _, loc = cv2.minMaxLoc(res)
        if mv < MIN_CORR:
            continue
        out[i] = (x1 - win + loc[0], y1 - win + loc[1])
        ok[i] = True
    return out, ok


def solve(K, X, uv, ok, guess):
    """Rigid model pose from 2D-3D matches, starting from the prior pose."""
    if int(ok.sum()) < 12:
        return None
    rvec, _ = cv2.Rodrigues(guess[0])
    tvec = np.asarray(guess[1], float).reshape(3, 1)
    good, rvec, tvec, inl = cv2.solvePnPRansac(
        X[ok].astype(np.float32), uv[ok].astype(np.float32), K, None, rvec, tvec, True,
        iterationsCount=80, reprojectionError=3.0, confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
    if not good or inl is None or len(inl) < 12:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


def register_local(g0, g1, src, pred, X, K, guess, vis, win):
    """Search each patch point within `win` px of its predicted position, then fit a rigid pose."""
    uv, ok = local_match(g0, g1, src, pred, win)
    sol = solve(K, X, uv, ok & vis, guess)
    if sol is None:
        return pred
    return project_c(K, (sol[0] @ X.T).T + sol[1])[0]


def homography(g0, g1):
    p0 = cv2.goodFeaturesToTrack(g0, 400, 0.01, 12)
    if p0 is None:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=3)
    ok = st.reshape(-1) == 1
    if ok.sum() < 20:
        return None
    H, _ = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 3.0)
    return H


def orb_motion(g0, g1, K, magnitude):
    """Classical monocular ego-motion: ORB matches, essential matrix, cheirality check. The unit translation is
    given its true magnitude, which favours this baseline."""
    orb = cv2.ORB_create(1500)
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(d0) < 20 or len(d1) < 20:
        return None
    m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d0, d1), key=lambda x: x.distance)[:300]
    if len(m) < 20:
        return None
    p0 = np.float32([k0[x.queryIdx].pt for x in m])
    p1 = np.float32([k1[x.trainIdx].pt for x in m])
    E, _ = cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, 1.0)
    if E is None or E.shape != (3, 3):
        return None
    _, R, t, _ = cv2.recoverPose(E, p0, p1, K)
    return R, t.ravel() * magnitude


def lk_pnp(g0, g1, src, X, K, guess, vis):
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, src.reshape(-1, 1, 2).astype(np.float32), None,
                                        winSize=(21, 21), maxLevel=3)
    if p1 is None:
        return src
    sol = solve(K, X, p1.reshape(-1, 2), (st.reshape(-1) == 1) & vis, guess)
    return src if sol is None else project_c(K, (sol[0] @ X.T).T + sol[1])[0]


def flow_pnp(g0, g1, src, X, K, guess, vis):
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    h, w = g0.shape
    p1, ok = src.copy(), vis.copy()
    for i, (u, v) in enumerate(src):
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            p1[i] = src[i] + flow[vi, ui]
        else:
            ok[i] = False
    sol = solve(K, X, p1, ok, guess)
    return src if sol is None else project_c(K, (sol[0] @ X.T).T + sol[1])[0]


def icp(src_cam, depth_path, K, X, radius=15.0, iters=5, step=8):
    """Rigid ICP of the patch onto the next depth map (the simulator's depth, which favours this baseline)."""
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    vs, us = np.mgrid[0:d.shape[0]:step, 0:d.shape[1]:step]
    z = d[vs, us].astype(np.float64)
    keep = z > 0
    cloud = np.c_[(us[keep] - K[0, 2]) * z[keep] / K[0, 0], (vs[keep] - K[1, 2]) * z[keep] / K[1, 1], z[keep]]
    if cloud.size == 0:
        return None
    Y = src_cam.copy()
    for _ in range(iters):
        d2 = ((Y[:, None, :] - cloud[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        near = np.sqrt(d2[np.arange(len(Y)), j]) < radius
        if near.sum() < 12:
            return None
        Rk, tk = kabsch(Y[near], cloud[j[near]])
        Y = (Rk @ Y.T).T + tk
    R, t = kabsch(X, Y)
    return project_c(K, (R @ X.T).T + t)[0]


# ----------------------------------------------------------------------------------------- protocol
def choose_stride(K, pose, traj, win):
    """Frame gap at which the camera-induced image motion leaves the search window and the tissue motion stays inside."""
    stats = []
    for s in (2, 3, 4, 5, 6, 8):
        cam, tis = [], []
        for t in range(0, len(pose) - s, s):
            uva, za = project_w(traj[t], pose[t], K)
            uvb, zb = project_w(traj[t + s], pose[t + s], K)
            Re, te = cam_static(pose[t], pose[t + s])
            Xc = (pose[t][:3, :3].T @ (traj[t] - pose[t][:3, 3]).T).T
            uvp, _ = project_c(K, (Re @ Xc.T).T + te)
            vis = (za > 1) & (zb > 1)
            if vis.mean() < 0.7:
                continue
            cam.append(px(uva[vis], uvp[vis]))
            tis.append(px(uvp[vis], uvb[vis]))
        if cam:
            stats.append((s, float(np.median(cam)), float(np.median(tis))))
    for s, cam, tis in stats:
        if cam > win + 6 and tis < win - 2:
            return s, cam, tis, False
    for s, cam, tis in stats:
        if cam > win and tis < win - 2:
            return s, cam, tis, True
    return None


class EgoLite:
    def __init__(self, ckpt, K, w, h):
        ck = torch.load(ckpt, map_location="cpu")
        self.model = build_from_args(dict(ck["args"]), pretrained=False)
        self.model.load_state_dict(ck["model"])
        self.model = self.model.cuda().eval()
        self.set_K(K, w, h)

    def set_K(self, K, w, h):
        Kd = {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
              "width": w, "height": h}
        self.Ks = torch.from_numpy(scaled_K(Kd, w, h)).cuda()

    @torch.no_grad()
    def motion(self, path_a, path_b):
        imgs, phs, _ = load_frames([path_a, path_b])
        img = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
        ph = torch.from_numpy(phs)[:, None].cuda()
        o = self.model(img[:1], img[1:], self.Ks[None], ph[:1], ph[1:])
        return o["R"][0].cpu().numpy(), o["t"][0].cpu().numpy() * DEPTH_UNIT_MM


def evaluate_sequence(seq, net, windows, main_win):
    K, pose, traj = load_clip(seq)
    Xm = traj[0]
    picked = choose_stride(K, pose, traj, main_win)
    if picked is None:
        raise ValueError("no frame gap separates camera motion from tissue motion")
    stride, cam_px, tis_px, relaxed = picked
    first = cv2.imread(os.path.join(seq, "color", "000000.png"))
    net.set_K(K, first.shape[1], first.shape[0])
    err = {m: [] for m in GLOBAL}
    err.update({f"{m}@{w}": [] for m in LOCAL for w in windows})
    ego_cam, examples = [], []
    for t in range(0, len(pose) - stride, stride):
        pa = os.path.join(seq, "color", f"{t:06d}.png")
        pb = os.path.join(seq, "color", f"{t + stride:06d}.png")
        g0 = cv2.imread(pa, cv2.IMREAD_GRAYSCALE)
        im1 = cv2.imread(pb)
        g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        Xc = (pose[t][:3, :3].T @ (traj[t] - pose[t][:3, 3]).T).T
        R0, t0 = kabsch(Xm, Xc)
        src, za = project_w(traj[t], pose[t], K)
        truth, zb = project_w(traj[t + stride], pose[t + stride], K)
        vis = (za > 1) & (zb > 1)
        if vis.mean() < 0.7:
            continue
        Re, te = cam_static(pose[t], pose[t + stride])
        Rn, tn = net.motion(pa, pb)
        H = homography(g0, g1)
        orb = orb_motion(g0, g1, K, float(np.linalg.norm(te)))
        # priors: predicted image position of the patch and the matching pose guess for the rigid fit
        priors = {"local_none": (src, (R0, t0))}
        uvH = src if H is None else cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
        try:
            rv0, _ = cv2.Rodrigues(R0)
            okH, rvH, tvH = cv2.solvePnP(Xm.astype(np.float32), uvH.astype(np.float32), K, None,
                                         rv0, t0.reshape(3, 1).astype(np.float64), True, cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            okH = False
        guessH = (cv2.Rodrigues(rvH)[0], tvH.ravel()) if okH else (R0, t0)
        priors["local_homography"] = (uvH, guessH)
        for name, motion in (("local_orb", orb), ("local_egolite", (Rn, tn)), ("local_true", (Re, te))):
            if motion is None:
                priors[name] = (src, (R0, t0))
                continue
            Rm, tm = motion
            uvp, zp = project_c(K, (Rm @ Xc.T).T + tm)
            if np.median(zp) < 1:
                priors[name] = (src, (R0, t0))
                continue
            priors[name] = (uvp, (Rm @ R0, Rm @ t0 + tm))
        uv_true_cam = project_c(K, (Re @ Xc.T).T + te)[0]
        ego_cam.append(px(priors["local_egolite"][0][vis], uv_true_cam[vis]))
        err["homography"].append(px(uvH[vis], truth[vis]))
        err["lk_pnp"].append(px(lk_pnp(g0, g1, src, Xm, K, (R0, t0), vis)[vis], truth[vis]))
        err["flow_pnp"].append(px(flow_pnp(g0, g1, src, Xm, K, (R0, t0), vis)[vis], truth[vis]))
        ic = icp(Xc, os.path.join(seq, "depth", f"{t + stride:06d}.png"), K, Xm)
        err["icp"].append(px((src if ic is None else ic)[vis], truth[vis]))
        got_main = {}
        for w in windows:
            for name in LOCAL:
                pred, guess = priors[name]
                got = register_local(g0, g1, src, pred, Xm, K, guess, vis, w)
                err[f"{name}@{w}"].append(px(got[vis], truth[vis]))
                if w == main_win:
                    got_main[name] = got
        examples.append((err[f"local_none@{main_win}"][-1], err[f"local_egolite@{main_win}"][-1], im1,
                         truth, got_main["local_none"], got_main["local_egolite"]))
    if not examples:
        raise ValueError("no valid update")
    return {"sequence": os.path.basename(seq), "stride": stride, "relaxed_stride": relaxed,
            "camera_px": cam_px, "tissue_px": tis_px, "n": len(examples),
            "egolite_vs_true_camera_px": float(np.median(ego_cam)), "errors": err}, examples


def pick_example(examples, med_none, med_ego):
    """The update whose two errors are closest to the sequence medians, with no other filter (representative frame)."""
    return min(examples, key=lambda e: abs(e[0] - med_none) + abs(e[1] - med_ego))


def save_gallery(rows, dest):
    """Grid of equal-size cells, three sequences per row. Each sequence is a pair: camera motion left in | after EgoLite.
    Column headers are drawn once; each cell carries the sequence number and the error of that update."""
    out_w, out_h = 300, 180
    font = cv2.FONT_HERSHEY_SIMPLEX

    def label(img, text, org, scale=0.9):
        cv2.putText(img, text, org, font, scale, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(img, text, org, font, scale, (25, 25, 25), 2, cv2.LINE_AA)

    panels = []
    for index, im, truth, raw, ego, raw_err, ego_err in rows:
        pts = np.vstack([truth, raw, ego])
        box = (max(0, int(pts[:, 0].min()) - 36), max(0, int(pts[:, 1].min()) - 28),
               min(im.shape[1], int(pts[:, 0].max()) + 36), min(im.shape[0], int(pts[:, 1].max()) + 28))
        pair = []
        for est, e, first in ((raw, raw_err, True), (ego, ego_err, False)):
            x0, y0, x1, y1 = box
            src = im[y0:y1, x0:x1]
            scale = min(out_w / max(1, src.shape[1]), out_h / max(1, src.shape[0]))
            nw, nh = max(1, int(src.shape[1] * scale)), max(1, int(src.shape[0] * scale))
            crop = np.full((out_h, out_w, 3), 248, np.uint8)
            ox, oy = (out_w - nw) // 2, (out_h - nh) // 2
            crop[oy:oy + nh, ox:ox + nw] = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)

            def pt(u, v):
                return int(round((u - x0) * scale + ox)), int(round((v - y0) * scale + oy))
            for a_, b_ in zip(est, truth):
                cv2.line(crop, pt(*a_), pt(*b_), (255, 255, 255), 1, cv2.LINE_AA)
            for u in truth:
                cv2.circle(crop, pt(*u), 6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(crop, pt(*u), 5, (40, 160, 40), 2, cv2.LINE_AA)
            for u in est:
                cv2.circle(crop, pt(*u), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(crop, pt(*u), 3, (40, 40, 210), -1, cv2.LINE_AA)
            label(crop, (f"{index}   " if first else "") + f"{e:.1f} px", (8, 28))
            pair.append(crop)
        panels.append(np.hstack([pair[0], np.full((out_h, 6, 3), 255, np.uint8), pair[1]]))
    per_row = 3 if len(panels) % 3 == 0 else 2
    while len(panels) % per_row:
        panels.append(np.full_like(panels[0], 255))
    gutter = 14
    rows_img = []
    for i in range(0, len(panels), per_row):
        row = panels[i]
        for k in range(1, per_row):
            row = np.hstack([row, np.full((out_h, gutter, 3), 255, np.uint8), panels[i + k]])
        rows_img += [row, np.full((8, row.shape[1], 3), 255, np.uint8)]
    width = rows_img[0].shape[1]
    header = np.full((40, width, 3), 255, np.uint8)
    for k in range(per_row):
        x = k * (2 * out_w + 6 + gutter)
        for j, txt in enumerate(("camera motion left in", "after EgoLite")):
            (tw, _), _ = cv2.getTextSize(txt, font, 0.8, 2)
            cx = x + j * (out_w + 6) + (out_w - tw) // 2
            cv2.putText(header, txt, (cx, 28), font, 0.8, (25, 25, 25), 2, cv2.LINE_AA)
    legend = np.full((44, width, 3), 255, np.uint8)
    cv2.circle(legend, (20, 22), 7, (40, 160, 40), 2, cv2.LINE_AA)
    cv2.putText(legend, "true tissue patch", (38, 30), font, 0.8, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.circle(legend, (330, 22), 6, (40, 40, 210), -1, cv2.LINE_AA)
    cv2.putText(legend, "registered model", (346, 30), font, 0.8, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.imwrite(dest, np.vstack([header] + rows_img + [legend]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT / "sim_data_abs" / "test"))
    ap.add_argument("--ckpt", default=str(ROOT / "results" / "egolite" / "r_v2" / "last.pth"))
    ap.add_argument("--out", default=str(ROOT / "results" / "ar_reg"))
    ap.add_argument("--fig", default=str(ROOT / "results" / "ar_reg" / "fig_ar_registration.png"))
    ap.add_argument("--windows", default="10,14,18,26,34")
    ap.add_argument("--main-window", type=int, default=18)
    args = ap.parse_args()
    windows = [int(w) for w in args.windows.split(",")]
    os.makedirs(args.out, exist_ok=True)
    seqs = select_sequences(args.root)
    net = None
    results, gallery, skipped = [], [], []
    for path_mm, amp_mm, seq in seqs:
        if net is None:
            first = cv2.imread(os.path.join(seq, "color", "000000.png"))
            net = EgoLite(args.ckpt, np.eye(3) * 500, first.shape[1], first.shape[0])
        try:
            res, examples = evaluate_sequence(seq, net, windows, args.main_window)
        except (ValueError, FileNotFoundError) as exc:
            skipped.append({"sequence": os.path.basename(seq), "reason": str(exc)})
            continue
        res.update({"camera_path_mm": path_mm, "respiration_mm": amp_mm})
        results.append(res)
        e = res["errors"]
        ex = pick_example(examples, np.median(e[f"local_none@{args.main_window}"]),
                          np.median(e[f"local_egolite@{args.main_window}"]))
        gallery.append((len(gallery) + 1, ex[2], ex[3], ex[4], ex[5], ex[0], ex[1]))
        print(f"sequence {len(results)} ({res['sequence']}): stride {res['stride']}, {res['n']} updates, "
              f"none {np.median(e[f'local_none@{args.main_window}']):.1f} px, "
              f"EgoLite {np.median(e[f'local_egolite@{args.main_window}']):.1f} px", flush=True)

    mw = args.main_window
    methods = list(GLOBAL) + [f"{m}@{mw}" for m in LOCAL]
    seq_med = {m: np.array([np.median(r["errors"][m]) for r in results]) for m in methods}
    pooled = {m: np.concatenate([r["errors"][m] for r in results]) for m in methods}
    ref = seq_med[f"local_egolite@{mw}"]
    table = {}
    for m in methods:
        diff = seq_med[m] - ref
        p = None if m == f"local_egolite@{mw}" or np.allclose(diff, 0) else float(wilcoxon(seq_med[m], ref).pvalue)
        table[m] = {"median_of_sequence_medians_px": float(np.median(seq_med[m])),
                    "within_10px_mean_over_sequences": float(np.mean([np.mean(np.array(r["errors"][m]) < 10) for r in results])),
                    "within_10px_pooled": float(np.mean(pooled[m] < 10)),
                    "lowest_on_sequences": int(sum(seq_med[m][i] <= min(seq_med[k][i] for k in methods if not k.startswith("local_true")) + 1e-9
                                               for i in range(len(results)))),
                    "wilcoxon_p_vs_egolite": p}
    window_curve = {m: {str(w): float(np.median([np.median(r["errors"][f"{m}@{w}"]) for r in results])) for w in windows}
                    for m in LOCAL}
    summary = {"n_sequences": len(results), "n_updates": int(sum(r["n"] for r in results)), "main_window_px": mw,
               "egolite_vs_true_camera_px_median": float(np.median([r["egolite_vs_true_camera_px"] for r in results])),
               "table": table, "window_curve": window_curve, "skipped": skipped,
               "sequences": [{k: v for k, v in r.items() if k != "errors"} | {
                   "medians": {m: float(np.median(r["errors"][m])) for m in r["errors"]}} for r in results]}
    json.dump(summary, open(os.path.join(args.out, "ar_registration_summary.json"), "w"), indent=1)
    os.makedirs(os.path.dirname(os.path.abspath(args.fig)), exist_ok=True)
    save_gallery(gallery, args.fig)
    print(f"\n{len(results)} sequences, {summary['n_updates']} updates; EgoLite vs true camera "
          f"{summary['egolite_vs_true_camera_px_median']:.1f} px")
    for m in methods:
        t = table[m]
        p = "-" if t["wilcoxon_p_vs_egolite"] is None else f"{t['wilcoxon_p_vs_egolite']:.4f}"
        print(f"{m:22s} {t['median_of_sequence_medians_px']:6.1f} px  within10 {t['within_10px_mean_over_sequences']:4.0%}"
              f"  lowest {t['lowest_on_sequences']}/{len(results)}  p {p}")
    print("window curve (median px):")
    for m, curve in window_curve.items():
        print(f"  {m:18s} " + "  ".join(f"{w}:{v:5.1f}" for w, v in curve.items()))


if __name__ == "__main__":
    main()
