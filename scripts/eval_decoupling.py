"""Direct decoupling metrics on sequences with absolute-motion ground truth.

Tissue absolute motion (main metric). Grid points of frame 0 on tissue are tracked
through the window; at every frame k the point is back-projected with the predicted
depth, moved to the world frame with a camera trajectory, and compared with its frame-0
world position. Ground truth follows the same point with flows rebuilt from depth, pose
and the world displacement field, accumulating the per-frame world displacement.
Per-frame tissue motion (~0.1-0.4 mm at 10 fps) is below depth noise, so the error is
measured on this accumulated displacement (mm, world frame = first camera frame).
  ours      estimated trajectory, predicted flow and depth
  oracle    ground-truth trajectory with our flow and depth (decoupling upper bound)
  relonly   camera assumed still: observed motion attributed to tissue (no decoupling)
  zero      tissue assumed still
Also: per-pair camera / scene flow EPE (px at the 320x256 input) and anchor AUROC.

  python scripts/eval_decoupling.py --ckpt results/egolite/r_v2/last.pth --list lists/r2s_test.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.metrics import load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames  # noqa: E402
from endosim.lite.data import H_IN, STRIDE, W_IN, abs_static_labels, load_frames  # noqa: E402
from endosim.lite.egolite import build_from_args  # noqa: E402
from endosim.lite.infer import _intrinsics, _run_pairs  # noqa: E402

UNIT = 50.0
EPS = 0.2


def backproject(u, v, Z, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return torch.stack([(u - cx) / fx * Z, (v - cy) / fy * Z, Z], -1)


def project(X, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = X[..., 2].clamp(min=1e-3)
    return torch.stack([fx * X[..., 0] / z + cx, fy * X[..., 1] / z + cy], -1)


def sample_grid(M, pu, pv, h, w):
    """Bilinear sample of an (h, w) grid map at network-input pixel positions."""
    gx = (2 * (pu + 0.5) / (w * STRIDE)) - 1
    gy = (2 * (pv + 0.5) / (h * STRIDE)) - 1
    g = torch.stack([gx, gy], -1).reshape(1, 1, -1, 2).to(M.dtype)
    s = F.grid_sample(M[None, None], g, mode="bilinear", padding_mode="border", align_corners=False)
    return s.reshape(pu.shape), (gx.abs() < 1) & (gy.abs() < 1)


def gt_grid(d, k, h, w):
    """World displacement (mm), depth (mm), tissue mask and stationarity on the (h, w) grid."""
    z = np.load(os.path.join(d, "abs_motion", f"{k:06d}.npz"))
    wd = z["world_disp"].astype(np.float32)
    s = wd.shape[0] // h
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            wd = np.nanmean(wd[: h * s, : w * s].reshape(h, s, w, s, 3), axis=(1, 3))
    dep = cv2.imread(os.path.join(d, "depth", f"{k:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    dep = cv2.resize(dep, (w, h), interpolation=cv2.INTER_NEAREST)
    inst = cv2.imread(os.path.join(d, "object_mask", f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
    inst = cv2.resize(inst, (w, h), interpolation=cv2.INTER_NEAREST)
    lab = abs_static_labels(d, k, inst, h, w)
    return wd, dep, inst, lab


def chain_c2w(R, t):
    """Relative k -> k+1 poses (X_{k+1} = R X_k + t) to camera-to-world, T(0) = I."""
    T = [torch.eye(4, dtype=R.dtype, device=R.device)]
    for k in range(R.shape[0]):
        M = torch.eye(4, dtype=R.dtype, device=R.device)
        M[:3, :3], M[:3, 3] = R[k], t[k]
        T.append(T[-1] @ torch.linalg.inv(M))
    return torch.stack(T)


def to_world(T, X):
    return X @ T[:3, :3].T + T[:3, 3]


@torch.no_grad()
def evaluate(model, seqs, max_frames=64):
    h, w = H_IN // STRIDE, W_IN // STRIDE
    names = ("ours", "oracle", "relonly", "zero")
    acc = {k: [] for k in names}
    pred_vec = {k: [] for k in names}
    img_acc = {k: [] for k in names}
    img_vec = {k: [] for k in names}
    img_gt = []
    gt_vec_all = []
    gt_mag_all = []
    cam_epe, scene_epe, scene_mag, ys, ps, per_seq = [], [], [], [], [], []
    for d in seqs:
        gt_all = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
        idx = np.arange(min(len(gt_all), max_frames))
        imgs, phs, _ = load_frames(list_color_frames(d, indices=idx))
        img = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
        ph = torch.from_numpy(phs)[:, None].cuda()
        K = torch.from_numpy(_intrinsics(d)).cuda().double()
        ii = torch.arange(len(idx) - 1, device="cuda")
        o = _run_pairs(model, img, ph, K.float(), ii, ii + 1, 16, extra=("f_obs",))
        u, v = o["u"].double(), o["v"].double()
        Z = o["Z"].double() * UNIT
        f_obs = o["f_obs"].double()
        P = f_obs.shape[0]
        T_est = chain_c2w(o["R"].double(), o["t"].double() * UNIT)
        T_gt = torch.from_numpy(gt_all[idx]).cuda()
        T_gt = torch.linalg.inv(T_gt[0])[None] @ T_gt
        # per-pair ground truth on the grid
        gts = []
        for j in range(P):
            k = int(idx[j])
            if not os.path.exists(os.path.join(d, "abs_motion", f"{k:06d}.npz")):
                break
            wd, dep, inst, lab = gt_grid(d, k, h, w)
            Trel = torch.linalg.inv(T_gt[j + 1]) @ T_gt[j]
            Rg, tg = Trel[:3, :3], Trel[:3, 3]
            wdw = torch.from_numpy(np.nan_to_num(wd)).cuda().double()
            Dc = wdw @ T_gt[j][:3, :3]                                       # camera-j axes
            Xg = backproject(u, v, torch.from_numpy(dep).cuda().double(), K)
            uv = torch.stack([u, v], -1)
            cam_gt = project(Xg @ Rg.T + tg, K) - uv
            tot_gt = project((Xg + Dc) @ Rg.T + tg, K) - uv
            ok = torch.from_numpy(np.isfinite(wd).all(-1) & (dep > 0)).cuda()
            gts.append({"wd": wdw, "tot": tot_gt, "ok": ok, "inst": inst, "lab": lab, "dep": dep})
            f_cam = o["f_cam"][j].double().permute(1, 2, 0)
            tis = ok & torch.from_numpy(inst == 1).cuda()
            if tis.any():
                cam_epe.append((f_cam - cam_gt).norm(dim=-1)[tis].cpu())
                sc_gt = tot_gt - cam_gt
                scene_epe.append((f_obs[j].permute(1, 2, 0) - f_cam - sc_gt).norm(dim=-1)[tis].cpu())
                scene_mag.append(sc_gt.norm(dim=-1)[tis].cpu())
            if lab is not None:
                okl = lab >= 0
                ys.append((lab[okl] == 0).astype(np.int8))
                ps.append(torch.softmax(o["logits"][j].float(), 0)[0].cpu().numpy()[okl])
        if len(gts) < 2:
            continue
        # track frame-0 tissue grid points
        start = gts[0]["ok"] & torch.from_numpy(gts[0]["inst"] == 1).cuda()
        pu_p, pv_p = u[start].clone(), v[start].clone()               # predicted track
        pu_g, pv_g = pu_p.clone(), pv_p.clone()                       # ground-truth track
        alive = torch.ones_like(pu_p, dtype=torch.bool)
        Dgt = torch.zeros(pu_p.shape[0], 3, dtype=torch.float64, device="cuda")
        X0c = backproject(pu_p, pv_p, Z[0][start], K)
        X0 = {"ours": to_world(T_est[0], X0c), "oracle": to_world(T_gt[0], X0c), "relonly": X0c}
        p0 = torch.stack([pu_p, pv_p], -1)
        X0g = backproject(pu_p, pv_p, torch.from_numpy(gts[0]["dep"]).cuda().double()[start], K)
        seq_err, peak = [], torch.zeros_like(pu_p)
        for k in range(1, len(gts)):
            g = gts[k - 1]
            s_wd = torch.stack([sample_grid(g["wd"][..., c], pu_g, pv_g, h, w)[0] for c in range(3)], -1)
            s_tot = torch.stack([sample_grid(g["tot"][..., c], pu_g, pv_g, h, w)[0] for c in range(2)], -1)
            s_ok, ins_g = sample_grid(g["ok"].double(), pu_g, pv_g, h, w)
            fu, ins_p = sample_grid(f_obs[k - 1][0], pu_p, pv_p, h, w)
            fv, _ = sample_grid(f_obs[k - 1][1], pu_p, pv_p, h, w)
            Dgt = Dgt + s_wd
            pu_g, pv_g = pu_g + s_tot[:, 0], pv_g + s_tot[:, 1]
            pu_p, pv_p = pu_p + fu, pv_p + fv
            alive = alive & ins_g & ins_p & (s_ok > 0.99)
            if k >= Z.shape[0]:
                break
            Zk, ins_z = sample_grid(Z[k], pu_p, pv_p, h, w)
            alive = alive & ins_z
            Xkc = backproject(pu_p, pv_p, Zk, K)
            Dp = {"ours": to_world(T_est[k], Xkc) - X0["ours"],
                  "oracle": to_world(T_gt[k], Xkc) - X0["oracle"],
                  "relonly": Xkc - X0["relonly"], "zero": torch.zeros_like(Dgt)}
            if not bool(alive.any()):
                break
            # image-space tissue motion: tracked position minus where camera motion alone puts the point
            def cam_only(T0k, X):
                return project(X @ T0k[:3, :3].T + T0k[:3, 3], K)
            T0k_est = torch.linalg.inv(T_est[k]) @ T_est[0]
            T0k_gt = torch.linalg.inv(T_gt[k]) @ T_gt[0]
            pk_p = torch.stack([pu_p, pv_p], -1)
            s_gt = torch.stack([pu_g, pv_g], -1) - cam_only(T0k_gt, X0g)
            s2d = {"ours": pk_p - cam_only(T0k_est, X0c), "oracle": pk_p - cam_only(T0k_gt, X0c),
                   "relonly": pk_p - p0, "zero": torch.zeros_like(pk_p)}
            for name, S in s2d.items():
                img_acc[name].append((S - s_gt).norm(dim=-1)[alive].cpu())
                img_vec[name].append(S[alive].cpu())
            img_gt.append(s_gt[alive].cpu())
            gm = Dgt.norm(dim=-1)
            peak = torch.maximum(peak, gm)
            for name, D in Dp.items():
                acc[name].append((D - Dgt).norm(dim=-1)[alive].cpu())
                pred_vec[name].append(D[alive].cpu())
            gt_mag_all.append(gm[alive].cpu())
            gt_vec_all.append(Dgt[alive].cpu())
            seq_err.append(float((Dp["ours"] - Dgt).norm(dim=-1)[alive].mean()))
        moving = peak >= 1.0
        per_seq.append({"seq": os.path.basename(d), "tissue_epe_mm": float(np.mean(seq_err)) if seq_err else None,
                        "frac_moving_points": float(moving.float().mean())})
    cat = lambda xs: torch.cat(xs).numpy() if xs else np.array([np.nan])  # noqa: E731
    gmag = cat(gt_mag_all)
    out = {"tissue_abs_motion_epe_mm": {k: float(cat(v).mean()) for k, v in acc.items()},
           "tissue_abs_motion_epe_mm_median": {k: float(np.median(cat(v))) for k, v in acc.items()},
           "tissue_gt_displacement_mm_mean": float(np.nanmean(gmag)),
           "moving_tissue_epe_mm": {k: float(cat(v)[gmag >= 1.0].mean()) for k, v in acc.items()},
           "static_tissue_epe_mm": {k: float(cat(v)[gmag < EPS].mean()) for k, v in acc.items()},
           "camera_flow_epe_px": float(cat(cam_epe).mean()),
           "camera_flow_epe_px_median": float(np.median(cat(cam_epe))),
           "scene_flow_epe_px": float(cat(scene_epe).mean()),
           "scene_flow_epe_px_median": float(np.median(cat(scene_epe))),
           "scene_flow_gt_mag_px": float(cat(scene_mag).mean()),
           "n_point_frames": int(cat(acc["ours"]).size)}
    # can the estimate tell which tissue moved, how much and where to?
    from scipy.stats import spearmanr
    gv = torch.cat(gt_vec_all).numpy() if gt_vec_all else np.zeros((0, 3))
    mov, sta = gmag >= 1.0, gmag < EPS
    det = {}
    for name in ("ours", "oracle", "relonly"):
        pv = torch.cat(pred_vec[name]).numpy() if pred_vec[name] else np.zeros((0, 3))
        pm = np.linalg.norm(pv, axis=1)
        sel = mov | sta
        auc = (float(roc_auc_score(mov[sel].astype(int), pm[sel]))
               if sel.any() and 0 < mov[sel].mean() < 1 else None)
        rho = float(spearmanr(pm, gmag).correlation) if len(pm) > 10 else None
        cos = (float(np.mean(np.sum(pv[mov] * gv[mov], 1)
                             / (np.linalg.norm(pv[mov], axis=1) * np.linalg.norm(gv[mov], axis=1) + 1e-9)))
               if mov.any() else None)
        det[name] = {"moving_vs_static_auroc": auc, "magnitude_spearman": rho, "moving_direction_cosine": cos}
    det["zero"] = {"moving_vs_static_auroc": 0.5, "magnitude_spearman": 0.0, "moving_direction_cosine": 0.0}
    out["tissue_motion_detection"] = det
    # image-space tissue motion (accumulated from frame 0), px at the network input
    ig = torch.cat(img_gt).numpy() if img_gt else np.zeros((0, 2))
    igm = np.linalg.norm(ig, axis=1)
    mov2, sta2 = igm >= 1.0, igm < 0.25
    img = {"gt_mag_px_mean": float(igm.mean()) if len(igm) else None}
    for name in names:
        e = cat(img_acc[name])
        pv = torch.cat(img_vec[name]).numpy() if img_vec[name] else np.zeros((0, 2))
        pm = np.linalg.norm(pv, axis=1)
        sel = mov2 | sta2
        auc = (float(roc_auc_score(mov2[sel].astype(int), pm[sel]))
               if name != "zero" and sel.any() and 0 < mov2[sel].mean() < 1 else 0.5)
        cos = (float(np.mean(np.sum(pv[mov2] * ig[mov2], 1)
                             / (np.linalg.norm(pv[mov2], axis=1) * igm[mov2] + 1e-9)))
               if name != "zero" and mov2.any() else 0.0)
        img[name] = {"epe_px": float(e.mean()), "epe_px_median": float(np.median(e)),
                     "moving_epe_px": float(e[mov2].mean()) if mov2.any() else None,
                     "moving_vs_static_auroc": auc, "moving_direction_cosine": cos}
    out["tissue_motion_image"] = img
    if ys:
        Y, Pp = np.concatenate(ys), np.concatenate(ps)
        out["anchor_auroc"] = float(roc_auc_score(Y, Pp)) if 0 < Y.mean() < 1 else None
        out["true_static_frac"] = float(Y.mean())
    out["per_seq"] = per_seq
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--list", default="lists/r2s_test.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablate", default=None)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_from_args(ck["args"])
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()
    model.ablate = args.ablate
    seqs = [l.strip() for l in open(args.list, encoding="utf-8") if l.strip()]
    res = evaluate(model, seqs)
    res |= {"ckpt": args.ckpt, "list": args.list, "ablate": args.ablate}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_seq"}, indent=1))


if __name__ == "__main__":
    main()
