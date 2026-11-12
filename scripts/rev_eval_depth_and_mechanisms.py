"""Comprehensive reproducible audit for VRIH Revision R1.1, R1.2, R1.3, R1.4, R1.6, and R2.3.

Optimized:
- Sets torch.set_num_threads(2) & cv2.setNumThreads(1) to avoid CPU thread-pool contention.
- Batches pair forward passes within each sequence (BATCH=2, <0.4 GiB VRAM).
- Preloads decoded frame tensors per cohort once where cohorts are small, and skips already-written JSON files.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

cv2.setNumThreads(1)
torch.set_num_threads(2)

ROOT = Path(r"D:\ego_motiion_Camera")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from baseline_sota import eval_with_scale_correction  # noqa: E402
from endosim.eval.metrics import load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames, select_frame_indices  # noqa: E402
from endosim.lite.data import H_IN, STRIDE, W_IN, abs_static_labels, load_frames  # noqa: E402
from endosim.lite.egolite import build_from_args, chain  # noqa: E402
from endosim.lite.infer import _intrinsics, infer_sequence  # noqa: E402

OUT = ROOT / "results" / "revision_vrih"
OUT.mkdir(parents=True, exist_ok=True)
DEPTH_UNIT_MM = 50.0
BATCH = 2


def load_egolite(ckpt_path: str | Path, device: str = "cuda"):
    ck = torch.load(str(ckpt_path), map_location="cpu")
    cfg = dict(ck.get("args", {}))
    m = build_from_args(cfg, pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


def read_list(list_rel: str) -> list[str]:
    p = ROOT / list_rel
    seqs = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]
    return [s for s in seqs if os.path.isfile(os.path.join(s, "pose_c2w.txt"))]


def depth_metrics(z_pred_mm: np.ndarray, z_gt_mm: np.ndarray, valid: np.ndarray) -> dict | None:
    if int(valid.sum()) < 50:
        return None
    p = np.clip(z_pred_mm[valid].astype(np.float64), 1e-3, 1e6)
    g = z_gt_mm[valid].astype(np.float64)
    scale = float(np.median(g) / np.median(p))
    ps = p * scale
    ratio = np.maximum(g / ps, ps / g)
    dlog = np.log(ps) - np.log(g)
    silog = float(np.sqrt(max(0.0, np.mean(dlog ** 2) - (np.mean(dlog)) ** 2)) * 100.0)
    return {
        "abs_rel": float(np.mean(np.abs(g - ps) / g)),
        "sq_rel": float(np.mean(((g - ps) ** 2) / g)),
        "rmse_mm": float(np.sqrt(np.mean((g - ps) ** 2))),
        "rmse_log": float(np.sqrt(np.mean(dlog ** 2))),
        "silog": silog,
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
        "abs_rel_fixed50": float(np.mean(np.abs(g - p) / g)),
        "rmse_fixed50_mm": float(np.sqrt(np.mean((g - p) ** 2))),
        "median_scale_ratio": scale,
        "gt_median_mm": float(np.median(g)),
        "pred50_median_mm": float(np.median(p)),
    }


def aggregate_depth(records: list[dict]) -> dict:
    keys = [k for k in records[0] if k != "seq_id"]
    agg = {"n_items": len(records)}
    for k in keys:
        arr = np.array([r[k] for r in records], dtype=np.float64)
        agg[k] = float(np.mean(arr))
        if k == "median_scale_ratio":
            agg["median_scale_ratio_std"] = float(np.std(arr))
            agg["median_scale_ratio_min"] = float(np.min(arr))
            agg["median_scale_ratio_max"] = float(np.max(arr))
    return agg


@torch.no_grad()
def eval_depth_sim_cohort(models: dict[str, torch.nn.Module], seqs: list[str], sample_pairs: int = 6) -> dict:
    """Loads each sequence once and evaluates all passed models simultaneously!"""
    recs = {mname: [] for mname in models}
    t0 = time.time()
    for si, d in enumerate(seqs):
        gt_all = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
        idx = select_frame_indices(gt_all, "uniform", max_frames=64)
        paths = list_color_frames(d, indices=idx)
        if len(paths) < 2:
            continue
        pids = np.linspace(0, len(paths) - 2, min(sample_pairs, len(paths) - 1), dtype=int)
        # Load only needed unique frame paths!
        need = sorted(set(int(x) for pid in pids for x in (pid, pid + 1)))
        pos_map = {orig_i: k for k, orig_i in enumerate(need)}
        imgs, phs, _ = load_frames([paths[i] for i in need])
        img_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
        ph_t = torch.from_numpy(phs)[:, None].cuda()
        K = torch.from_numpy(_intrinsics(d)).cuda()
        i0_idx = [pos_map[int(pid)] for pid in pids]
        i1_idx = [pos_map[int(pid) + 1] for pid in pids]

        z_gts = []
        for pid in pids:
            fidx = int(idx[pid])
            z_gt = cv2.imread(os.path.join(d, "depth", f"{fidx:06d}.png"), cv2.IMREAD_UNCHANGED)
            z_gts.append(cv2.resize(z_gt.astype(np.float32), (W_IN, H_IN), interpolation=cv2.INTER_NEAREST)
                         if z_gt is not None else None)

        for mname, mod in models.items():
            zs = []
            for st in range(0, len(pids), BATCH):
                ed = min(len(pids), st + BATCH)
                a0 = img_t[i0_idx[st:ed]]
                a1 = img_t[i1_idx[st:ed]]
                p0 = ph_t[i0_idx[st:ed]] if mod.use_ph else None
                p1 = ph_t[i1_idx[st:ed]] if mod.use_ph else None
                o = mod(a0, a1, K[None].expand(ed - st, 3, 3), p0, p1)
                z_up = F.interpolate(o["Z"][:, None].float() * DEPTH_UNIT_MM, size=(H_IN, W_IN),
                                     mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
                zs.append(z_up)
            z_all = np.concatenate(zs, 0)
            seq_m = []
            for z_pred, z_gt in zip(z_all, z_gts):
                if z_gt is None:
                    continue
                m = depth_metrics(z_pred, z_gt, (z_gt > 1.0) & (z_gt < 500.0))
                if m:
                    seq_m.append(m)
            if seq_m:
                avg = {k: float(np.mean([x[k] for x in seq_m])) for k in seq_m[0]}
                avg["seq_id"] = os.path.basename(d)
                recs[mname].append(avg)
        if (si + 1) % 30 == 0:
            print(f"    depth sim {si + 1}/{len(seqs)} ({time.time() - t0:.1f}s)", flush=True)
    return {mname: aggregate_depth(r) for mname, r in recs.items()}


@torch.no_grad()
def eval_depth_real_scared_c3vd(models: dict[str, torch.nn.Module]) -> dict:
    res = {mname: {} for mname in models}
    scared_seqs = read_list("lists/scared_official.txt")
    s_recs = {mname: [] for mname in models}
    for d in scared_seqs:
        meta = json.loads(Path(d, "meta.json").read_text(encoding="utf-8"))
        kf_dir = meta["src_path"]
        xyz = cv2.imread(os.path.join(kf_dir, "left_depth_map.tiff"), cv2.IMREAD_UNCHANGED)
        paths = list_color_frames(d, indices=[0, 1])
        if xyz is None or len(paths) < 2:
            continue
        z_gt = cv2.resize(xyz[..., 0].astype(np.float32), (W_IN, H_IN), interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(z_gt) & (z_gt > 5.0) & (z_gt < 300.0)
        imgs, phs, _ = load_frames(paths)
        K = torch.from_numpy(_intrinsics(d)).cuda()
        img = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
        ph = torch.from_numpy(phs)[:, None].cuda()
        for mname, mod in models.items():
            o = mod(img[:1], img[1:], K[None], ph[:1] if mod.use_ph else None, ph[1:] if mod.use_ph else None)
            z_up = F.interpolate(o["Z"][:, None].float() * DEPTH_UNIT_MM, size=(H_IN, W_IN),
                                 mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            m = depth_metrics(z_up, z_gt, valid)
            if m:
                m["seq_id"] = meta["seq_id"]
                s_recs[mname].append(m)
    for mname in models:
        res[mname]["scared_keyframes_11"] = aggregate_depth(s_recs[mname])
        res[mname]["scared_disjoint_d2_d3_8"] = aggregate_depth(
            [r for r in s_recs[mname] if "_dataset_2_" in r["seq_id"] or "_dataset_3_" in r["seq_id"]])

    c3vd_d = str(ROOT / "sim_data" / "real_refs" / "cecum_t1_a_cecum_t1_a")
    meta = json.loads(Path(c3vd_d, "meta.json").read_text(encoding="utf-8"))
    src_dir = meta["src_path"]
    gt_all = load_pose_txt(os.path.join(c3vd_d, "pose_c2w.txt"))
    idx = select_frame_indices(gt_all, "uniform", max_frames=64)
    paths = list_color_frames(c3vd_d, indices=idx)
    imgs, phs, _ = load_frames(paths)
    img_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
    ph_t = torch.from_numpy(phs)[:, None].cuda()
    K = torch.from_numpy(_intrinsics(c3vd_d)).cuda()
    c_recs = {mname: [] for mname in models}
    for k in range(0, len(paths) - 1, 2):
        fidx = int(idx[k])
        dtiff = cv2.imread(os.path.join(src_dir, f"{fidx:04d}_depth.tiff"), cv2.IMREAD_UNCHANGED)
        if dtiff is None:
            continue
        val_full = (dtiff > 0) & (dtiff < 65535)
        z_mm = dtiff.astype(np.float32) / 65535.0 * 100.0
        z_mm[~val_full] = 0.0
        z_gt = cv2.resize(z_mm, (W_IN, H_IN), interpolation=cv2.INTER_NEAREST)
        for mname, mod in models.items():
            o = mod(img_t[k:k + 1], img_t[k + 1:k + 2], K[None],
                    ph_t[k:k + 1] if mod.use_ph else None, ph_t[k + 1:k + 2] if mod.use_ph else None)
            z_up = F.interpolate(o["Z"][:, None].float() * DEPTH_UNIT_MM, size=(H_IN, W_IN),
                                 mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            m = depth_metrics(z_up, z_gt, (z_gt > 1.0) & (z_gt < 99.5))
            if m:
                m["seq_id"] = f"frame_{fidx}"
                c_recs[mname].append(m)
    for mname in models:
        res[mname]["c3vd_cecum_t1_a_pairs"] = aggregate_depth(c_recs[mname])
    return res


@torch.no_grad()
def eval_all_infer_variants_cohort(model, seqs: list[str], max_frames: int = 64,
                                   with_solver_choices: bool = False) -> dict:
    variants = ["full", "nogn", "uniform", "nolayer", "novar", "staticonly"]
    if with_solver_choices:
        variants += ["stride4_gn", "fullres_gn", "oob_mask", "fb_occ_mask"]
    per_var = {v: [] for v in variants}
    timing = {v: [] for v in (["stride8_gn", "stride4_gn", "fullres_gn", "fb_occ_extra"] if with_solver_choices else [])}
    t0_all = time.time()

    for si, d in enumerate(seqs):
        gt_all = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
        idx = select_frame_indices(gt_all, "uniform", max_frames=max_frames) if max_frames else np.arange(len(gt_all))
        paths = list_color_frames(d, indices=idx)
        imgs, phs, _ = load_frames(paths)
        K = torch.from_numpy(_intrinsics(d)).cuda()
        img = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
        ph = torch.from_numpy(phs)[:, None].cuda()
        n = len(paths)
        accum = {v: ([], []) for v in variants}

        for st in range(0, n - 1, BATCH):
            ed = min(n - 1, st + BATCH)
            b_sz = ed - st
            i0, i1 = img[st:ed], img[st + 1:ed + 1]
            p0, p1 = ph[st:ed], ph[st + 1:ed + 1]
            Kb = K[None].expand(b_sz, 3, 3)

            fl_full = model.observed_flow(i0, i1)
            o = model(i0, i1, Kb, p0, p1, flow=fl_full)
            Z, u, v, f_obs = o["Z"].float(), o["u"].float(), o["v"].float(), o["f_obs"].float()
            R0, t0 = o["R0"].float(), o["t0"].float()
            prob = torch.softmax(o["logits"].float(), 1)
            share = torch.sigmoid(model.deform_logit.float())
            inv_var = torch.exp(-o["logvar"].float())

            w_map = {
                "full": (prob[:, 0] + share * prob[:, 1]) * inv_var,
                "uniform": torch.ones_like(inv_var),
                "nolayer": inv_var,
                "novar": prob[:, 0] + share * prob[:, 1],
                "staticonly": prob[:, 0] * inv_var,
            }
            accum["nogn"][0].append(R0)
            accum["nogn"][1].append(t0)

            do_time = with_solver_choices and (si < 5)
            for vname, w_pix in w_map.items():
                if vname == "full" and do_time:
                    torch.cuda.synchronize()
                    ts0 = time.perf_counter()
                R, t, _, _ = model.solve_pose(Z, u, v, Kb.float(), f_obs, w_pix, R0, t0)
                if vname == "full" and do_time:
                    torch.cuda.synchronize()
                    timing["stride8_gn"].append((time.perf_counter() - ts0) * 1000.0 / b_sz)
                accum[vname][0].append(R)
                accum[vname][1].append(t)

            if with_solver_choices:
                w_full = w_map["full"]
                up, vp = u[None] + f_obs[:, 0], v[None] + f_obs[:, 1]
                in_b = ((up >= 0) & (up <= W_IN - 1) & (vp >= 0) & (vp <= H_IN - 1)).float()
                Ro, to, _, _ = model.solve_pose(Z, u, v, Kb.float(), f_obs, w_full * in_b, R0, t0)
                accum["oob_mask"][0].append(Ro)
                accum["oob_mask"][1].append(to)

                if do_time:
                    torch.cuda.synchronize()
                    tfb0 = time.perf_counter()
                fl_bwd = model.observed_flow(i1, i0)
                f_bwd_s8 = F.avg_pool2d(fl_bwd, STRIDE, STRIDE).float()
                f_bwd_w = model._warp(f_bwd_s8, u, v, f_obs, H_IN, W_IN)
                fb_ok = (((f_obs + f_bwd_w).norm(dim=1) < 1.5) & (in_b > 0.5)).float()
                if do_time:
                    torch.cuda.synchronize()
                    timing["fb_occ_extra"].append((time.perf_counter() - tfb0) * 1000.0 / b_sz)
                Rfb, tfb, _, _ = model.solve_pose(Z, u, v, Kb.float(), f_obs, w_full * fb_ok, R0, t0)
                accum["fb_occ_mask"][0].append(Rfb)
                accum["fb_occ_mask"][1].append(tfb)

                f_s4 = F.avg_pool2d(fl_full, 4, 4).float()
                Z_s4 = F.interpolate(Z[:, None], size=(64, 80), mode="bilinear", align_corners=False)[:, 0]
                w_s4 = F.interpolate(w_full[:, None], size=(64, 80), mode="bilinear", align_corners=False)[:, 0]
                v4, u4 = torch.meshgrid(torch.arange(64, device="cuda", dtype=torch.float32) * 4 + 1.5,
                                        torch.arange(80, device="cuda", dtype=torch.float32) * 4 + 1.5, indexing="ij")
                if do_time:
                    torch.cuda.synchronize()
                    ts4 = time.perf_counter()
                Rs4, ts4_v, _, _ = model.solve_pose(Z_s4, u4, v4, Kb.float(), f_s4, w_s4, R0, t0)
                if do_time:
                    torch.cuda.synchronize()
                    timing["stride4_gn"].append((time.perf_counter() - ts4) * 1000.0 / b_sz)
                accum["stride4_gn"][0].append(Rs4)
                accum["stride4_gn"][1].append(ts4_v)

                Z_s1 = F.interpolate(Z[:, None], size=(H_IN, W_IN), mode="bilinear", align_corners=False)[:, 0]
                w_s1 = F.interpolate(w_full[:, None], size=(H_IN, W_IN), mode="bilinear", align_corners=False)[:, 0]
                v1, u1 = torch.meshgrid(torch.arange(H_IN, device="cuda", dtype=torch.float32),
                                        torch.arange(W_IN, device="cuda", dtype=torch.float32), indexing="ij")
                if do_time:
                    torch.cuda.synchronize()
                    ts1 = time.perf_counter()
                Rs1, ts1_v, _, _ = model.solve_pose(Z_s1, u1, v1, Kb.float(), fl_full.float(), w_s1, R0, t0)
                if do_time:
                    torch.cuda.synchronize()
                    timing["fullres_gn"].append((time.perf_counter() - ts1) * 1000.0 / b_sz)
                accum["fullres_gn"][0].append(Rs1)
                accum["fullres_gn"][1].append(ts1_v)

        gt_sel = gt_all[idx]
        ones = torch.ones(n - 1, device="cuda")
        for vname in variants:
            est = chain(torch.cat(accum[vname][0], 0), torch.cat(accum[vname][1], 0), ones).cpu().numpy().astype(np.float64)
            per_var[vname].append(float(eval_with_scale_correction(est, gt_sel)["ate_sim3"]["rmse"]))
        if (si + 1) % 25 == 0:
            print(f"    infer variants {si + 1}/{len(seqs)} ({time.time() - t0_all:.1f}s)", flush=True)

    ref = np.array(per_var["full"])
    out = {}
    for vname, vals in per_var.items():
        arr = np.array(vals)
        p = None if vname == "full" or np.allclose(arr, ref) else float(wilcoxon(arr, ref).pvalue)
        out[vname] = {"n": len(arr), "mean_mm": float(arr.mean()), "median_mm": float(np.median(arr)), "wilcoxon_p_vs_full": p}
    if with_solver_choices:
        out["_timing_ms_median"] = {k: float(np.median(v[2:])) for k, v in timing.items() if len(v) > 2}
    return out


@torch.no_grad()
def audit_labels_and_regions(m_seq, m_pair, seqs: list[str], sample_pairs: int = 5) -> dict:
    h, w = H_IN // STRIDE, W_IN // STRIDE
    conf_seq = np.zeros((3, 3), dtype=np.int64)
    conf_pair = np.zeros((3, 3), dtype=np.int64)

    buckets = ["truly_static", "slow_breathing", "fast_deforming", "instrument"]
    stats_seq = {b: {"Ps": [], "Pd": [], "Pi": [], "es": [], "wnorm": []} for b in buckets}
    stats_pair = {b: {"Ps": [], "Pd": [], "Pi": [], "es": [], "wnorm": []} for b in buckets}

    reg_names = ["static_unocc", "deform_unocc", "instr_unocc", "occluded_or_oob"]
    reg_stats = {r: {"feat_diff": [], "es": [], "flow_res": [], "huber_w": [], "eff_w": []} for r in reg_names}

    for d in seqs:
        gt_all = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
        n_tot = len(gt_all)
        has_abs = os.path.isdir(os.path.join(d, "abs_motion"))
        K = torch.from_numpy(_intrinsics(d)).cuda()
        pids = np.linspace(0, max(0, n_tot - 3), min(sample_pairs, max(1, n_tot - 2)), dtype=int)
        for k in pids:
            p0 = os.path.join(d, "color", f"{k:06d}.png")
            p1 = os.path.join(d, "color", f"{k + 1:06d}.png")
            if not (os.path.exists(p0) and os.path.exists(p1)):
                continue
            z_raw = cv2.imread(os.path.join(d, "depth", f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
            mm_raw = cv2.imread(os.path.join(d, "motion_mask", f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
            im_raw = cv2.imread(os.path.join(d, "object_mask", f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
            if z_raw is None or mm_raw is None:
                continue
            z = cv2.resize(z_raw.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            mm = cv2.resize(mm_raw, (w, h), interpolation=cv2.INTER_NEAREST)
            lab_p = np.where(np.isin(mm, [1, 2, 3]), mm.astype(np.int64) - 1, -100)
            inst = cv2.resize(im_raw, (w, h), interpolation=cv2.INTER_NEAREST) if im_raw is not None else None
            lab_s = abs_static_labels(d, int(k), inst, h, w) if has_abs else None
            if lab_s is None:
                lab_s = lab_p.copy()

            imgs, phs, _ = load_frames([p0, p1])
            img = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
            ph = torch.from_numpy(phs)[:, None].cuda()

            f0 = m_seq.enc(img[:1])
            f1 = m_seq.enc(img[1:])
            fl = m_seq.observed_flow(img[:1], img[1:])
            o_s = m_seq(img[:1], img[1:], K[None], ph[:1], ph[1:], flow=fl)
            u, v = o_s["u"].float(), o_s["v"].float()
            f1w = m_seq._warp(f1, u, v, o_s["f_obs"].float(), H_IN, W_IN)
            fdiff = (f0 - f1w).abs().mean(1)[0].cpu().numpy()

            T_rel = np.linalg.inv(gt_all[k + 1]) @ gt_all[k]
            R_gt = torch.from_numpy(T_rel[:3, :3]).float().cuda()[None]
            t_gt = torch.from_numpy(T_rel[:3, 3]).float().cuda()[None]
            zt = torch.from_numpy(z).float().cuda()[None]
            f_cam_gt, Xp_gt = m_seq.project(zt, u, v, K[None].float(), R_gt, t_gt)
            up_gt, vp_gt = u[None] + f_cam_gt[:, 0], v[None] + f_cam_gt[:, 1]
            in_bounds = (up_gt >= 0) & (up_gt <= W_IN - 1) & (vp_gt >= 0) & (vp_gt <= H_IN - 1)
            cam_ok = ((zt > 0) & (Xp_gt[..., 2] > 0.2 * zt) & (f_cam_gt.norm(dim=1) < 64.0) & in_bounds)[0].cpu().numpy()

            e_res = (o_s["f_obs"] - o_s["f_cam"]).norm(dim=1)[0].cpu().numpy()
            huber = np.where(e_res > 1.5, 1.5 / np.maximum(e_res, 1e-6), 1.0)
            es_s = torch.exp(o_s["logvar"][0].float()).cpu().numpy()
            w_s = o_s["w"][0].float().cpu().numpy()
            w_s_norm = w_s / max(float(w_s.mean()), 1e-8)
            eff_w = (w_s * huber) / max(float((w_s * huber).mean()), 1e-8)
            prob_s = torch.softmax(o_s["logits"][0].float(), 0).cpu().numpy()
            pred_s = prob_s.argmax(0)

            valid_z = z > 0
            masks_reg = {
                "static_unocc": valid_z & cam_ok & (lab_s == 0),
                "deform_unocc": valid_z & cam_ok & (lab_s == 1),
                "instr_unocc": valid_z & cam_ok & (lab_s == 2),
                "occluded_or_oob": valid_z & (~cam_ok),
            }
            for rname, rmask in masks_reg.items():
                if rmask.any():
                    reg_stats[rname]["feat_diff"].append(float(np.median(fdiff[rmask])))
                    reg_stats[rname]["es"].append(float(np.median(es_s[rmask])))
                    reg_stats[rname]["flow_res"].append(float(np.median(e_res[rmask])))
                    reg_stats[rname]["huber_w"].append(float(np.median(huber[rmask])))
                    reg_stats[rname]["eff_w"].append(float(np.median(eff_w[rmask])))

            if m_pair is not None and has_abs:
                o_p = m_pair(img[:1], img[1:], K[None], ph[:1], ph[1:], flow=fl)
                prob_p = torch.softmax(o_p["logits"][0].float(), 0).cpu().numpy()
                pred_p = prob_p.argmax(0)
                es_p = torch.exp(o_p["logvar"][0].float()).cpu().numpy()
                w_p = o_p["w"][0].float().cpu().numpy()
                w_p_norm = w_p / max(float(w_p.mean()), 1e-8)

                val_l = (lab_s >= 0) & (lab_p >= 0) & valid_z
                for c_gt in range(3):
                    for c_pr in range(3):
                        conf_seq[c_gt, c_pr] += int(((lab_s == c_gt) & (pred_s == c_pr) & val_l).sum())
                        conf_pair[c_gt, c_pr] += int(((lab_s == c_gt) & (pred_p == c_pr) & val_l).sum())

                b_masks = {
                    "truly_static": val_l & (lab_s == 0),
                    "slow_breathing": val_l & (lab_p == 0) & (lab_s == 1),
                    "fast_deforming": val_l & (lab_p == 1) & (lab_s == 1),
                    "instrument": val_l & (lab_s == 2),
                }
                for bname, bm in b_masks.items():
                    if bm.any():
                        for st_dict, pr, es_arr, wn_arr in ((stats_seq, prob_s, es_s, w_s_norm),
                                                            (stats_pair, prob_p, es_p, w_p_norm)):
                            st_dict[bname]["Ps"].append(float(pr[0][bm].mean()))
                            st_dict[bname]["Pd"].append(float(pr[1][bm].mean()))
                            st_dict[bname]["Pi"].append(float(pr[2][bm].mean()))
                            st_dict[bname]["es"].append(float(np.median(es_arr[bm])))
                            st_dict[bname]["wnorm"].append(float(wn_arr[bm].mean()))

    def cls_metrics(conf: np.ndarray) -> dict:
        names = ["static", "deforming", "instrument"]
        res = {}
        for i, nm in enumerate(names):
            tp = float(conf[i, i])
            fp = float(conf[:, i].sum() - tp)
            fn = float(conf[i, :].sum() - tp)
            res[nm] = {
                "support_cells": int(conf[i, :].sum()),
                "precision": tp / max(tp + fp, 1.0),
                "recall": tp / max(tp + fn, 1.0),
                "iou": tp / max(tp + fp + fn, 1.0),
            }
        return res

    def summ_bucket(sd: dict) -> dict:
        return {b: {k: float(np.mean(v)) for k, v in d.items() if v} for b, d in sd.items()}

    return {
        "region_audit_r_v2": {r: {k: float(np.median(v)) for k, v in d.items() if v} | {"n_pairs": len(d["es"])}
                              for r, d in reg_stats.items()},
        "seq_label_model": {"class_metrics_vs_seq_gt": cls_metrics(conf_seq), "buckets": summ_bucket(stats_seq)},
        "pair_label_model": {"class_metrics_vs_seq_gt": cls_metrics(conf_pair), "buckets": summ_bucket(stats_pair)},
    }


def main():
    print("[1/5] Loading checkpoints ...", flush=True)
    m_rv2 = load_egolite(ROOT / "results/egolite/r_v2/last.pth")
    m_noph = load_egolite(ROOT / "results/egolite/abl_noph/last.pth")
    m_pair = load_egolite(ROOT / "results/egolite/r_v2_pairlab_full/last.pth")

    dev_seqs = read_list("lists/simtest92.txt")
    core_seqs = read_list("lists/sim_confirmatory265.txt")
    abs_seqs = read_list("lists/abs_test94.txt")
    r2s_seqs = read_list("lists/r2s_test.txt")
    scared_seqs = read_list("lists/scared_official.txt")
    smis_seqs = read_list("lists/stereomis.txt")
    c3vd_seqs = read_list("lists/c3vd_full.txt")

    f_depth = OUT / "r1_2_depth_accuracy.json"
    if not f_depth.is_file():
        print("[2/5] Evaluating depth accuracy (R1.2) ...", flush=True)
        mods = {"r_v2": m_rv2, "abl_noph": m_noph}
        sim_dev = eval_depth_sim_cohort(mods, dev_seqs)
        sim_abs = eval_depth_sim_cohort(mods, abs_seqs)
        sim_r2s = eval_depth_sim_cohort(mods, r2s_seqs)
        real_d = eval_depth_real_scared_c3vd(mods)
        depth_res = {
            mname: {
                "dev_92": sim_dev[mname],
                "abs_94": sim_abs[mname],
                "r2s_40": sim_r2s[mname],
                **real_d[mname],
            }
            for mname in mods
        }
        depth_res["learned_scalars_r_v2"] = {
            "beta": float(m_rv2.beta.item()),
            "depth_bias": float(m_rv2.depth_bias.item()),
            "deform_alpha": float(torch.sigmoid(m_rv2.deform_logit).item()),
        }
        f_depth.write_text(json.dumps(depth_res, indent=2), encoding="utf-8")
        print("  Saved r1_2_depth_accuracy.json", flush=True)

    del m_noph
    torch.cuda.empty_cache()

    f_t6 = OUT / "r1_6_r2_3_table6_expanded.json"
    f_d3 = OUT / "r1_3_design_choices_audit.json"
    f_l1 = OUT / "r1_1_label_learning_audit.json"

    if not (f_t6.is_file() and f_d3.is_file()):
        print("[3/5] Running Table 6 + R1.3 solver/occlusion variants ...", flush=True)
        dev_all = eval_all_infer_variants_cohort(m_rv2, dev_seqs, max_frames=64, with_solver_choices=True)
        abs_all = eval_all_infer_variants_cohort(m_rv2, abs_seqs, max_frames=64, with_solver_choices=True)
        smis_all = eval_all_infer_variants_cohort(m_rv2, smis_seqs, max_frames=64, with_solver_choices=True)
        c3vd_t6 = eval_all_infer_variants_cohort(m_rv2, c3vd_seqs, max_frames=64, with_solver_choices=False)
        scared_t6 = eval_all_infer_variants_cohort(m_rv2, scared_seqs, max_frames=64, with_solver_choices=False)
        r2s_t6 = eval_all_infer_variants_cohort(m_rv2, r2s_seqs, max_frames=64, with_solver_choices=False)
        core_t6 = eval_all_infer_variants_cohort(m_rv2, core_seqs, max_frames=64, with_solver_choices=False)

        t6_keys = ["full", "nogn", "uniform", "nolayer", "novar", "staticonly"]
        t6_res = {
            cohort: {k: d[k] for k in t6_keys}
            for cohort, d in [("dev_92", dev_all), ("core_265", core_t6), ("abs_94", abs_all),
                              ("r2s_40", r2s_t6), ("scared_11", scared_t6), ("stereomis_11", smis_all),
                              ("c3vd_10", c3vd_t6)]
        }
        f_t6.write_text(json.dumps(t6_res, indent=2), encoding="utf-8")
        print("  Saved r1_6_r2_3_table6_expanded.json", flush=True)
    else:
        dev_all = abs_all = smis_all = None

    if not (f_l1.is_file() and f_d3.is_file()):
        print("[4/5] Auditing label learning & region s(p) / occlusion distributions ...", flush=True)
        abs_audit = audit_labels_and_regions(m_rv2, m_pair, abs_seqs)
        r2s_audit = audit_labels_and_regions(m_rv2, m_pair, r2s_seqs)
        dev_reg = audit_labels_and_regions(m_rv2, None, dev_seqs)

        f_l1.write_text(json.dumps({
            "abs_94": {"seq_model": abs_audit["seq_label_model"], "pair_model": abs_audit["pair_label_model"]},
            "r2s_40": {"seq_model": r2s_audit["seq_label_model"], "pair_model": r2s_audit["pair_label_model"]},
        }, indent=2), encoding="utf-8")

        s_keys = ["full", "stride4_gn", "fullres_gn", "oob_mask", "fb_occ_mask", "_timing_ms_median"]
        f_d3.write_text(json.dumps({
            "solver_and_occlusion_variants": {
                "dev_92": {k: dev_all[k] for k in s_keys},
                "abs_94": {k: abs_all[k] for k in s_keys},
                "stereomis_11": {k: smis_all[k] for k in s_keys},
            },
            "region_distributions_r_v2": {
                "dev_92": dev_reg["region_audit_r_v2"],
                "abs_94": abs_audit["region_audit_r_v2"],
            },
        }, indent=2), encoding="utf-8")
        print("  Saved r1_1 and r1_3 audits", flush=True)

    del m_pair, m_rv2
    torch.cuda.empty_cache()

    f_s4 = OUT / "r1_4_seeds_and_stages_real.json"
    if not f_s4.is_file():
        print("[5/5] Evaluating seeds (s0, s1, s2) and 6-stage progression ...", flush=True)

        def quick_cohort_mean(mod, seqs_list):
            vals = []
            for d in seqs_list:
                est, gt, _, _, _ = infer_sequence(mod, d, max_frames=64, batch=4)
                vals.append(float(eval_with_scale_correction(est, gt)["ate_sim3"]["rmse"]))
            return float(np.mean(vals))

        cohorts_map = {
            "dev_92": dev_seqs, "core_265": core_seqs, "abs_94": abs_seqs, "r2s_40": r2s_seqs,
            "scared_11": scared_seqs, "stereomis_11": smis_seqs, "c3vd_10": c3vd_seqs,
        }
        seed_ckpts = {
            "seed0_r_v2": ROOT / "results/egolite/r_v2/last.pth",
            "seed1_r_v2_s1": ROOT / "results/egolite/r_v2_s1/last.pth",
            "seed2_r_v2_s2": ROOT / "results/egolite/r_v2_s2/last.pth",
        }
        seed_res = {}
        for sname, spath in seed_ckpts.items():
            mod = load_egolite(spath)
            seed_res[sname] = {cname: quick_cohort_mean(mod, slist) for cname, slist in cohorts_map.items()}
            del mod
            torch.cuda.empty_cache()

        seed_summary = {
            cname: {
                "mean_mm": float(np.mean([seed_res[s][cname] for s in seed_res])),
                "std_mm": float(np.std([seed_res[s][cname] for s in seed_res], ddof=1)),
                "runs": [round(seed_res[s][cname], 3) for s in seed_res],
            }
            for cname in cohorts_map
        }

        stage_ckpts = [
            ("stage1_v1", ROOT / "results/egolite/v1/best.pth"),
            ("stage2_v2", ROOT / "results/egolite/v2/best.pth"),
            ("stage3_v3abs", ROOT / "results/egolite/v3abs/best.pth"),
            ("stage4_s_big_ft", ROOT / "results/egolite/s_big_ft/last.pth"),
            ("stage5_r_v1", ROOT / "results/egolite/r_v1/last.pth"),
            ("stage6_r_v2", ROOT / "results/egolite/r_v2/last.pth"),
        ]
        stage_res = {}
        for st_name, st_path in stage_ckpts:
            if not st_path.is_file():
                st_path = st_path.with_name("last.pth")
            mod = load_egolite(st_path)
            stage_res[st_name] = {cname: round(quick_cohort_mean(mod, slist), 3) for cname, slist in cohorts_map.items()}
            del mod
            torch.cuda.empty_cache()

        f_s4.write_text(json.dumps({
            "stage6_three_seeds_all_cohorts": seed_summary,
            "six_stage_progression": stage_res,
        }, indent=2), encoding="utf-8")
        print("  Saved r1_4_seeds_and_stages_real.json", flush=True)

    print("ALL DONE!", flush=True)


if __name__ == "__main__":
    main()
