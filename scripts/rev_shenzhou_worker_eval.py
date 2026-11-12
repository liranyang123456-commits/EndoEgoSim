"""Worker-side evaluation script for Shenzhou (GTX 1070 8GB, dedicated GPU).

Runs inside the unpacked `shenzhou_eval` folder where all cohorts are local pre-resized 320x256 sequences.
Uses batch_size=16 (~42 ms/pair on GTX 1070) and writes the 4 revision JSON ledgers:
  1. r1_2_depth_accuracy.json
  2. r1_6_r2_3_table6_expanded.json
  3. r1_1_label_learning_audit.json + r1_3_design_choices_audit.json
  4. r1_4_seeds_and_stages_real.json (including Hamlyn-R2S out-of-generator zero-shot!)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from endosim.eval.metrics import evaluate_trajectory, load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames  # noqa: E402
from endosim.lite.data import H_IN, STRIDE, W_IN, abs_static_labels, load_frames  # noqa: E402
from endosim.lite.egolite import build_from_args, chain  # noqa: E402
from endosim.lite.infer import _intrinsics, infer_sequence  # noqa: E402

OUT = ROOT / "out_json"
OUT.mkdir(parents=True, exist_ok=True)
DEPTH_UNIT_MM = 50.0
BATCH = 16


def eval_sim3(est: np.ndarray, gt: np.ndarray) -> float:
    return float(evaluate_trajectory(est, gt)["ate_sim3"]["rmse"])


def load_egolite(rel_ckpt: str) -> torch.nn.Module:
    p = ROOT / rel_ckpt
    if not p.is_file():
        p = p.with_name("last.pth")
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    m = build_from_args(dict(ck.get("args", {})), pretrained=False)
    m.load_state_dict(ck["model"])
    return m.cuda().eval()


def read_list(tag: str) -> list[str]:
    lp = ROOT / "lists" / f"{tag}.txt"
    if not lp.is_file():
        return []
    return [str(ROOT / ln.strip()) for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]


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
    return agg


@torch.no_grad()
def eval_depth_sim(models: dict[str, torch.nn.Module], seqs: list[str], sample_pairs: int = 8) -> dict:
    recs = {m: [] for m in models}
    for d in seqs:
        paths = list_color_frames(d)
        if len(paths) < 2:
            continue
        pids = np.linspace(0, len(paths) - 2, min(sample_pairs, len(paths) - 1), dtype=int)
        need = sorted(set(int(x) for pid in pids for x in (pid, pid + 1)))
        pos = {orig: k for k, orig in enumerate(need)}
        imgs, phs, _ = load_frames([paths[i] for i in need])
        img_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
        ph_t = torch.from_numpy(phs)[:, None].cuda()
        K = torch.from_numpy(_intrinsics(d)).cuda()
        a0 = img_t[[pos[int(pid)] for pid in pids]]
        a1 = img_t[[pos[int(pid) + 1] for pid in pids]]
        p0 = ph_t[[pos[int(pid)] for pid in pids]]
        p1 = ph_t[[pos[int(pid) + 1] for pid in pids]]
        z_gts = [cv2.imread(os.path.join(d, "depth", f"{int(pid):06d}.png"), cv2.IMREAD_UNCHANGED) for pid in pids]

        for mname, mod in models.items():
            o = mod(a0, a1, K[None].expand(len(pids), 3, 3), p0 if mod.use_ph else None, p1 if mod.use_ph else None)
            z_up = F.interpolate(o["Z"][:, None].float() * DEPTH_UNIT_MM, size=(H_IN, W_IN),
                                 mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            sm = []
            for zp, zg in zip(z_up, z_gts):
                if zg is None:
                    continue
                zg_f = zg.astype(np.float32)
                m = depth_metrics(zp, zg_f, (zg_f > 1.0) & (zg_f < 500.0))
                if m:
                    sm.append(m)
            if sm:
                avg = {k: float(np.mean([x[k] for x in sm])) for k in sm[0]}
                avg["seq_id"] = os.path.basename(d)
                recs[mname].append(avg)
    return {m: aggregate_depth(r) for m, r in recs.items()}


@torch.no_grad()
def eval_depth_real(models: dict[str, torch.nn.Module]) -> dict:
    idx_info = json.loads((ROOT / "real_depth" / "index.json").read_text(encoding="utf-8"))
    out = {m: {} for m in models}
    s_recs = {m: [] for m in models}
    for sid in idx_info["scared"]:
        base = ROOT / "real_depth" / "scared" / sid
        npz = np.load(base / "gt.npz")
        z_gt, K = npz["z_gt"], torch.from_numpy(npz["K"]).cuda()
        valid = np.isfinite(z_gt) & (z_gt > 5.0) & (z_gt < 300.0)
        imgs, phs, _ = load_frames([str(base / "f0.png"), str(base / "f1.png")])
        img_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
        ph_t = torch.from_numpy(phs)[:, None].cuda()
        for mname, mod in models.items():
            o = mod(img_t[:1], img_t[1:], K[None], ph_t[:1] if mod.use_ph else None, ph_t[1:] if mod.use_ph else None)
            zp = F.interpolate(o["Z"][:, None].float() * DEPTH_UNIT_MM, size=(H_IN, W_IN),
                               mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            dm = depth_metrics(zp, z_gt, valid)
            if dm:
                dm["seq_id"] = sid
                s_recs[mname].append(dm)
    for mname in models:
        out[mname]["scared_keyframes_11"] = aggregate_depth(s_recs[mname])
        out[mname]["scared_disjoint_d2_d3_8"] = aggregate_depth(
            [r for r in s_recs[mname] if "_dataset_2_" in r["seq_id"] or "_dataset_3_" in r["seq_id"]])

    c_recs = {m: [] for m in models}
    for pid in idx_info["c3vd"]:
        base = ROOT / "real_depth" / "c3vd" / pid
        npz = np.load(base / "gt.npz")
        z_gt, K = npz["z_gt"], torch.from_numpy(npz["K"]).cuda()
        valid = (z_gt > 1.0) & (z_gt < 99.5)
        imgs, phs, _ = load_frames([str(base / "f0.png"), str(base / "f1.png")])
        img_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
        ph_t = torch.from_numpy(phs)[:, None].cuda()
        for mname, mod in models.items():
            o = mod(img_t[:1], img_t[1:], K[None], ph_t[:1] if mod.use_ph else None, ph_t[1:] if mod.use_ph else None)
            zp = F.interpolate(o["Z"][:, None].float() * DEPTH_UNIT_MM, size=(H_IN, W_IN),
                               mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            dm = depth_metrics(zp, z_gt, valid)
            if dm:
                dm["seq_id"] = pid
                c_recs[mname].append(dm)
    for mname in models:
        out[mname]["c3vd_cecum_t1_a_pairs"] = aggregate_depth(c_recs[mname])
    return out


@torch.no_grad()
def eval_all_infer_variants(model, seqs: list[str], with_solver_choices: bool = False) -> dict:
    variants = ["full", "nogn", "uniform", "nolayer", "novar", "staticonly"]
    if with_solver_choices:
        variants += ["stride4_gn", "fullres_gn", "oob_mask", "fb_occ_mask"]
    per_var = {v: [] for v in variants}
    timing = {v: [] for v in (["stride8_gn", "stride4_gn", "fullres_gn", "fb_occ_extra"] if with_solver_choices else [])}

    for si, d in enumerate(seqs):
        gt_all = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
        paths = list_color_frames(d)
        imgs, phs, _ = load_frames(paths)
        K = torch.from_numpy(_intrinsics(d)).cuda()
        img = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
        ph = torch.from_numpy(phs)[:, None].cuda()
        n = len(paths)
        accum = {v: ([], []) for v in variants}
        b_step = 4 if with_solver_choices else BATCH

        for st in range(0, n - 1, b_step):
            ed = min(n - 1, st + b_step)
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
            accum["nogn"][0].append(R0); accum["nogn"][1].append(t0)
            do_time = with_solver_choices and (si < 6)
            for vname, w_pix in w_map.items():
                if vname == "full" and do_time:
                    torch.cuda.synchronize(); ts0 = time.perf_counter()
                R, t, _, _ = model.solve_pose(Z, u, v, Kb.float(), f_obs, w_pix, R0, t0)
                if vname == "full" and do_time:
                    torch.cuda.synchronize(); timing["stride8_gn"].append((time.perf_counter() - ts0) * 1000.0 / b_sz)
                accum[vname][0].append(R); accum[vname][1].append(t)

            if with_solver_choices:
                w_full = w_map["full"]
                up, vp = u[None] + f_obs[:, 0], v[None] + f_obs[:, 1]
                in_b = ((up >= 0) & (up <= W_IN - 1) & (vp >= 0) & (vp <= H_IN - 1)).float()
                Ro, to, _, _ = model.solve_pose(Z, u, v, Kb.float(), f_obs, w_full * in_b, R0, t0)
                accum["oob_mask"][0].append(Ro); accum["oob_mask"][1].append(to)

                if do_time:
                    torch.cuda.synchronize(); tfb0 = time.perf_counter()
                fl_bwd = model.observed_flow(i1, i0)
                f_bwd_s8 = F.avg_pool2d(fl_bwd, STRIDE, STRIDE).float()
                f_bwd_w = model._warp(f_bwd_s8, u, v, f_obs, H_IN, W_IN)
                fb_ok = (((f_obs + f_bwd_w).norm(dim=1) < 1.5) & (in_b > 0.5)).float()
                if do_time:
                    torch.cuda.synchronize(); timing["fb_occ_extra"].append((time.perf_counter() - tfb0) * 1000.0 / b_sz)
                Rfb, tfb, _, _ = model.solve_pose(Z, u, v, Kb.float(), f_obs, w_full * fb_ok, R0, t0)
                accum["fb_occ_mask"][0].append(Rfb); accum["fb_occ_mask"][1].append(tfb)

                f_s4 = F.avg_pool2d(fl_full, 4, 4).float()
                Z_s4 = F.interpolate(Z[:, None], size=(64, 80), mode="bilinear", align_corners=False)[:, 0]
                w_s4 = F.interpolate(w_full[:, None], size=(64, 80), mode="bilinear", align_corners=False)[:, 0]
                v4, u4 = torch.meshgrid(torch.arange(64, device="cuda", dtype=torch.float32) * 4 + 1.5,
                                        torch.arange(80, device="cuda", dtype=torch.float32) * 4 + 1.5, indexing="ij")
                if do_time:
                    torch.cuda.synchronize(); ts4 = time.perf_counter()
                Rs4, ts4_v, _, _ = model.solve_pose(Z_s4, u4, v4, Kb.float(), f_s4, w_s4, R0, t0)
                if do_time:
                    torch.cuda.synchronize(); timing["stride4_gn"].append((time.perf_counter() - ts4) * 1000.0 / b_sz)
                accum["stride4_gn"][0].append(Rs4); accum["stride4_gn"][1].append(ts4_v)

                Z_s1 = F.interpolate(Z[:, None], size=(H_IN, W_IN), mode="bilinear", align_corners=False)[:, 0]
                w_s1 = F.interpolate(w_full[:, None], size=(H_IN, W_IN), mode="bilinear", align_corners=False)[:, 0]
                v1, u1 = torch.meshgrid(torch.arange(H_IN, device="cuda", dtype=torch.float32),
                                        torch.arange(W_IN, device="cuda", dtype=torch.float32), indexing="ij")
                if do_time:
                    torch.cuda.synchronize(); ts1 = time.perf_counter()
                Rs1, ts1_v, _, _ = model.solve_pose(Z_s1, u1, v1, Kb.float(), fl_full.float(), w_s1, R0, t0)
                if do_time:
                    torch.cuda.synchronize(); timing["fullres_gn"].append((time.perf_counter() - ts1) * 1000.0 / b_sz)
                accum["fullres_gn"][0].append(Rs1); accum["fullres_gn"][1].append(ts1_v)

        ones = torch.ones(n - 1, device="cuda")
        for vname in variants:
            est = chain(torch.cat(accum[vname][0], 0), torch.cat(accum[vname][1], 0), ones).cpu().numpy().astype(np.float64)
            per_var[vname].append(eval_sim3(est, gt_all[:n]))

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
def audit_labels_and_regions(m_seq, m_pair, seqs: list[str], sample_pairs: int = 6) -> dict:
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
        paths = list_color_frames(d)
        n_tot = len(paths)
        has_abs = os.path.isdir(os.path.join(d, "abs_motion"))
        K = torch.from_numpy(_intrinsics(d)).cuda()
        pids = np.linspace(0, max(0, n_tot - 2), min(sample_pairs, max(1, n_tot - 1)), dtype=int)
        for k in pids:
            z_raw = cv2.imread(os.path.join(d, "depth", f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
            mm = cv2.imread(os.path.join(d, "motion_mask", f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
            im = cv2.imread(os.path.join(d, "object_mask", f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
            if z_raw is None or mm is None:
                continue
            z = cv2.resize(z_raw.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            lab_p = np.where(np.isin(mm, [1, 2, 3]), mm.astype(np.int64) - 1, -100)
            lab_s = abs_static_labels(d, int(k), im, h, w) if has_abs else None
            if lab_s is None:
                lab_s = lab_p.copy()

            imgs, phs, _ = load_frames([paths[k], paths[k + 1]])
            img = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
            ph = torch.from_numpy(phs)[:, None].cuda()
            f0 = m_seq.enc(img[:1]); f1 = m_seq.enc(img[1:])
            fl = m_seq.observed_flow(img[:1], img[1:])
            o_s = m_seq(img[:1], img[1:], K[None], ph[:1], ph[1:], flow=fl)
            u, v = o_s["u"].float(), o_s["v"].float()
            fdiff = (f0 - m_seq._warp(f1, u, v, o_s["f_obs"].float(), H_IN, W_IN)).abs().mean(1)[0].cpu().numpy()

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
            tp = float(conf[i, i]); fp = float(conf[:, i].sum() - tp); fn = float(conf[i, :].sum() - tp)
            res[nm] = {"support_cells": int(conf[i, :].sum()),
                       "precision": tp / max(tp + fp, 1.0),
                       "recall": tp / max(tp + fn, 1.0),
                       "iou": tp / max(tp + fp + fn, 1.0)}
        return res

    return {
        "region_audit_r_v2": {r: {k: float(np.median(v)) for k, v in d.items() if v} | {"n_pairs": len(d["es"])}
                              for r, d in reg_stats.items()},
        "seq_label_model": {"class_metrics_vs_seq_gt": cls_metrics(conf_seq),
                            "buckets": {b: {k: float(np.mean(v)) for k, v in d.items() if v} for b, d in stats_seq.items()}},
        "pair_label_model": {"class_metrics_vs_seq_gt": cls_metrics(conf_pair),
                             "buckets": {b: {k: float(np.mean(v)) for k, v in d.items() if v} for b, d in stats_pair.items()}},
    }


def main():
    t_start = time.time()
    print("[Shenzhou] Loading checkpoints ...", flush=True)
    m_rv2 = load_egolite("results/egolite/r_v2/last.pth")
    m_noph = load_egolite("results/egolite/abl_noph/last.pth")
    m_pair = load_egolite("results/egolite/r_v2_pairlab_full/last.pth")

    dev_seqs = read_list("dev_92")
    core_seqs = read_list("core_265")
    abs_seqs = read_list("abs_94")
    r2s_seqs = read_list("r2s_40")
    scared_seqs = read_list("scared_11")
    smis_seqs = read_list("stereomis_11")
    c3vd_seqs = read_list("c3vd_10")
    ham_seqs = read_list("hamlyn_r2s_124")

    print("[1/4] Depth accuracy ...", flush=True)
    mods = {"r_v2": m_rv2, "abl_noph": m_noph}
    sd_dev = eval_depth_sim(mods, dev_seqs)
    sd_abs = eval_depth_sim(mods, abs_seqs)
    sd_r2s = eval_depth_sim(mods, r2s_seqs)
    rd = eval_depth_real(mods)
    depth_res = {m: {"dev_92": sd_dev[m], "abs_94": sd_abs[m], "r2s_40": sd_r2s[m], **rd[m]} for m in mods}
    depth_res["learned_scalars_r_v2"] = {
        "beta": float(m_rv2.beta.item()),
        "depth_bias": float(m_rv2.depth_bias.item()),
        "deform_alpha": float(torch.sigmoid(m_rv2.deform_logit).item()),
    }
    (OUT / "r1_2_depth_accuracy.json").write_text(json.dumps(depth_res, indent=2), encoding="utf-8")
    print(f"  done depth ({time.time() - t_start:.0f}s)", flush=True)
    del m_noph; torch.cuda.empty_cache()

    print("[2/4] Table 6 + Solver/Occlusion variants ...", flush=True)
    dev_all = eval_all_infer_variants(m_rv2, dev_seqs, with_solver_choices=True)
    abs_all = eval_all_infer_variants(m_rv2, abs_seqs, with_solver_choices=True)
    smis_all = eval_all_infer_variants(m_rv2, smis_seqs, with_solver_choices=True)
    c3vd_t6 = eval_all_infer_variants(m_rv2, c3vd_seqs, with_solver_choices=False)
    scared_t6 = eval_all_infer_variants(m_rv2, scared_seqs, with_solver_choices=False)
    r2s_t6 = eval_all_infer_variants(m_rv2, r2s_seqs, with_solver_choices=False)
    core_t6 = eval_all_infer_variants(m_rv2, core_seqs, with_solver_choices=False)

    t6_keys = ["full", "nogn", "uniform", "nolayer", "novar", "staticonly"]
    t6_res = {c: {k: d[k] for k in t6_keys} for c, d in [
        ("dev_92", dev_all), ("core_265", core_t6), ("abs_94", abs_all),
        ("r2s_40", r2s_t6), ("scared_11", scared_t6), ("stereomis_11", smis_all), ("c3vd_10", c3vd_t6)]}
    (OUT / "r1_6_r2_3_table6_expanded.json").write_text(json.dumps(t6_res, indent=2), encoding="utf-8")
    print(f"  done Table 6 ({time.time() - t_start:.0f}s)", flush=True)

    print("[3/4] Label & Region audits ...", flush=True)
    abs_aud = audit_labels_and_regions(m_rv2, m_pair, abs_seqs)
    r2s_aud = audit_labels_and_regions(m_rv2, m_pair, r2s_seqs)
    dev_aud = audit_labels_and_regions(m_rv2, None, dev_seqs)
    (OUT / "r1_1_label_learning_audit.json").write_text(json.dumps({
        "abs_94": {"seq_model": abs_aud["seq_label_model"], "pair_model": abs_aud["pair_label_model"]},
        "r2s_40": {"seq_model": r2s_aud["seq_label_model"], "pair_model": r2s_aud["pair_label_model"]},
    }, indent=2), encoding="utf-8")
    s_keys = ["full", "stride4_gn", "fullres_gn", "oob_mask", "fb_occ_mask", "_timing_ms_median"]
    (OUT / "r1_3_design_choices_audit.json").write_text(json.dumps({
        "solver_and_occlusion_variants": {"dev_92": {k: dev_all[k] for k in s_keys},
                                          "abs_94": {k: abs_all[k] for k in s_keys},
                                          "stereomis_11": {k: smis_all[k] for k in s_keys}},
        "region_distributions_r_v2": {"dev_92": dev_aud["region_audit_r_v2"],
                                      "abs_94": abs_aud["region_audit_r_v2"]},
    }, indent=2), encoding="utf-8")
    print(f"  done Label & Region ({time.time() - t_start:.0f}s)", flush=True)
    del m_pair, m_rv2; torch.cuda.empty_cache()

    print("[4/4] Seeds & 6-stage progression (+ Hamlyn-R2S) ...", flush=True)

    @torch.no_grad()
    def cohort_ates(mod, slist):
        vals = []
        for d in slist:
            est, gt, _, _, _ = infer_sequence(mod, d, max_frames=64, batch=BATCH)
            vals.append(eval_sim3(est, gt))
        return vals

    cohorts_map = {"dev_92": dev_seqs, "core_265": core_seqs, "abs_94": abs_seqs, "r2s_40": r2s_seqs,
                   "scared_11": scared_seqs, "stereomis_11": smis_seqs, "c3vd_10": c3vd_seqs}
    if ham_seqs:
        cohorts_map["hamlyn_r2s_124"] = ham_seqs

    seed_ckpts = {"seed0": "results/egolite/r_v2/last.pth",
                  "seed1": "results/egolite/r_v2_s1/last.pth",
                  "seed2": "results/egolite/r_v2_s2/last.pth"}
    seed_means = {}
    for sn, sp in seed_ckpts.items():
        mod = load_egolite(sp)
        seed_means[sn] = {cn: float(np.mean(cohort_ates(mod, sl))) for cn, sl in cohorts_map.items()}
        del mod; torch.cuda.empty_cache()
    seed_summary = {
        cn: {"mean_mm": float(np.mean([seed_means[s][cn] for s in seed_means])),
             "std_mm": float(np.std([seed_means[s][cn] for s in seed_means], ddof=1)),
             "runs": [round(seed_means[s][cn], 3) for s in seed_means]}
        for cn in cohorts_map
    }

    stage_ckpts = [("stage1_v1", "results/egolite/v1/best.pth"),
                   ("stage2_v2", "results/egolite/v2/best.pth"),
                   ("stage3_v3abs", "results/egolite/v3abs/best.pth"),
                   ("stage4_s_big_ft", "results/egolite/s_big_ft/last.pth"),
                   ("stage5_r_v1", "results/egolite/r_v1/last.pth"),
                   ("stage6_r_v2", "results/egolite/r_v2/last.pth")]
    stage_res = {}
    for stn, stp in stage_ckpts:
        mod = load_egolite(stp)
        stage_res[stn] = {cn: round(float(np.mean(cohort_ates(mod, sl))), 3) for cn, sl in cohorts_map.items()}
        del mod; torch.cuda.empty_cache()

    (OUT / "r1_4_seeds_and_stages_real.json").write_text(json.dumps({
        "stage6_three_seeds_all_cohorts": seed_summary,
        "six_stage_progression": stage_res,
    }, indent=2), encoding="utf-8")
    print(f"ALL SHENZHOU EVALS DONE in {time.time() - t_start:.0f}s!", flush=True)


if __name__ == "__main__":
    main()
