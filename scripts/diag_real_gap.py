"""Why does EgoLite lose on real video? Pose oracles, static-anchor statistics, frame gap.

For each sequence (uniform 64 frames, as in the benchmark):
  pred / identity       predicted trajectory; a camera that never moves
  gt_rot / gt_tdir / gt_tmag / gt_trans   one pose component replaced by ground truth
  p_static, p_deform, p_instr             mean predicted motion-layer probabilities
  w_static_share        share of GN weight on pixels whose predicted layer is static
  scene_frac            median |scene flow| / |observed flow| (how much motion is not camera)
  dense{k}              k-1 extra frames between consecutive benchmark frames, chained,
                        evaluated at the same 64 timestamps
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.metrics import evaluate_trajectory, load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames, select_frame_indices  # noqa: E402
from endosim.lite.data import load_frames  # noqa: E402
from endosim.lite.egolite import build_from_args, chain  # noqa: E402
from endosim.lite.infer import _intrinsics  # noqa: E402

UNIT = 50.0


@torch.no_grad()
def run_pairs(model, imgs, phs, K, batch=16):
    P = len(imgs) - 1
    keys = ("R", "t", "logits", "w", "Z", "f_obs", "f_cam")
    acc = {k: [] for k in keys}
    for a in range(0, P, batch):
        b = min(a + batch, P)
        i0 = torch.from_numpy(imgs[a:b]).permute(0, 3, 1, 2).cuda()
        i1 = torch.from_numpy(imgs[a + 1:b + 1]).permute(0, 3, 1, 2).cuda()
        p0 = torch.from_numpy(phs[a:b])[:, None].cuda()
        p1 = torch.from_numpy(phs[a + 1:b + 1])[:, None].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(i0, i1, K[None].expand(b - a, 3, 3), p0, p1)
        for k in keys:
            acc[k].append(o[k].float())
    return {k: torch.cat(v) for k, v in acc.items()}


def ate_of(R, t, gt):
    T = chain(R, t, torch.ones(R.shape[0], device=R.device)).cpu().numpy().astype(np.float64)
    if not np.isfinite(T).all():
        return float("nan")
    return evaluate_trajectory(T, gt)["ate_sim3"]["rmse"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/egolite/s_big_ft/last.pth")
    ap.add_argument("--list", default="lists/stereomis.txt")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--dense", default="4,16")
    ap.add_argument("--out", default="notes/diag_real_gap_stereomis.json")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_from_args(ck["args"])
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()
    dense_ks = [int(k) for k in args.dense.split(",") if k]
    seqs = [l.strip() for l in open(args.list, encoding="utf-8")
            if l.strip() and not l.startswith("#")]
    if args.n:
        seqs = seqs[: args.n]
    rows = []
    for d in seqs:
        gt_all = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
        idx = select_frame_indices(gt_all, protocol="uniform", max_frames=64)
        gt = gt_all[idx]
        K = torch.from_numpy(_intrinsics(d)).cuda()
        imgs, phs, _ = load_frames(list_color_frames(d, indices=idx))
        o = run_pairs(model, imgs, phs, K)
        Trel = np.linalg.inv(gt[1:]) @ gt[:-1]
        Rg = torch.from_numpy(Trel[:, :3, :3]).float().cuda()
        tg = torch.from_numpy(Trel[:, :3, 3] / UNIT).float().cuda()
        R, t = o["R"], o["t"]
        tn, tgn = t.norm(dim=-1, keepdim=True), tg.norm(dim=-1, keepdim=True)
        prob = torch.softmax(o["logits"], 1)
        hard = prob.argmax(1)
        w = o["w"]
        fo = o["f_obs"].norm(dim=1)
        fs = (o["f_obs"] - o["f_cam"]).norm(dim=1)
        row = {
            "seq": os.path.basename(d.rstrip("/\\")), "n_frames": int(len(gt_all)),
            "gap_med": float(np.median(np.diff(idx))),
            "pred": ate_of(R, t, gt),
            "identity": ate_of(torch.eye(3, device="cuda").expand_as(R), torch.zeros_like(t), gt),
            "gt_rot": ate_of(Rg, t, gt),
            "gt_tdir": ate_of(R, tg / tgn.clamp(min=1e-9) * tn, gt),
            "gt_tmag": ate_of(R, t / tn.clamp(min=1e-9) * tgn, gt),
            "gt_trans": ate_of(R, tg, gt),
            "p_static": float(prob[:, 0].mean()), "p_deform": float(prob[:, 1].mean()),
            "p_instr": float(prob[:, 2].mean()),
            "w_static_share": float((w * (hard == 0)).sum() / w.sum().clamp(min=1e-9)),
            "flow_med_px": float(fo.median()),
            "scene_frac": float((fs / fo.clamp(min=0.5)).median()),
            "depth_med_mm": float(o["Z"].median() * UNIT),
            "t_ratio_med": float((tn / tgn.clamp(min=1e-9)).median()),
        }
        for k in dense_ks:
            if row["gap_med"] < 2:
                continue
            dense = np.unique(np.concatenate(
                [np.linspace(idx[i], idx[i + 1], k + 1).round().astype(int)[:-1]
                 for i in range(len(idx) - 1)] + [idx[-1:]]))
            pos = np.searchsorted(dense, idx)
            di, dp, _ = load_frames(list_color_frames(d, indices=dense))
            od = run_pairs(model, di, dp, K)
            T = chain(od["R"], od["t"], torch.ones(len(dense) - 1, device="cuda"))
            T = T.cpu().numpy().astype(np.float64)[pos]
            row[f"dense{k}"] = (evaluate_trajectory(T, gt)["ate_sim3"]["rmse"]
                                if np.isfinite(T).all() else float("nan"))
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    keys = [k for k in rows[0] if k not in ("seq",)]
    keys = sorted({k for r in rows for k in r if k != "seq"})
    mean = {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys}
    print("MEAN", {k: round(v, 3) for k, v in mean.items()})
    json.dump({"ckpt": args.ckpt, "list": args.list, "mean": mean, "rows": rows},
              open(args.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
