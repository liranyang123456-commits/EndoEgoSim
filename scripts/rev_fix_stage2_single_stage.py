"""True leak-free single-stage from-scratch training (3 seeds) using generic pretrained RAFT-large (P0-2 fix).

Strictly starts from ImageNet-pretrained ResNet-18 (IMAGENET1K_V1) and generic torchvision RAFT-large
(Raft_Large_Weights.DEFAULT, NEVER fine-tuned on any endoscopic or synthetic data; ZERO Stage-1..5 weights).
Precomputes 1,500 fresh 8-frame clip batches (batch=3x8) with generic RAFT-large, trains 3 independent seeds
for 6,000 steps (4 passes) on E:A:R = 2:1:1, and evaluates all 3 seeds with the same generic RAFT-large
on all 8 cohorts.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from endosim.eval.metrics import evaluate_trajectory, load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames, select_frame_indices  # noqa: E402
from endosim.lite.data import H_IN, STRIDE, W_IN, SimClips, load_frames, worker_init  # noqa: E402
from endosim.lite.egolite import build_from_args, chain, pixel_log_ratio, pixel_rho, se3_exp  # noqa: E402
from endosim.lite.infer import _intrinsics  # noqa: E402

OUT_DIR = ROOT / "results" / "revision_vrih"
CKPT_ROOT = OUT_DIR / "ckpts"
DEPTH_UNIT_MM = 50.0


def geodesic(Ra, Rb):
    c = ((Ra @ Rb.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    return torch.acos(c.clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def read_list(rel: str) -> list[str]:
    out = []
    for ln in (ROOT / rel).read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        s = ln.strip()
        if not os.path.isdir(s):
            cand = ROOT / s.replace("\\", "/").split("ego_motiion_Camera/")[-1].lstrip("/")
            if cand.is_dir():
                s = str(cand)
        if os.path.isfile(os.path.join(s, "pose_c2w.txt")):
            out.append(s)
    return out


@torch.no_grad()
def precompute_generic_raft_batches(n_steps: int = 1500, batch_clips: int = 3, clip_len: int = 8) -> list[dict]:
    cache_file = CKPT_ROOT / f"cached_generic_raft_{n_steps}x{batch_clips}x{clip_len}.pt"
    if cache_file.is_file():
        print(f"  [Cache generic RAFT] Loading {cache_file}", flush=True)
        return torch.load(cache_file, map_location="cpu", weights_only=False)
    t0 = time.time()
    flow_net = raft_large(weights=Raft_Large_Weights.DEFAULT).cuda().eval()
    _env_sim = os.environ.get("EGOLITE_SIM_TRAIN_DIR", "")
    s_root = _env_sim if (_env_sim and os.path.isdir(_env_sim)) else (
        r"sim_data\train" if os.path.isdir(r"sim_data\train") else str(ROOT / "sim_data/train")
    )
    roots = [(s_root, 2.0), (str(ROOT / "sim_data_abs/train"), 1.0), (str(ROOT / "sim_data_r2s/train"), 1.0)]
    ds = SimClips(clip=clip_len, length=n_steps * batch_clips, seed=101, roots=roots, stride_p=(0.7, 0.2, 0.1))
    dl = DataLoader(ds, batch_size=batch_clips, shuffle=False, num_workers=4, drop_last=True, worker_init_fn=worker_init)

    cached = []
    for it, b in enumerate(dl):
        B, S = b["img"].shape[:2]
        P = S - 1
        img_f = b["img"].cuda(non_blocking=True)
        i0 = img_f[:, :P].reshape(B * P, 3, H_IN, W_IN)
        i1 = img_f[:, 1:].reshape(B * P, 3, H_IN, W_IN)
        with torch.autocast(device_type="cuda", enabled=False):
            fl = flow_net(i0.float() * 2.0 - 1.0, i1.float() * 2.0 - 1.0, num_flow_updates=12)[-1]
        cached.append({
            "img": (b["img"] * 255.0).round().to(torch.uint8),
            "ph": (b["ph"] * 255.0).round().to(torch.uint8),
            "K": b["K"],
            "depth": b["depth"].to(torch.float16),
            "layer": b["layer"].to(torch.int8),
            "T_wc": b["T_wc"],
            "f_s8": F.avg_pool2d(fl, STRIDE, STRIDE).half().cpu(),
        })
        if (it + 1) % 300 == 0 or it == 0:
            print(f"  [Cache generic RAFT] {it+1}/{n_steps} ({time.time()-t0:.0f}s)", flush=True)
    del flow_net
    torch.cuda.empty_cache()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, cache_file)
    return cached


def forward_s8(model, img0, img1, K, ph0, ph1, f_obs_s8):
    N, _, H, W = img0.shape
    s = model.stride
    f0 = model.enc(img0)
    f1 = model.enc(img1)
    h, w = f0.shape[-2:]
    f_obs = f_obs_s8.float()
    u, v = model._grid(h, w, img0.device, f_obs.dtype)
    f1w = model._warp(f1, u, v, f_obs, H, W).to(f0.dtype)
    p0 = F.avg_pool2d(ph0, s, s)
    p1w = model._warp(F.avg_pool2d(ph1, s, s), u, v, f_obs, H, W)
    parts = [f0, f1w, (f0 - f1w).abs(), f_obs.to(f0.dtype) / s, p0.to(f0.dtype), (p0 - p1w).to(f0.dtype)]
    x = model.pix_extra(model.pix(torch.cat(parts, 1)))
    logits = model.layer_out(x).float()
    logvar = model.logvar_out(x).float().clamp(-6, 6)[:, 0]
    r = model.depth_out(x).float()[:, 0]
    hc = p0.float()[:, 0]
    hc = hc - hc.mean((1, 2), keepdim=True)
    logz = model.beta * hc + r + model.depth_bias
    Z = logz.exp()
    pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1).float()
    with torch.autocast(device_type="cuda", enabled=False):
        xi0 = model.pose_out(pooled) * 0.05
        R0, t0 = se3_exp(xi0)
        prob = torch.softmax(logits, 1)
        share = torch.sigmoid(model.deform_logit.float())
        w_pix = (prob[:, 0] + share * prob[:, 1]) * torch.exp(-logvar)
        R, t, f_cam, Xp, _ = model.solve_pose(
            Z.float(), u.float(), v.float(), K.float(), f_obs, w_pix, R0.float(), t0.float(), return_H=True
        )
        zp = Xp[..., 2]
        rho = zp.flatten(1).median(1).values.clamp(min=1e-3)
    return {"R": R, "t": t, "R0": R0, "t0": t0, "rho": rho, "zp": zp,
            "logits": logits, "logvar": logvar, "Z": Z.float(), "logz": logz.float(),
            "f_obs": f_obs, "f_cam": f_cam, "u": u.float(), "v": v.float()}


def train_seed(seed: int, cached: list[dict], epochs: int = 4, lr: float = 2e-4) -> Path:
    ckpt_p = CKPT_ROOT / f"single_stage_generic_raft_s{seed}" / "last.pth"
    if ckpt_p.is_file():
        return ckpt_p
    ckpt_p.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    cfg = {"no_ph": False, "gn_iters": 3, "abs_depth": True, "flow_net": "large",
           "head_extra": 6, "ctx": True, "train_flow": False}
    model = build_from_args(cfg, pretrained=True).cuda()
    model.flow = nn.Identity()
    torch.cuda.empty_cache()

    w = {"t": 10.0, "r": 10.0, "t0": 2.0, "r0": 2.0, "depth": 1.0, "layer": 0.5,
         "cam": 1.0, "unc": 0.1, "rho": 2.0, "cons": 1.0, "abs": 5.0}
    enc = [p for n, p in model.named_parameters() if n.startswith("enc.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("enc.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc, "lr": lr * 0.5}, {"params": rest, "lr": lr}], weight_decay=1e-4)
    tot_iters = len(cached) * epochs
    warm = 400
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1.0 + math.cos(math.pi * (it - warm) / max(1, tot_iters - warm)))
    )
    model.train()
    t0 = time.time()
    rng_order = np.random.default_rng(seed + 10)
    for ep in range(epochs):
        order = rng_order.permutation(len(cached))
        for idx_b in order:
            b = cached[int(idx_b)]
            img = b["img"].cuda(non_blocking=True).float() / 255.0
            ph = b["ph"].cuda(non_blocking=True).float() / 255.0
            K = b["K"].cuda()
            depth = b["depth"].cuda().float()
            layer = b["layer"].cuda().long()
            Twc = b["T_wc"].cuda()
            f_s8 = b["f_s8"].cuda(non_blocking=True).float()
            B, S = img.shape[:2]
            P = S - 1

            def pairs(x, off=0):
                return x[:, off:off + P].reshape(B * P, *x.shape[2:])

            Kp = K[:, None].expand(B, P, 3, 3).reshape(B * P, 3, 3)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = forward_s8(model, pairs(img), pairs(img, 1), Kp, pairs(ph), pairs(ph, 1), f_s8)

            T_rel = (torch.linalg.inv(Twc[:, 1:]) @ Twc[:, :-1]).reshape(B * P, 4, 4)
            R_gt, t_gt = T_rel[:, :3, :3], T_rel[:, :3, 3]
            z_gt = pairs(depth)
            valid = z_gt > 0
            med = torch.full((z_gt.shape[0],), DEPTH_UNIT_MM, device="cuda")
            t_gt_n, z_gt_n = t_gt / med[:, None], z_gt / med[:, None, None]

            L = {}
            L["t"] = (out["t"] - t_gt_n).abs().sum(-1).mean()
            L["r"] = geodesic(out["R"], R_gt).mean()
            L["t0"] = (out["t0"] - t_gt_n).abs().sum(-1).mean()
            L["r0"] = geodesic(out["R0"], R_gt).mean()
            L["depth"] = (out["logz"] - torch.log(z_gt_n.clamp(min=1e-3))).abs()[valid].mean()
            lab = pairs(layer)
            L["layer"] = F.cross_entropy(out["logits"], lab, weight=torch.tensor([1.0, 1.0, 3.0], device="cuda"), ignore_index=-100)
            f_cam_gt, Xp_gt = model.project(z_gt, out["u"], out["v"], Kp, R_gt, t_gt)
            cam_ok = valid & (Xp_gt[..., 2] > 0.2 * z_gt) & (f_cam_gt.norm(dim=1) < 64.0)
            err_cam = ((out["f_cam"] - f_cam_gt).abs().sum(1) / 8.0).clamp(max=8.0)
            L["cam"] = err_cam[cam_ok].mean() if cam_ok.any() else err_cam.sum() * 0.0
            e = ((out["f_obs"] - f_cam_gt).abs().sum(1) / 8.0).clamp(max=8.0)
            stat = cam_ok & (lab == 0)
            L["unc"] = (e * torch.exp(-out["logvar"]) + out["logvar"])[stat].mean() if stat.any() else out["logvar"].sum() * 0.0

            cur = torch.tensor([bi * P + k for bi in range(B) for k in range(P - 1)], device="cuda", dtype=torch.long)
            sub = {k: out[k][cur] for k in ("zp", "f_cam", "logits", "rho")}
            sub["u"], sub["v"] = out["u"], out["v"]
            log_ratio, inside = pixel_log_ratio(sub, out["Z"][cur + 1])
            cons_ok = inside & stat[cur]
            L["cons"] = log_ratio.abs()[cons_ok].mean() if cons_ok.any() else log_ratio.sum() * 0.0
            L["rho"] = torch.log(pixel_rho(sub, out["Z"][cur + 1])).abs().mean()
            Rb, tb = out["R"].reshape(B, P, 3, 3), out["t"].reshape(B, P, 3)
            ones = torch.ones(P, device="cuda")
            L["abs"] = torch.stack([
                (chain(Rb[i], tb[i], ones)[1:, :3, 3] - Twc[i, 1:, :3, 3] / DEPTH_UNIT_MM).abs().sum(-1).mean()
                for i in range(B)
            ]).mean()

            tot = sum(w[k] * v for k, v in L.items())
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(tot):
                tot.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
    print(f"  [Trained generic-RAFT single-stage seed {seed}] {tot_iters} steps in {time.time()-t0:.1f}s", flush=True)
    torch.save({"model": model.state_dict(), "args": cfg, "seed": seed, "flow_weights": "Raft_Large_Weights.DEFAULT"}, ckpt_p)
    del model
    torch.cuda.empty_cache()
    return ckpt_p


@torch.no_grad()
def evaluate_seeds(ckpts: dict[str, Path]) -> dict:
    flow_net = raft_large(weights=Raft_Large_Weights.DEFAULT).cuda().eval()
    models = {}
    for name, cp in ckpts.items():
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        m = build_from_args(ck["args"], pretrained=False)
        m.flow = nn.Identity()
        m.load_state_dict(ck["model"], strict=True)
        models[name] = m.cuda().eval()

    cohorts = {
        "dev_92": read_list("lists/simtest92.txt"),
        "core_265": read_list("lists/sim_confirmatory265.txt"),
        "abs_94": read_list("lists/abs_test94.txt"),
        "r2s_40": read_list("lists/r2s_test.txt"),
        "scared_11": read_list("lists/scared_official.txt"),
        "stereomis_11": read_list("lists/stereomis.txt"),
        "c3vd_10": read_list("lists/c3vd_full.txt"),
    }
    _ham_dir = os.environ.get("EGOLITE_HAMLYN_R2S_DIR") or (
        str(ROOT / "sim_data_r2s_ham2" / "train")
        if (ROOT / "sim_data_r2s_ham2" / "train").is_dir()
        else r"sim_data_r2s_ham2\train"
    )
    ham = sorted(glob.glob(os.path.join(_ham_dir, "seq_*")))
    ham = [d for d in ham if os.path.isfile(os.path.join(d, "pose_c2w.txt"))]
    if ham:
        cohorts["hamlyn_r2s_124"] = ham

    per = {cn: {vn: [] for vn in models} for cn in cohorts}
    for cn, seqs in cohorts.items():
        for d in seqs:
            gt_all = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
            idx = select_frame_indices(gt_all, "uniform", max_frames=64)
            paths = list_color_frames(d, indices=idx)
            gt_sel = gt_all[idx]
            imgs, phs, _ = load_frames(paths)
            K = torch.from_numpy(_intrinsics(d)).cuda()
            img = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
            ph = torch.from_numpy(phs)[:, None].cuda()
            n = len(paths)
            f_chunks = []
            for st in range(0, n - 1, 8):
                ed = min(n - 1, st + 8)
                with torch.autocast(device_type="cuda", enabled=False):
                    fl = flow_net(img[st:ed].float() * 2.0 - 1.0, img[st + 1:ed + 1].float() * 2.0 - 1.0, num_flow_updates=12)[-1]
                f_chunks.append(F.avg_pool2d(fl, STRIDE, STRIDE).float())
            f_s8 = torch.cat(f_chunks, 0)
            ones = torch.ones(n - 1, device="cuda")

            for vn, mod in models.items():
                Rs, ts = [], []
                for st in range(0, n - 1, 32):
                    ed = min(n - 1, st + 32)
                    o = forward_s8(mod, img[st:ed], img[st + 1:ed + 1], K[None].expand(ed - st, 3, 3),
                                   ph[st:ed], ph[st + 1:ed + 1], f_s8[st:ed])
                    Rs.append(o["R"].float())
                    ts.append(o["t"].float())
                est = chain(torch.cat(Rs, 0), torch.cat(ts, 0), ones).cpu().numpy().astype(np.float64)
                per[cn][vn].append(float(evaluate_trajectory(est, gt_sel)["ate_sim3"]["rmse"]))
        vals = [float(np.mean(per[cn][vn])) for vn in models]
        print(f"  [Eval {cn:14s}] seeds={[round(x, 2) for x in vals]} mean={np.mean(vals):.2f}+/-{np.std(vals, ddof=1):.2f}mm", flush=True)

    out = {
        "protocol": "True leak-free single-stage from scratch: ImageNet ResNet-18 + frozen generic torchvision Raft_Large_Weights.DEFAULT (ZERO endoscopic/stage-1..5 weights), 6,000 steps (4 passes over 1,500 clips, batch 3x8) on E:A:R=2:1:1",
        "seeds": {vn: {cn: {"mean_mm": float(np.mean(per[cn][vn])), "median_mm": float(np.median(per[cn][vn]))} for cn in cohorts} for vn in models},
        "summary_3seeds": {
            cn: {
                "mean_mm": float(np.mean([np.mean(per[cn][vn]) for vn in models])),
                "std_mm": float(np.std([np.mean(per[cn][vn]) for vn in models], ddof=1)),
                "runs": [round(float(np.mean(per[cn][vn])), 3) for vn in models],
            }
            for cn in cohorts
        },
    }
    out_p = OUT_DIR / "r1_4_single_stage_generic_raft.json"
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved", out_p, flush=True)
    return out


def main():
    cached = precompute_generic_raft_batches(n_steps=1500, batch_clips=3, clip_len=8)
    ckpts = {f"seed_{s}": train_seed(s, cached, epochs=4, lr=2e-4) for s in (0, 1, 2)}
    del cached
    torch.cuda.empty_cache()
    evaluate_seeds(ckpts)


if __name__ == "__main__":
    main()
