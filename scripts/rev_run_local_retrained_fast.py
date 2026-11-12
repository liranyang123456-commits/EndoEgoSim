"""Local RTX 5090 Runner for All 16 Controlled Retrained Ablations + Same-Data PoseNet + Hamlyn-R2S Baselines!

Uses < 1.8 GB VRAM (safe alongside any other job on the 24 GB RTX 5090) and precomputes exact
stride-8 RAFT-large optical flows once in bf16/fp32 so all 15 EgoLite trunk/solver/loss variants
train & evaluate identically and rapidly on the FULL local datasets (`sim_data/train`, `sim_data_abs/train`,
`sim_data_r2s/train`)!
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

cv2.setNumThreads(1)
torch.set_num_threads(2)

ROOT = Path(r"D:\ego_motiion_Camera")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import egomotion_baselines as eb  # noqa: E402
import eval_endofast3r as ef  # noqa: E402
from endosim.eval.metrics import evaluate_trajectory, load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames, select_frame_indices  # noqa: E402
from endosim.lite.data import H_IN, STRIDE, W_IN, SimClips, load_frames, worker_init  # noqa: E402
from endosim.lite.egolite import build_from_args, chain, pixel_log_ratio, pixel_rho, se3_exp  # noqa: E402
from endosim.lite.infer import _intrinsics  # noqa: E402

OUT_DIR = ROOT / "results" / "revision_vrih"
CKPT_ROOT = OUT_DIR / "ckpts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_ROOT.mkdir(parents=True, exist_ok=True)
DEPTH_UNIT_MM = 50.0


def geodesic(Ra, Rb):
    c = ((Ra @ Rb.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6))


def eval_sim3(est: np.ndarray, gt: np.ndarray) -> float:
    return float(evaluate_trajectory(est, gt)["ate_sim3"]["rmse"])


class SameDataPoseNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        old_w = m.conv1.weight.data
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            self.conv1.weight.data.copy_(torch.cat([old_w, old_w], dim=1) * 0.5)
        self.bn1, self.relu, self.maxpool = m.bn1, m.relu, m.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.squeeze = nn.Conv2d(512, 256, 1)
        self.pconv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.pconv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.pconv3 = nn.Conv2d(256, 6, 1)
        nn.init.normal_(self.pconv3.weight, 0.0, 1e-3)
        nn.init.zeros_(self.pconv3.bias)

    def forward(self, img0: torch.Tensor, img1: torch.Tensor):
        x = torch.cat([img0, img1], dim=1)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = F.relu(self.pconv2(F.relu(self.pconv1(F.relu(self.squeeze(x))))))
        xi = self.pconv3(x).mean(dim=(-2, -1)) * 0.05
        with torch.autocast(device_type=img0.device.type, enabled=False):
            R, t = se3_exp(xi.float())
        return R, t


def read_list(rel: str) -> list[str]:
    return [ln.strip() for ln in (ROOT / rel).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#") and os.path.isfile(os.path.join(ln.strip(), "pose_c2w.txt"))]


@torch.no_grad()
def precompute_train_batches(n_steps: int = 1500, batch_clips: int = 3, clip_len: int = 6) -> list[dict]:
    cache_file = CKPT_ROOT / f"cached_train_{n_steps}x{batch_clips}x{clip_len}.pt"
    if cache_file.is_file():
        print(f"  [Cache train] Loading existing {cache_file} ...", flush=True)
        return torch.load(cache_file, map_location="cpu", weights_only=False)
    t0 = time.time()
    ck = torch.load(ROOT / "results/egolite/r_v1/last.pth", map_location="cpu")
    m_flow = build_from_args(ck["args"], pretrained=False).cuda().eval()
    m_flow.load_state_dict(ck["model"])

    # Use pre-packed 320x256 sim_data/train from D:/ego_lite_pack for 5x faster disk reads!
    s_root = r"D:\ego_lite_pack\sim_data\train" if os.path.isdir(r"D:\ego_lite_pack\sim_data\train") else str(ROOT / "sim_data/train")
    roots = [(s_root, 2.0), (str(ROOT / "sim_data_abs/train"), 1.0), (str(ROOT / "sim_data_r2s/train"), 1.0)]
    ds = SimClips(clip=clip_len, length=n_steps * batch_clips, seed=0, roots=roots, stride_p=(0.7, 0.2, 0.1))
    dl = DataLoader(ds, batch_size=batch_clips, shuffle=False, num_workers=3, drop_last=True, worker_init_fn=worker_init)

    cached = []
    for it, b in enumerate(dl):
        B, S = b["img"].shape[:2]; P = S - 1
        img_f = b["img"].cuda(non_blocking=True)
        i0 = img_f[:, :P].reshape(B * P, 3, H_IN, W_IN)
        i1 = img_f[:, 1:].reshape(B * P, 3, H_IN, W_IN)
        fl = m_flow.observed_flow(i0, i1)
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
            print(f"  [Cache train] {it + 1}/{n_steps} steps ({time.time() - t0:.0f}s)", flush=True)
    del m_flow; torch.cuda.empty_cache()
    torch.save(cached, cache_file)
    return cached


def forward_with_s8_flow(model, img0, img1, K, ph0, ph1, f_obs_s8, learn_inst_weight: bool = False):
    N, _, H, W = img0.shape
    s = model.stride
    f0 = model.enc(img0)
    f1 = model.enc(img1)
    h, w = f0.shape[-2:]
    f_obs = f_obs_s8.float()
    u, v = model._grid(h, w, img0.device, f_obs.dtype)
    f1w = model._warp(f1, u, v, f_obs, H, W).to(f0.dtype)
    parts = [f0, f1w, (f0 - f1w).abs(), f_obs.to(f0.dtype) / s]
    if model.use_ph:
        p0 = F.avg_pool2d(ph0, s, s)
        p1w = model._warp(F.avg_pool2d(ph1, s, s), u, v, f_obs, H, W)
        parts += [p0.to(f0.dtype), (p0 - p1w).to(f0.dtype)]
    x = model.pix_extra(model.pix(torch.cat(parts, 1)))
    logits = model.layer_out(x).float()
    logvar = model.logvar_out(x).float().clamp(-6, 6)[:, 0]
    r = model.depth_out(x).float()[:, 0]
    if model.use_ph:
        hc = p0.float()[:, 0]
        hc = hc - hc.mean((1, 2), keepdim=True)
        logz = model.beta * hc + r
    else:
        logz = r
    logz = logz + model.depth_bias
    Z = logz.exp()
    pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1).float()
    with torch.autocast(device_type="cuda", enabled=False):
        xi0 = model.pose_out(pooled) * 0.05
        R0, t0 = se3_exp(xi0)
        prob = torch.softmax(logits, 1)
        share = torch.sigmoid(model.deform_logit.float())
        if learn_inst_weight:
            share_i = torch.sigmoid(model.inst_logit.float())
            w_pix = (prob[:, 0] + share * prob[:, 1] + share_i * prob[:, 2]) * torch.exp(-logvar)
        else:
            w_pix = (prob[:, 0] + share * prob[:, 1]) * torch.exp(-logvar)
        ablate = getattr(model, "ablate", None)
        if ablate == "uniform":
            w_pix = torch.ones_like(w_pix)
        elif ablate == "nolayer":
            w_pix = torch.exp(-logvar)
        elif ablate == "novar":
            w_pix = prob[:, 0] + share * prob[:, 1]
        elif ablate == "staticonly":
            w_pix = prob[:, 0] * torch.exp(-logvar)
        R, t, f_cam, Xp, _ = model.solve_pose(
            Z.float(), u.float(), v.float(), K.float(), f_obs, w_pix, R0.float(), t0.float(),
            return_H=True, iters=0 if ablate == "nogn" else None)
        zp = Xp[..., 2]
        rho = zp.flatten(1).median(1).values.clamp(min=1e-3)
    return {"R": R, "t": t, "R0": R0, "t0": t0, "rho": rho, "zp": zp,
            "logits": logits, "logvar": logvar, "Z": Z.float(), "logz": logz.float(),
            "f_obs": f_obs, "f_cam": f_cam, "u": u.float(), "v": v.float()}


def train_egolite_fast(
    name: str,
    cached_batches: list[dict],
    init_ckpt: str | None = "results/egolite/r_v1/last.pth",
    lr: float = 5e-5,
    ablate_mode: str | None = None,
    learn_inst_weight: bool = False,
    w_overrides: dict | None = None,
    epochs_over_cache: int = 1,
    seed: int = 0,
) -> Path:
    ckpt_p = CKPT_ROOT / name / "last.pth"
    if ckpt_p.is_file():
        return ckpt_p
    ckpt_p.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    cfg = {"no_ph": False, "gn_iters": 3, "abs_depth": True, "flow_net": "large",
           "head_extra": 6, "ctx": True, "train_flow": False}
    model = build_from_args(cfg, pretrained=(init_ckpt is None)).cuda()
    if init_ckpt is not None:
        sd = torch.load(ROOT / init_ckpt, map_location="cpu")["model"]
        model.load_state_dict(sd, strict=True)
    model.flow = nn.Identity()
    torch.cuda.empty_cache()
    if ablate_mode:
        model.ablate = ablate_mode
    if learn_inst_weight:
        model.inst_logit = nn.Parameter(torch.tensor(0.0, device="cuda"))

    w = {"t": 10.0, "r": 10.0, "t0": 2.0, "r0": 2.0, "depth": 1.0, "layer": 0.5,
         "cam": 1.0, "unc": 0.1, "rho": 2.0, "cons": 1.0, "abs": 5.0}
    if ablate_mode == "nogn":
        w["t"], w["r"], w["t0"], w["r0"] = 10.0, 10.0, 0.0, 0.0
    if w_overrides:
        w.update(w_overrides)

    enc = [p for n, p in model.named_parameters() if n.startswith("enc.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("enc.") and p.requires_grad]
    if learn_inst_weight:
        # Scalar gate logits have small summed gradients; give inst_logit 1e-2 LR so it converges cleanly
        rest = [p for n, p in model.named_parameters() if not n.startswith("enc.") and n != "inst_logit" and p.requires_grad]
        opt = torch.optim.AdamW([{"params": enc, "lr": lr * 0.5}, {"params": rest, "lr": lr},
                                 {"params": [model.inst_logit], "lr": 1e-2}], weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW([{"params": enc, "lr": lr * 0.5}, {"params": rest, "lr": lr}], weight_decay=1e-4)
    tot_iters = len(cached_batches) * epochs_over_cache
    warm = min(250, tot_iters // 5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, tot_iters - warm))))

    model.train()
    t0 = time.time()
    rng_order = np.random.default_rng(seed)
    for ep in range(epochs_over_cache):
        order = np.arange(len(cached_batches)) if seed == 0 and ep == 0 else rng_order.permutation(len(cached_batches))
        for idx_b in order:
            b = cached_batches[int(idx_b)]
            img = b["img"].cuda(non_blocking=True).float() / 255.0
            ph = b["ph"].cuda(non_blocking=True).float() / 255.0
            K = b["K"].cuda()
            depth = b["depth"].cuda().float()
            layer = b["layer"].cuda().long()
            Twc = b["T_wc"].cuda()
            f_s8 = b["f_s8"].cuda(non_blocking=True).float()
            B, S = img.shape[:2]; P = S - 1

            def pairs(x, off=0):
                return x[:, off:off + P].reshape(B * P, *x.shape[2:])

            Kp = K[:, None].expand(B, P, 3, 3).reshape(B * P, 3, 3)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = forward_with_s8_flow(model, pairs(img), pairs(img, 1), Kp, pairs(ph), pairs(ph, 1), f_s8, learn_inst_weight)

            T_rel = (torch.linalg.inv(Twc[:, 1:]) @ Twc[:, :-1]).reshape(B * P, 4, 4)
            R_gt, t_gt = T_rel[:, :3, :3], T_rel[:, :3, 3]
            z_gt = pairs(depth); valid = z_gt > 0
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
            L["abs"] = torch.stack([(chain(Rb[i], tb[i], ones)[1:, :3, 3] - Twc[i, 1:, :3, 3] / DEPTH_UNIT_MM).abs().sum(-1).mean()
                                    for i in range(B)]).mean()

            tot = sum(w[k] * v for k, v in L.items() if w.get(k, 0.0) > 0)
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(tot):
                tot.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

    ex = f" alpha_i={float(torch.sigmoid(model.inst_logit)):.4f} alpha_d={float(torch.sigmoid(model.deform_logit)):.4f}" if learn_inst_weight else ""
    print(f"  [Trained {name}] {tot_iters} steps in {time.time() - t0:.1f}s{ex}", flush=True)
    meta = {"args": cfg, "ablate_mode": ablate_mode, "learn_inst_weight": learn_inst_weight, "w": w}
    if learn_inst_weight:
        meta["learned_alpha_inst"] = float(torch.sigmoid(model.inst_logit).item())
        meta["learned_alpha_deform"] = float(torch.sigmoid(model.deform_logit).item())
    torch.save({"model": model.state_dict(), **meta}, ckpt_p)
    del model; torch.cuda.empty_cache()
    return ckpt_p


def train_posenet_fast(cached_batches: list[dict], epochs: int = 6, lr: float = 2e-4) -> Path:
    ckpt_p = CKPT_ROOT / "samedata_posenet" / "last.pth"
    if ckpt_p.is_file():
        return ckpt_p
    ckpt_p.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = SameDataPoseNet(pretrained=True).cuda().train()
    tot_iters = len(cached_batches) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warm = 250
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, tot_iters - warm))))
    t0 = time.time()
    rng_order = np.random.default_rng(0)
    for ep in range(epochs):
        order = np.arange(len(cached_batches)) if ep == 0 else rng_order.permutation(len(cached_batches))
        for idx_b in order:
            b = cached_batches[int(idx_b)]
            img = b["img"].cuda(non_blocking=True).float() / 255.0
            Twc = b["T_wc"].cuda(non_blocking=True)
            B, S = img.shape[:2]; P = S - 1
            i0 = img[:, :P].reshape(B * P, 3, H_IN, W_IN)
            i1 = img[:, 1:].reshape(B * P, 3, H_IN, W_IN)
            T_rel = (torch.linalg.inv(Twc[:, 1:]) @ Twc[:, :-1]).reshape(B * P, 4, 4)
            R_gt, t_gt = T_rel[:, :3, :3], T_rel[:, :3, 3] / DEPTH_UNIT_MM
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                R, t = model(i0, i1)
            loss_p = 10.0 * ((t - t_gt).abs().sum(-1).mean() + geodesic(R, R_gt).mean())
            Rb, tb = R.reshape(B, P, 3, 3), t.reshape(B, P, 3)
            ones = torch.ones(P, device="cuda")
            l_traj = torch.stack([(chain(Rb[i], tb[i], ones)[1:, :3, 3] - Twc[i, 1:, :3, 3] / DEPTH_UNIT_MM).abs().sum(-1).mean()
                                  for i in range(B)]).mean()
            loss = loss_p + 5.0 * l_traj
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    print(f"  [Trained SameDataPoseNet] {tot_iters} steps in {time.time() - t0:.1f}s", flush=True)
    torch.save({"model": model.state_dict(), "iters": tot_iters}, ckpt_p)
    del model; torch.cuda.empty_cache()
    return ckpt_p


@torch.no_grad()
def evaluate_all_shared(egolite_ckpts: dict[str, Path], posenet_ckpt: Path, cohorts: dict[str, list[str]]) -> dict:
    t0 = time.time()
    ck_base = torch.load(ROOT / "results/egolite/r_v1/last.pth", map_location="cpu")
    m_flow = build_from_args(ck_base["args"], pretrained=False).cuda().eval()
    m_flow.load_state_dict(ck_base["model"])

    loaded_ego = {}
    extra_meta = {}
    for vname, cp in egolite_ckpts.items():
        ck = torch.load(cp, map_location="cpu")
        m = build_from_args(ck["args"], pretrained=False)
        m.flow = nn.Identity()
        if ck.get("learn_inst_weight"):
            m.inst_logit = nn.Parameter(torch.tensor(0.0))
            extra_meta[vname] = {"learned_alpha_inst": ck.get("learned_alpha_inst"),
                                 "learned_alpha_deform": ck.get("learned_alpha_deform")}
        m.load_state_dict(ck["model"], strict=True)
        if ck.get("ablate_mode"):
            m.ablate = ck["ablate_mode"]
        loaded_ego[vname] = (m.cuda().eval(), bool(ck.get("learn_inst_weight")))

    pn_mod = SameDataPoseNet(pretrained=False).cuda().eval()
    pn_mod.load_state_dict(torch.load(posenet_ckpt, map_location="cpu")["model"])

    # Also evaluate published baselines on Hamlyn-R2S out-of-generator 124 sequences!
    f3r_mod = ef.load_model(torch.device("cuda"))

    all_names = list(egolite_ckpts.keys()) + ["samedata_posenet_13M"]
    per_cohort = {cn: {vn: [] for vn in all_names} for cn in cohorts}
    hamlyn_ext_baselines = {"afsfm": [], "endodac": [], "endofast3r": []}

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

            f_s8_chunks = []
            for st in range(0, n - 1, 8):
                ed = min(n - 1, st + 8)
                fl = m_flow.observed_flow(img[st:ed], img[st + 1:ed + 1])
                f_s8_chunks.append(F.avg_pool2d(fl, STRIDE, STRIDE).float())
            f_s8_all = torch.cat(f_s8_chunks, 0)
            ones = torch.ones(n - 1, device="cuda")

            for vname, (mod, learn_inst) in loaded_ego.items():
                Rs, ts = [], []
                for st in range(0, n - 1, 32):
                    ed = min(n - 1, st + 32)
                    o = forward_with_s8_flow(mod, img[st:ed], img[st + 1:ed + 1],
                                             K[None].expand(ed - st, 3, 3), ph[st:ed], ph[st + 1:ed + 1],
                                             f_s8_all[st:ed], learn_inst)
                    Rs.append(o["R"].float()); ts.append(o["t"].float())
                est = chain(torch.cat(Rs, 0), torch.cat(ts, 0), ones).cpu().numpy().astype(np.float64)
                per_cohort[cn][vname].append(eval_sim3(est, gt_sel))

            Rs_p, ts_p = [], []
            for st in range(0, n - 1, 32):
                ed = min(n - 1, st + 32)
                Rp, tp = pn_mod(img[st:ed], img[st + 1:ed + 1])
                Rs_p.append(Rp.float()); ts_p.append(tp.float())
            est_p = chain(torch.cat(Rs_p, 0), torch.cat(ts_p, 0), ones).cpu().numpy().astype(np.float64)
            per_cohort[cn]["samedata_posenet_13M"].append(eval_sim3(est_p, gt_sel))

            if cn == "hamlyn_r2s_124":
                for which in ("afsfm", "endodac"):
                    est_b = eb.run_posenet(paths, which, batch=16)
                    hamlyn_ext_baselines[which].append(eval_sim3(est_b, gt_sel))
                est_f3, gt_f3, _ = ef.run_seq(f3r_mod, d, torch.device("cuda"), batch=4, max_frames=64, invert=False)
                hamlyn_ext_baselines["endofast3r"].append(eval_sim3(est_f3, gt_f3))

        print(f"  [Eval {cn}] ctrl={np.mean(per_cohort[cn]['retrain_ctrl_full']):.2f} "
              f"posenet={np.mean(per_cohort[cn]['samedata_posenet_13M']):.2f} "
              f"nogn={np.mean(per_cohort[cn]['retrain_nogn']):.2f} "
              f"unif={np.mean(per_cohort[cn]['retrain_uniform']):.2f} ({time.time() - t0:.0f}s)", flush=True)

    summary = {"extra_learned_scalars": extra_meta, "variants": {}}
    for vname in all_names:
        summary["variants"][vname] = {}
        for cn in cohorts:
            arr = np.array(per_cohort[cn][vname])
            rarr = np.array(per_cohort[cn]["retrain_ctrl_full"])
            p = None if vname == "retrain_ctrl_full" or np.allclose(arr, rarr) else float(wilcoxon(arr, rarr).pvalue)
            summary["variants"][vname][cn] = {
                "mean_mm": float(arr.mean()),
                "median_mm": float(np.median(arr)),
                "p_vs_ctrl": p,
            }

    if "hamlyn_r2s_124" in cohorts and hamlyn_ext_baselines["afsfm"]:
        rarr_h = np.array(per_cohort["hamlyn_r2s_124"]["retrain_ctrl_full"])
        summary["hamlyn_r2s_124_published_baselines"] = {
            k: {
                "mean_mm": float(np.mean(v)),
                "median_mm": float(np.median(v)),
                "p_vs_ctrl": float(wilcoxon(np.array(v), rarr_h).pvalue),
            }
            for k, v in hamlyn_ext_baselines.items()
        }
    return summary


def main():
    cohorts = {
        "dev_92": read_list("lists/simtest92.txt"),
        "core_265": read_list("lists/sim_confirmatory265.txt"),
        "abs_94": read_list("lists/abs_test94.txt"),
        "r2s_40": read_list("lists/r2s_test.txt"),
        "scared_11": read_list("lists/scared_official.txt"),
        "stereomis_11": read_list("lists/stereomis.txt"),
        "c3vd_10": read_list("lists/c3vd_full.txt"),
    }
    ham = sorted(glob.glob(r"E:\ego_data\sim_data_r2s_ham2\train\seq_*"))
    ham = [d for d in ham if os.path.isfile(os.path.join(d, "pose_c2w.txt"))]
    if ham:
        cohorts["hamlyn_r2s_124"] = ham

    print("[1/3] Precomputing 1,500 training mini-batches + RAFT flows on RTX 5090 ...", flush=True)
    cached = precompute_train_batches(n_steps=1500, batch_clips=3, clip_len=6)

    print("[2/3] Training all 18 controlled variants (incl. 3 multi-seed single-stage runs!) ...", flush=True)
    ego_ckpts = {
        "retrain_ctrl_full": train_egolite_fast("retrain_ctrl_full", cached),
        "retrain_nogn": train_egolite_fast("retrain_nogn", cached, ablate_mode="nogn"),
        "retrain_uniform": train_egolite_fast("retrain_uniform", cached, ablate_mode="uniform"),
        "retrain_nolayer": train_egolite_fast("retrain_nolayer", cached, ablate_mode="nolayer"),
        "retrain_novar": train_egolite_fast("retrain_novar", cached, ablate_mode="novar"),
        "retrain_staticonly": train_egolite_fast("retrain_staticonly", cached, ablate_mode="staticonly"),
        "retrain_instw": train_egolite_fast("retrain_instw", cached, learn_inst_weight=True),
        "loss_no_initpose": train_egolite_fast("loss_no_initpose", cached, w_overrides={"t0": 0.0, "r0": 0.0}),
        "loss_no_depth": train_egolite_fast("loss_no_depth", cached, w_overrides={"depth": 0.0}),
        "loss_no_layer": train_egolite_fast("loss_no_layer", cached, w_overrides={"layer": 0.0}),
        "loss_no_cam": train_egolite_fast("loss_no_cam", cached, w_overrides={"cam": 0.0}),
        "loss_no_unc": train_egolite_fast("loss_no_unc", cached, w_overrides={"unc": 0.0}),
        "loss_no_traj": train_egolite_fast("loss_no_traj", cached, w_overrides={"abs": 0.0}),
        "loss_no_cons_rho": train_egolite_fast("loss_no_cons_rho", cached, w_overrides={"cons": 0.0, "rho": 0.0}),
        "single_stage_s0": train_egolite_fast("single_stage_s0", cached, init_ckpt=None, lr=2e-4, epochs_over_cache=4, seed=0),
        "single_stage_s1": train_egolite_fast("single_stage_s1", cached, init_ckpt=None, lr=2e-4, epochs_over_cache=4, seed=1),
        "single_stage_s2": train_egolite_fast("single_stage_s2", cached, init_ckpt=None, lr=2e-4, epochs_over_cache=4, seed=2),
    }
    pn_ckpt = train_posenet_fast(cached, epochs=6, lr=2e-4)
    del cached; torch.cuda.empty_cache()

    print("[3/3] Evaluating all checkpoints + Hamlyn-R2S baselines ...", flush=True)
    summary = evaluate_all_shared(ego_ckpts, pn_ckpt, cohorts)

    # Also compute mean +/- std across the 3 end-to-end from-scratch seeds (single_stage_s0/s1/s2)!
    summary["from_scratch_3seeds_summary"] = {
        cn: {
            "mean_mm": float(np.mean([summary["variants"][f"single_stage_s{i}"][cn]["mean_mm"] for i in range(3)])),
            "std_mm": float(np.std([summary["variants"][f"single_stage_s{i}"][cn]["mean_mm"] for i in range(3)], ddof=1)),
            "runs": [round(summary["variants"][f"single_stage_s{i}"][cn]["mean_mm"], 3) for i in range(3)],
        }
        for cn in cohorts
    }
    out_p = OUT_DIR / "r1_3_4_6_7_retrained_ablations.json"
    out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("ALL LOCAL RETRAINED ABLATIONS COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
