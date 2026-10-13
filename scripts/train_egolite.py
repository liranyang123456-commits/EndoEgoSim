"""Train EgoLite on EndoEgoSim clips.

  python scripts/train_egolite.py --out results/egolite/run1 --iters 20000
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.metrics import evaluate_trajectory  # noqa: E402
from endosim.lite.data import SimClips, worker_init  # noqa: E402
from endosim.lite.egolite import (NEW_MODULE_PREFIXES, build_from_args, chain,  # noqa: E402
                                  pixel_log_ratio, pixel_rho)
from endosim.lite.infer import infer_sequence
from endosim.lite.window_ba import ba_window  # noqa: E402


DEPTH_UNIT_MM = 50.0


def geodesic(Ra, Rb):
    c = ((Ra @ Rb.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6))


def masked_median(x, m):
    out = []
    for xi, mi in zip(x, m):
        v = xi[mi]
        out.append(v.median() if v.numel() else xi.new_tensor(1.0))
    return torch.stack(out)


def compute_losses(model, batch, device, w):
    img = batch["img"].to(device, non_blocking=True)
    ph = batch["ph"].to(device, non_blocking=True)
    K = batch["K"].to(device)
    depth = batch["depth"].to(device)
    layer = batch["layer"].to(device)
    Twc = batch["T_wc"].to(device)
    B, S = img.shape[:2]
    P = S - 1

    def pairs(x, off=0):
        return x[:, off:off + P].reshape(B * P, *x.shape[2:])

    Kp = K[:, None].expand(B, P, 3, 3).reshape(B * P, 3, 3)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(pairs(img), pairs(img, 1), Kp, pairs(ph), pairs(ph, 1), clip=(B, P))

    T_rel = torch.linalg.inv(Twc[:, 1:]) @ Twc[:, :-1]          # frame k -> k+1
    T_rel = T_rel.reshape(B * P, 4, 4)
    R_gt, t_gt = T_rel[:, :3, :3], T_rel[:, :3, 3]
    z_gt = pairs(depth)
    valid = z_gt > 0
    if model.abs_depth:
        med = torch.full((z_gt.shape[0],), DEPTH_UNIT_MM, device=device)
    else:
        med = masked_median(z_gt, valid).clamp(min=1e-3)
    t_gt_n = t_gt / med[:, None]
    z_gt_n = z_gt / med[:, None, None]

    L = {}
    L["t"] = (out["t"] - t_gt_n).abs().sum(-1).mean()
    L["r"] = geodesic(out["R"], R_gt).mean()
    L["t0"] = (out["t0"] - t_gt_n).abs().sum(-1).mean()
    L["r0"] = geodesic(out["R0"], R_gt).mean()
    if "R1" in out:
        L["t1"] = (out["t1"] - t_gt_n).abs().sum(-1).mean()
        L["r1"] = geodesic(out["R1"], R_gt).mean()
    if "tp" in out:
        L["tp"] = (out["tp"] - t_gt_n).abs().sum(-1).mean()
        L["rp"] = geodesic(out["Rp"], R_gt).mean()
    lz = torch.log(z_gt_n.clamp(min=1e-3))
    L["depth"] = (out["logz"] - lz).abs()[valid].mean()
    lab = pairs(layer)
    cw = torch.tensor([1.0, 1.0, 3.0], device=device)
    L["layer"] = F.cross_entropy(out["logits"], lab, weight=cw, ignore_index=-100)
    if "logits1" in out:
        L["layer1"] = F.cross_entropy(out["logits1"], lab, weight=cw, ignore_index=-100)
    if "sem_logits" in out:
        sem = pairs(batch["sem"].to(device))
        L["sem"] = (F.cross_entropy(out["sem_logits"], sem, ignore_index=-100)
                    if bool((sem >= 0).any()) else out["sem_logits"].sum() * 0.0)
    f_cam_gt, Xp_gt = model.project(z_gt.float(), out["u"], out["v"], Kp.float(), R_gt, t_gt)
    cam_ok = valid & (Xp_gt[..., 2] > 0.2 * z_gt) & (f_cam_gt.norm(dim=1) < 64.0)
    err_cam = ((out["f_cam"] - f_cam_gt).abs().sum(1) / 8.0).clamp(max=8.0)
    L["cam"] = err_cam[cam_ok].mean() if cam_ok.any() else err_cam.sum() * 0.0
    e = ((out["f_obs"] - f_cam_gt).abs().sum(1) / 8.0).clamp(max=8.0)
    stat = cam_ok & (lab == 0)
    if stat.any():
        L["unc"] = (e * torch.exp(-out["logvar"]) + out["logvar"])[stat].mean()
    else:
        L["unc"] = out["logvar"].sum() * 0.0
    if model.train_flow:
        L["flow"] = e[stat].mean() if stat.any() else out["f_obs"].sum() * 0.0
    if model.abs_depth:
        rho_gt = torch.ones_like(med)
    else:
        med_next = masked_median(pairs(depth, 1), pairs(depth, 1) > 0).clamp(min=1e-3)
        rho_gt = med_next / med
    cur = torch.tensor([b * P + k for b in range(B) for k in range(P - 1)], device=device, dtype=torch.long)
    sub = {k: out[k][cur] for k in ("zp", "f_cam", "logits", "rho")}
    sub["u"], sub["v"] = out["u"], out["v"]
    Z_next = out["Z"][cur + 1]
    log_ratio, inside = pixel_log_ratio(sub, Z_next)
    lr_gt = torch.log(rho_gt[cur])[:, None, None]
    cons_ok = inside & stat[cur]
    if cons_ok.any():
        L["cons"] = (log_ratio - lr_gt).abs()[cons_ok].mean()
    else:
        L["cons"] = log_ratio.sum() * 0.0
    rho_pix = pixel_rho(sub, Z_next)
    L["rho"] = (torch.log(rho_pix) - torch.log(rho_gt[cur])).abs().mean()
    if model.abs_depth:
        rho_all = torch.ones_like(out["rho"])
    else:
        rho_all = out["rho"].clone()
        rho_all[cur] = rho_pix
    Rb = out["R"].reshape(B, P, 3, 3)
    tb = out["t"].reshape(B, P, 3)
    rb = rho_all.reshape(B, P)
    med0 = med.reshape(B, P)[:, 0]
    l_abs = []
    for i in range(B):
        Tw = chain(Rb[i], tb[i], rb[i])
        l_abs.append((Tw[1:, :3, 3] - Twc[i, 1:, :3, 3] / med0[i]).abs().sum(-1).mean())
    L["abs"] = torch.stack(l_abs).mean()
    total = sum(w[k] * v for k, v in L.items())
    return total, L, out


def _ssim(x, y):
    mu_x, mu_y = F.avg_pool2d(x, 3, 1, 1), F.avg_pool2d(y, 3, 1, 1)
    sx = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sy = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sxy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))
    return s.clamp(0, 1)


def _photo(a, b):
    return (0.85 * (1 - _ssim(a, b)) / 2 + 0.15 * (a - b).abs()).mean(1)


def compute_real_losses(model, batch, device):
    """Self-supervision on unlabelled real video: anchor-weighted photometric consistency of the
    solved pose and predicted depth (automasked), plus stereo-depth supervision where available."""
    img = batch["img"].to(device, non_blocking=True)
    ph = batch["ph"].to(device, non_blocking=True)
    K = batch["K"].to(device)
    depth = batch["depth"].to(device)
    B, S = img.shape[:2]
    P = S - 1

    def pairs(x, off=0):
        return x[:, off:off + P].reshape(B * P, *x.shape[2:])

    Kp = K[:, None].expand(B, P, 3, 3).reshape(B * P, 3, 3)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(pairs(img), pairs(img, 1), Kp, pairs(ph), pairs(ph, 1))
    i0 = F.avg_pool2d(pairs(img).float(), 2)
    i1 = F.avg_pool2d(pairs(img, 1).float(), 2)
    h2, w2 = i0.shape[-2:]
    Z = F.interpolate(out["Z"][:, None].float(), size=(h2, w2), mode="bilinear", align_corners=False)[:, 0]
    fx, fy = Kp[:, 0, 0] / 2, Kp[:, 1, 1] / 2
    cx, cy = (Kp[:, 0, 2] + 0.5) / 2 - 0.5, (Kp[:, 1, 2] + 0.5) / 2 - 0.5
    vv, uu = torch.meshgrid(torch.arange(h2, device=device, dtype=torch.float32),
                            torch.arange(w2, device=device, dtype=torch.float32), indexing="ij")
    X = torch.stack([(uu - cx[:, None, None]) / fx[:, None, None] * Z,
                     (vv - cy[:, None, None]) / fy[:, None, None] * Z, Z], -1)
    Xp = torch.einsum("nij,nhwj->nhwi", out["R"].float(), X) + out["t"].float()[:, None, None]
    zp = Xp[..., 2].clamp(min=1e-3)
    up = fx[:, None, None] * Xp[..., 0] / zp + cx[:, None, None]
    vp = fy[:, None, None] * Xp[..., 1] / zp + cy[:, None, None]
    grid = torch.stack([2 * (up + 0.5) / w2 - 1, 2 * (vp + 0.5) / h2 - 1], -1)
    i1w = F.grid_sample(i1, grid, mode="bilinear", padding_mode="border", align_corners=False)
    photo = _photo(i0, i1w)
    ident = _photo(i0, i1)
    valid = (grid.abs() < 1).all(-1) & (Xp[..., 2] > 1e-3) & (photo < ident)
    anchor = torch.softmax(out["logits"].float(), 1)[:, :1].detach()
    anchor = F.interpolate(anchor, size=(h2, w2), mode="bilinear", align_corners=False)[:, 0]
    wgt = anchor * valid.float()
    L = {"rphoto": (wgt * photo).sum() / wgt.sum().clamp(min=1.0)}
    d = pairs(depth)
    ok = d > 0
    L["rdepth"] = ((torch.log(out["Z"].float() * DEPTH_UNIT_MM) - torch.log(d.clamp(min=1e-3)))
                   .abs()[ok].mean() if bool(ok.any()) else out["Z"].sum() * 0.0)
    return L


def compute_ba_losses(model, batch, out, device, skips, iters):
    """Train the camera/tissue confidence through differentiable multi-frame dense BA."""
    img=batch["img"].to(device,non_blocking=True); ph=batch["ph"].to(device,non_blocking=True)
    K=batch["K"].to(device); Twc=batch["T_wc"].to(device)
    B,S=img.shape[:2]; P=S-1; h,w=out["Z"].shape[-2:]
    pairs=lambda x,off=0,p=P:x[:,off:off+p].reshape(B*p,*x.shape[2:])
    extra={}
    for s in skips:
        p=S-s; i0=img[:,:p].reshape(B*p,*img.shape[2:]); i1=img[:,s:].reshape(B*p,*img.shape[2:])
        p0=ph[:,:p].reshape(B*p,*ph.shape[2:]); p1=ph[:,s:].reshape(B*p,*ph.shape[2:])
        Ks=K[:,None].expand(B,p,3,3).reshape(B*p,3,3)
        with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            fl=model.observed_flow(i0,i1)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            o=model(i0,i1,Ks,p0,p1,flow=fl)
        wk=o["w_ba"] if "w_ba" in o else o["w"]
        extra[s]=(o["f_obs"].view(B,p,2,h,w),wk.view(B,p,h,w))
    f1=out["f_obs"].view(B,P,2,h,w); wkey=out.get("w_ba",out["w"]); w1=wkey.view(B,P,h,w)
    Rb=out["R"].view(B,P,3,3); tb=out["t"].view(B,P,3); Zb=out["Z"].view(B,P,h,w)
    ii,jj=torch.triu_indices(S,S,1,device=device); lt=[]; lr=[]
    lw=getattr(model,"ba_logw",None); pw=torch.exp(model.ba_logp) if hasattr(model,"ba_logp") else 1.0
    for b in range(B):
        ed={1:(f1[b],w1[b])}; ed.update({s:(f[b],wt[b]) for s,(f,wt) in extra.items()})
        Tb,_=ba_window(Rb[b],tb[b],Zb[b],ed,K[b],out["u"],out["v"],logw=lw,
                       prior_weight=pw,iters=iters,unit=DEPTH_UNIT_MM)
        est=Tb[jj]@torch.linalg.inv(Tb[ii]); gt=torch.linalg.inv(Twc[b,jj])@Twc[b,ii]
        lt.append(((est[:,:3,3]-gt[:,:3,3])/DEPTH_UNIT_MM).abs().sum(-1).mean())
        lr.append(geodesic(est[:,:3,:3],gt[:,:3,:3]).mean())
    return {"bat":torch.stack(lt).mean(),"bar":torch.stack(lr).mean()}


def compute_ba_losses(model, batch, out, device, skips, iters):
    """Train the camera/tissue confidence through differentiable multi-frame dense BA."""
    from endosim.lite.window_ba import ba_window
    img=batch["img"].to(device,non_blocking=True); ph=batch["ph"].to(device,non_blocking=True)
    K=batch["K"].to(device); Twc=batch["T_wc"].to(device)
    B,S=img.shape[:2]; P=S-1; h,w=out["Z"].shape[-2:]
    extra={}
    for s in skips:
        if s>=S: continue
        p=S-s; i0=img[:,:p].reshape(B*p,*img.shape[2:]); i1=img[:,s:].reshape(B*p,*img.shape[2:])
        p0=ph[:,:p].reshape(B*p,*ph.shape[2:]); p1=ph[:,s:].reshape(B*p,*ph.shape[2:])
        Ks=K[:,None].expand(B,p,3,3).reshape(B*p,3,3)
        with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.bfloat16): fl=model.observed_flow(i0,i1)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16): o=model(i0,i1,Ks,p0,p1,flow=fl)
        wk=o["w_ba"] if model.ba else o["w"]
        extra[s]=(o["f_obs"].view(B,p,2,h,w),wk.view(B,p,h,w))
    w1=out["w_ba"] if model.ba else out["w"]
    f1=out["f_obs"].view(B,P,2,h,w); w1=w1.view(B,P,h,w)
    Rb=out["R"].view(B,P,3,3); tb=out["t"].view(B,P,3); Zb=out["Z"].view(B,P,h,w)
    ii,jj=torch.triu_indices(S,S,1,device=device); lt=[]; lr=[]
    logw=model.ba_logw if model.ba else None; pw=torch.exp(model.ba_logp) if model.ba else 1.0
    for b in range(B):
        edges={1:(f1[b],w1[b])}; edges.update({s:(f[b],ww[b]) for s,(f,ww) in extra.items()})
        Tb,_=ba_window(Rb[b],tb[b],Zb[b],edges,K[b],out["u"],out["v"],logw=logw,
                       prior_weight=pw,iters=iters,unit=DEPTH_UNIT_MM)
        est=Tb[jj]@torch.linalg.inv(Tb[ii]); gt=torch.linalg.inv(Twc[b,jj])@Twc[b,ii]
        lt.append(((est[:,:3,3]-gt[:,:3,3])/DEPTH_UNIT_MM).abs().sum(-1).mean())
        lr.append(geodesic(est[:,:3,:3],gt[:,:3,:3]).mean())
    return {"bat":torch.stack(lt).mean(),"bar":torch.stack(lr).mean()}


def compute_pseudo_losses(model, batch, device):
    """Metric pose distillation from π³ pseudo trajectories plus Hamlyn stereo-depth loss."""
    img, ph = batch["img"].to(device, non_blocking=True), batch["ph"].to(device, non_blocking=True)
    K, depth, Twc = batch["K"].to(device), batch["depth"].to(device), batch["T_wc"].to(device)
    B, S = img.shape[:2]
    P = S - 1
    pairs = lambda x, off=0: x[:, off:off + P].reshape(B * P, *x.shape[2:])
    Kp = K[:, None].expand(B, P, 3, 3).reshape(B * P, 3, 3)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(pairs(img), pairs(img, 1), Kp, pairs(ph), pairs(ph, 1), clip=(B, P))
    T_rel = (torch.linalg.inv(Twc[:, 1:]) @ Twc[:, :-1]).reshape(B * P, 4, 4)
    Rgt, tgt = T_rel[:, :3, :3], T_rel[:, :3, 3] / DEPTH_UNIT_MM
    L = {"pt": (out["t"] - tgt).abs().sum(-1).mean(), "pr": geodesic(out["R"], Rgt).mean(),
         "pt0": (out["t0"] - tgt).abs().sum(-1).mean(), "pr0": geodesic(out["R0"], Rgt).mean()}
    Rb, tb = out["R"].reshape(B, P, 3, 3), out["t"].reshape(B, P, 3)
    ones = torch.ones(P, device=device)
    L["pabs"] = torch.stack([(chain(Rb[b], tb[b], ones)[1:, :3, 3] - Twc[b, 1:, :3, 3] / DEPTH_UNIT_MM)
                             .abs().sum(-1).mean() for b in range(B)]).mean()
    d = pairs(depth); ok = d > 0
    L["pdepth"] = ((torch.log(out["Z"].float() * DEPTH_UNIT_MM) - torch.log(d.clamp(min=1e-3)))
                   .abs()[ok].mean() if bool(ok.any()) else out["Z"].sum() * 0.0)
    total = 10.0 * (L["pt"] + L["pr"]) + 2.0 * (L["pt0"] + L["pr0"]) + 5.0 * L["pabs"] + L["pdepth"]
    return total, L


@torch.no_grad()
def validate(model, seqs, device, metric=False):
    """Mean Sim(3) ATE; with metric=True, SE(3) ATE in mm (50 mm units, no scale alignment)."""
    model.eval()
    errs = []
    for d in seqs:
        est, gt, _, _, _ = infer_sequence(model, d, device)
        if not np.isfinite(est).all():
            errs.append(float("nan"))
            continue
        if metric:
            est = est.copy()
            est[:, :3, 3] *= DEPTH_UNIT_MM
            errs.append(evaluate_trajectory(est, gt)["ate_se3"]["rmse"])
        else:
            errs.append(evaluate_trajectory(est, gt)["ate_sim3"]["rmse"])
    model.train()
    e = np.asarray(errs, dtype=np.float64)
    return float(np.nanmean(e)), int(np.isnan(e).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--clip", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-n", type=int, default=24)
    ap.add_argument("--no-ph", action="store_true")
    ap.add_argument("--gn-iters", type=int, default=3)
    ap.add_argument("--init", default=None)
    ap.add_argument("--abs-depth", action="store_true",
                    help="predict depth in a fixed unit (50 mm) instead of per-frame median units")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flow-net", default="small", choices=["small", "large"])
    ap.add_argument("--head-extra", type=int, default=0,
                    help="zero-initialised dilated residual blocks after the pixel head")
    ap.add_argument("--ctx", action="store_true", help="add ResNet-18 layer4 context (zero-init)")
    ap.add_argument("--temporal", default=None, choices=["bi", "causal"],
                    help="GRU pose prior over the clip and a second, regularised GN solve")
    ap.add_argument("--train-flow", action="store_true",
                    help="fine-tune RAFT with rigid-flow supervision on stationary pixels")
    ap.add_argument("--depth-prior", default=None, choices=["midas"],
                    help="frozen monocular depth prior replacing the pseudo-height inputs")
    ap.add_argument("--real-weight", type=float, default=0.0,
                    help="weight of self-supervised losses on unlabelled real video (Hamlyn)")
    ap.add_argument("--real-batch", type=int, default=2)
    ap.add_argument("--pseudo", default="", help="Hamlyn π³ pseudo-pose labels (.npz files)")
    ap.add_argument("--pseudo-weight", type=float, default=0.0)
    ap.add_argument("--pseudo-batch", type=int, default=1)
    ap.add_argument("--pseudo-clip", type=int, default=4)
    ap.add_argument("--ba-skips", default="", help="additional flow-edge gaps for differentiable window BA")
    ap.add_argument("--ba-iters", type=int, default=4)
    ap.add_argument("--ba-weight", type=float, default=1.0)
    ap.add_argument("--ba-ckpt", default="", help="resume weights (and optimizer/scheduler if a training checkpoint)")
    ap.add_argument("--iter-refine", type=int, default=0,
                    help="rounds of residual -> motion layers -> pose refinement (shared weights)")
    ap.add_argument("--sem-head", type=int, default=0,
                    help="anatomy motion-group head (4 = anchor/large/medium/instrument), "
                         "supervised where anatomy/ maps exist")
    ap.add_argument("--anchor-t", type=int, default=0, nargs="?", const=1,
                    help="per-pixel temporal anchor module + second GN solve "
                         "(1: ~13-pair field; 2: ~63-pair field with pose-free flow statistics)")
    ap.add_argument("--data", default="sim_data/train:1",
                    help="comma-separated root:weight; roots with abs_motion/ use absolute-static labels")
    ap.add_argument("--stride-p", default="0.7,0.2,0.1", help="probabilities of frame strides 1,2,3")
    ap.add_argument("--val-abs", default="", help="validation root with abs_motion (metric SE(3) ATE)")
    ap.add_argument("--pair-labels", action="store_true",
                    help="label ablation: use the per-pair motion masks for every source instead of the "
                         "sequence-level (abs_motion) labels")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda"

    sd_full = None
    if args.ba_ckpt:
        ck = torch.load(args.ba_ckpt, map_location="cpu", weights_only=False)
        sd_full = ck["model"]
        print(f"[resume] {args.ba_ckpt}: iter={ck.get('iter')}", flush=True)
    model = build_from_args(vars(args), pretrained=True).to(device)
    if args.ba_ckpt:
        mismatch=model.load_state_dict(sd_full,strict=False)
        bad=[k for k in mismatch.missing_keys if not k.startswith(NEW_MODULE_PREFIXES)]
        if bad: raise RuntimeError(f"resume checkpoint missing keys: {bad[:10]}")
        model.ba_gaps=args.ba_skips
        print(f"[resume] restored BA checkpoint with {len(mismatch.missing_keys)} newly initialized keys",flush=True)
    if args.init:
        sd = torch.load(args.init, map_location="cpu")["model"]
        own = model.state_dict()
        k0 = "pix.0.0.weight"                     # pseudo-height maps are the last two inputs
        if args.no_ph and k0 in sd and sd[k0].shape[1] == own[k0].shape[1] + 2:
            sd[k0] = sd[k0][:, : own[k0].shape[1]].clone()
        model.ba_gaps = args.ba_skips
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [m for m in missing if not m.startswith(NEW_MODULE_PREFIXES)
               and not (args.flow_net != "small" and m.startswith("flow."))]
        if bad:
            raise RuntimeError(f"init mismatch missing={bad}")
        print(f"[init] {args.init}: {len(sd)} tensors loaded, {len(missing)} new", flush=True)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[EgoLite] params={n_par:.2f}M trainable={n_train:.2f}M flow={args.flow_net} "
          f"head_extra={args.head_extra} ctx={args.ctx} temporal={args.temporal} "
          f"train_flow={args.train_flow} anchor_t={args.anchor_t}", flush=True)
    json.dump(vars(args) | {"params_M": n_par}, open(os.path.join(args.out, "args.json"), "w"),
              indent=1)

    roots = [(s.rsplit(":", 1)[0], float(s.rsplit(":", 1)[1]) if ":" in s else 1.0)
             for s in args.data.split(",")]
    ds = SimClips(clip=args.clip, length=args.iters * args.batch + 1000, seed=args.seed,
                  roots=roots, stride_p=tuple(float(x) for x in args.stride_p.split(",")),
                  use_abs_labels=not args.pair_labels)
    print(f"train sequences: {[(r, len(g)) for (r, _), (g, _) in zip(roots, ds.groups)]}", flush=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    pin_memory=True, drop_last=True, worker_init_fn=worker_init,
                    persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    val = sorted(glob.glob("sim_data/val/seq_*"))
    val = val[:: max(1, len(val) // args.val_n)][: args.val_n]
    val_abs = sorted(glob.glob(os.path.join(args.val_abs, "seq_*"))) if args.val_abs else []
    val_abs = val_abs[:: max(1, len(val_abs) // args.val_n)][: args.val_n]

    enc = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    flow = [p for n, p in model.named_parameters() if n.startswith("flow.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters()
            if not n.startswith(("enc.", "flow.")) and p.requires_grad]
    groups = [{"params": enc, "lr": args.lr * 0.5}, {"params": rest, "lr": args.lr}]
    if flow:
        groups.append({"params": flow, "lr": args.lr * 0.2})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    warm = 500

    def lr_at(it):
        if it < warm:
            return (it + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, args.iters - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    w = {"t": 10.0, "r": 10.0, "t0": 2.0, "r0": 2.0, "depth": 1.0, "layer": 0.5,
         "cam": 1.0, "unc": 0.1, "rho": 2.0, "cons": 1.0, "abs": 5.0,
         "t1": 5.0, "r1": 5.0, "tp": 5.0, "rp": 5.0, "flow": 1.0, "layer1": 0.25, "sem": 0.5}
    if args.ba_ckpt:
        restored=torch.load(args.ba_ckpt,map_location="cpu",weights_only=False)
        start_it=int(restored.get("iter",0)); best=float(restored.get("best",float("inf")))
        missing,unexpected=model.load_state_dict(restored["model"],strict=False)
        bad=[k for k in missing if not k.startswith(NEW_MODULE_PREFIXES)]
        if bad: raise RuntimeError(f"resume checkpoint missing keys: {bad[:10]}")
        if "optimizer" in restored:
            opt.load_state_dict(restored["optimizer"])
            for _ in range(start_it): sched.step()
            print(f"[resume] restored optimiser at iter={start_it}",flush=True)
        else:
            print(f"[resume] weights only at iter={start_it}; optimiser state was not checkpointed",flush=True)
    else:
        start_it=0
        best=float("inf")
    real_iter = None
    if args.real_weight > 0:
        from endosim.lite.data import RealClips
        rds = RealClips(length=args.iters * args.real_batch + 100, seed=args.seed)
        print(f"real sequences: {len(rds.seqs)}", flush=True)
        real_iter = iter(DataLoader(rds, batch_size=args.real_batch, shuffle=False, num_workers=2,
                                    pin_memory=True, drop_last=True, worker_init_fn=worker_init,
                                    persistent_workers=True))
    pseudo_iter = None
    if args.pseudo_weight > 0:
        from endosim.lite.data import PseudoPoseClips
        pds = PseudoPoseClips(args.pseudo, clip=args.pseudo_clip,
                              length=args.iters * args.pseudo_batch + 100, seed=args.seed)
        print(f"pseudo-labelled windows: {len(pds.items)}", flush=True)
        pseudo_iter = iter(DataLoader(pds, batch_size=args.pseudo_batch, shuffle=False, num_workers=2,
                                      pin_memory=True, drop_last=True, worker_init_fn=worker_init,
                                      persistent_workers=True))
    t_start = time.time()
    model.train()
    it = start_it
    for batch in dl:
        if it >= args.iters:
            break
        total, L, out = compute_losses(model, batch, device, w)
        if args.ba_skips:
            Lb=compute_ba_losses(model,batch,out,device,[int(x) for x in args.ba_skips.split(',')],args.ba_iters)
            total=total+args.ba_weight*(Lb['bat']+Lb['bar']); L.update(Lb)
        if pseudo_iter is not None:
            Lp_total, Lp = compute_pseudo_losses(model, next(pseudo_iter), device)
            total = total + args.pseudo_weight * Lp_total
            L.update(Lp)
        if real_iter is not None:
            Lr = compute_real_losses(model, next(real_iter), device)
            total = total + args.real_weight * (Lr["rphoto"] + 0.5 * Lr["rdepth"])
            L.update(Lr)
        if not torch.isfinite(total):
            print(f"[{it}] non-finite loss, step skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            it += 1
            sched.step()
            continue
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        it += 1
        if it % 50 == 0 or it == 1:
            msg = " ".join(f"{k}={float(v):.4f}" for k, v in L.items())
            if args.ba_ckpt:
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "iter": it,
                            "best": best, "args": vars(args)}, args.ba_ckpt)
            print(f"[{it}/{args.iters}] total={float(total):.4f} {msg} "
                  f"beta={float(model.beta):.3f} share={float(torch.sigmoid(model.deform_logit)):.2f} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
        if it % args.val_every == 0 or it == args.iters:
            ate, nbad = validate(model, val, device)
            print(f"  [val] n={len(val)} ATE(Sim3) mean={ate:.3f} mm nonfinite={nbad}", flush=True)
            if val_abs:
                ate_a, nbad_a = validate(model, val_abs, device, metric=True)
                print(f"  [val-abs] n={len(val_abs)} ATE(SE3, metric) mean={ate_a:.3f} mm "
                      f"nonfinite={nbad_a}", flush=True)
            torch.save({"model": model.state_dict(), "optimizer":opt.state_dict(), "iter": it,
                        "best":best, "args": vars(args)}, os.path.join(args.out, "last.pth"))
            if ate < best:
                best = ate
                torch.save({"model": model.state_dict(), "optimizer":opt.state_dict(), "iter": it,
                            "best":best, "args": vars(args)}, os.path.join(args.out, "best.pth"))
                print(f"  [val] new best {best:.3f} mm", flush=True)
    print(f"done in {time.time() - t_start:.0f}s, best val {best:.3f} mm", flush=True)


if __name__ == "__main__":
    main()
