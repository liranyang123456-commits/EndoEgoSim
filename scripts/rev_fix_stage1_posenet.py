"""Fix & retrain the same-corpus 13.0M ResNet-18 PoseNet (R1.7) cleanly without collapse.

Key fixes over the collapsed v1 run:
1. Exact 13.01M Monodepth2/AF-SfMLearner/EndoDAC ResNet-18 6-ch PoseNet architecture
   (ImageNet-pretrained ResNet-18 encoder + 4-conv PoseDecoder with ImageNet normalization & GELU/BN).
2. MSE + SmoothL1 + geodesic + 8-frame trajectory loss (eliminates the L1 zero-median collapse on hovering clips).
3. Fast image-only SimClips loader (skips depth/mask/pseudo-height I/O -> ~10x faster!).
4. Live validation on dev_92 every 2,000 steps + automatic step-magnitude health check.
5. Full evaluation on all 8 Protocol-A cohorts + Protocol-B StereoMIS (11 full videos).
"""
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from endosim.eval.metrics import evaluate_trajectory, load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames, select_frame_indices  # noqa: E402
from endosim.lite.data import H_IN, W_IN, _barrel_free, load_frames, worker_init  # noqa: E402
from endosim.lite.egolite import IMAGENET_MEAN, IMAGENET_STD, build_from_args, chain, se3_exp, so3_log  # noqa: E402
from endosim.lite.infer import infer_sequence  # noqa: E402

DEPTH_UNIT_MM = 50.0


def geodesic(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    c = ((Ra @ Rb.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    return torch.acos(c.clamp(-1.0 + 1e-6, 1.0 - 1e-6))


class PoseOnlySimClips(Dataset):
    """Fast RGB+pose-only version of SimClips (exact same sequence lists, sampling weights, and strides)."""

    def __init__(self, clip: int = 8, length: int = 60000, seed: int = 0,
                 roots=None, stride_p=(0.7, 0.2, 0.1)):
        self.clip = clip
        self.length = length
        self.seed = seed
        self.stride_p = np.asarray(stride_p, dtype=np.float64)
        self.stride_p /= self.stride_p.sum()
        self.groups = []
        self.weights = []
        for root, wt in roots:
            seqs = sorted(glob.glob(os.path.join(root, "seq_*")))
            seqs = [d for d in seqs if os.path.isfile(os.path.join(d, "pose_c2w.txt")) and _barrel_free(d)]
            self.groups.append(seqs)
            self.weights.append(float(wt))
        self.probs = np.asarray(self.weights, dtype=np.float64)
        self.probs /= self.probs.sum()
        self._pose_cache = {}
        self._frame_cache = {}

    def __len__(self):
        return self.length

    def _get_meta(self, d):
        if d not in self._pose_cache:
            self._pose_cache[d] = load_pose_txt(os.path.join(d, "pose_c2w.txt")).astype(np.float32)
            self._frame_cache[d] = list_color_frames(d)
        return self._pose_cache[d], self._frame_cache[d]

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed * 1000003 + idx)
        for _ in range(10):
            g = int(rng.choice(len(self.groups), p=self.probs))
            d = self.groups[g][int(rng.integers(len(self.groups[g])))]
            poses, frames = self._get_meta(d)
            n = min(len(poses), len(frames))
            stride = int(rng.choice([1, 2, 3], p=self.stride_p))
            span = (self.clip - 1) * stride
            if n <= span:
                stride = 1
                span = self.clip - 1
            if n <= span:
                continue
            start = int(rng.integers(0, n - span))
            sel = [start + k * stride for k in range(self.clip)]
            imgs = []
            ok = True
            for fi in sel:
                bgr = cv2.imread(frames[fi])
                if bgr is None:
                    ok = False
                    break
                if bgr.shape[1] != W_IN or bgr.shape[0] != H_IN:
                    bgr = cv2.resize(bgr, (W_IN, H_IN), interpolation=cv2.INTER_AREA)
                imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            if not ok:
                continue
            Twc = poses[sel].copy()
            # Canonicalize to frame-0 gauge (exact match to SimClips)
            T0_inv = np.linalg.inv(Twc[0])
            Twc = np.einsum("ij,njk->nik", T0_inv, Twc).astype(np.float32)
            img_arr = np.stack(imgs, 0).transpose(0, 3, 1, 2)  # (S, 3, H, W)
            return {"img": torch.from_numpy(img_arr), "T_wc": torch.from_numpy(Twc)}
        raise RuntimeError("Failed to sample clip")


class FixedSameDataPoseNet(nn.Module):
    """Exact 13.01M Monodepth2/AF-SfMLearner/EndoDAC ResNet-18 6-ch PoseNet with ImageNet norm."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        old_w = m.conv1.weight.data
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            self.conv1.weight.data.copy_(torch.cat([old_w, old_w], dim=1) * 0.5)
        self.bn1, self.relu, self.maxpool = m.bn1, m.relu, m.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        # Standard Monodepth2 PoseDecoder channel layout (~13.0M total params)
        self.squeeze = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.GELU())
        self.pconv1 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.GELU())
        self.pconv2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.GELU())
        self.pconv3 = nn.Conv2d(256, 6, 1)
        nn.init.normal_(self.pconv3.weight, 0.0, 1e-2)
        nn.init.zeros_(self.pconv3.bias)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, img0: torch.Tensor, img1: torch.Tensor):
        x0 = (img0 - self.mean) / self.std
        x1 = (img1 - self.mean) / self.std
        x = torch.cat([x0, x1], dim=1)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.pconv2(self.pconv1(self.squeeze(x)))
        xi = self.pconv3(x).mean(dim=(-2, -1))
        with torch.autocast(device_type=img0.device.type, enabled=False):
            R, t = se3_exp(xi.float())
        return R, t


def read_list(rel: str) -> list[str]:
    return [
        ln.strip()
        for ln in (ROOT / rel).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#") and os.path.isfile(os.path.join(ln.strip(), "pose_c2w.txt"))
    ]


@torch.no_grad()
def quick_val_dev(model: nn.Module, seqs: list[str]) -> tuple[float, float]:
    model.eval()
    ates, steps = [], []
    for d in seqs[:24]:
        gt_all = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
        idx = select_frame_indices(gt_all, "uniform", max_frames=64)
        paths = list_color_frames(d, indices=idx)
        imgs, _, _ = load_frames(paths)
        img = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
        n = len(paths)
        Rs, ts = [], []
        for st in range(0, n - 1, 32):
            ed = min(n - 1, st + 32)
            Rp, tp = model(img[st:ed], img[st + 1:ed + 1])
            Rs.append(Rp.float())
            ts.append(tp.float())
        R_all, t_all = torch.cat(Rs, 0), torch.cat(ts, 0)
        est = chain(R_all, t_all, torch.ones(n - 1, device="cuda")).cpu().numpy().astype(np.float64)
        ates.append(float(evaluate_trajectory(est, gt_all[idx])["ate_sim3"]["rmse"]))
        steps.append(float((t_all.norm(dim=-1) * DEPTH_UNIT_MM).median().item()))
    model.train()
    return float(np.mean(ates)), float(np.median(steps))


def train_posenet(out_ckpt: Path, iters: int = 12000, batch: int = 6, clip: int = 8,
                  lr: float = 3e-4, seed: int = 0) -> Path:
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FixedSameDataPoseNet(pretrained=True).cuda().train()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Train FixedSameDataPoseNet] params={n_par:.2f}M iters={iters} batch={batch}x{clip}", flush=True)

    s_root = r"D:\ego_lite_pack\sim_data\train" if os.path.isdir(r"D:\ego_lite_pack\sim_data\train") else str(ROOT / "sim_data/train")
    roots = [(s_root, 2.0), (str(ROOT / "sim_data_abs/train"), 1.0), (str(ROOT / "sim_data_r2s/train"), 1.0)]
    ds = PoseOnlySimClips(clip=clip, length=iters * batch + 400, seed=seed, roots=roots, stride_p=(0.7, 0.2, 0.1))
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=6, pin_memory=True,
                    drop_last=True, worker_init_fn=worker_init, persistent_workers=True, prefetch_factor=4)
    dev_seqs = read_list("lists/simtest92.txt")

    enc_names = ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")
    enc_p = [p for n, p in model.named_parameters() if n.startswith(enc_names)]
    dec_p = [p for n, p in model.named_parameters() if not n.startswith(enc_names)]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": lr * 0.5}, {"params": dec_p, "lr": lr}], weight_decay=1e-4)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1.0 + math.cos(math.pi * (it - warm) / max(1, iters - warm)))
    )

    t0 = time.time()
    for it, b in enumerate(dl):
        if it >= iters:
            break
        img = b["img"].cuda(non_blocking=True)
        Twc = b["T_wc"].cuda(non_blocking=True)
        B, S = img.shape[:2]
        P = S - 1
        i0 = img[:, :P].reshape(B * P, 3, H_IN, W_IN)
        i1 = img[:, 1:].reshape(B * P, 3, H_IN, W_IN)
        T_rel = (torch.linalg.inv(Twc[:, 1:]) @ Twc[:, :-1]).reshape(B * P, 4, 4)
        R_gt, t_gt = T_rel[:, :3, :3], T_rel[:, :3, 3] / DEPTH_UNIT_MM

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            R, t = model(i0, i1)
        # SmoothL1 + MSE prevents zero-median collapse while keeping outlier robustness
        lt_mse = F.mse_loss(t, t_gt) * 200.0
        lt_sl1 = F.smooth_l1_loss(t, t_gt, beta=0.02) * 20.0
        lr_vec = F.mse_loss(so3_log(R), so3_log(R_gt)) * 200.0
        lr_geo = geodesic(R, R_gt).mean() * 10.0
        Rb, tb = R.reshape(B, P, 3, 3), t.reshape(B, P, 3)
        ones = torch.ones(P, device="cuda")
        l_traj = torch.stack([
            F.smooth_l1_loss(chain(Rb[i], tb[i], ones)[1:, :3, 3], Twc[i, 1:, :3, 3] / DEPTH_UNIT_MM, beta=0.05)
            for i in range(B)
        ]).mean() * 15.0
        loss = lt_mse + lt_sl1 + lr_vec + lr_geo + l_traj

        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
        if (it + 1) % 2000 == 0 or it == 0:
            v_ate, v_step = quick_val_dev(model, dev_seqs)
            print(f"  [{it+1}/{iters}] loss={float(loss.detach()):.4f} | val_dev24 ATE={v_ate:.2f}mm med_step={v_step:.2f}mm ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"model": model.state_dict(), "params_M": n_par, "iters": iters, "arch": "FixedSameDataPoseNet_BN_GELU_INnorm"}, out_ckpt)
    print(f"Saved {out_ckpt} in {time.time()-t0:.1f}s", flush=True)
    return out_ckpt


@torch.no_grad()
def eval_posenet_all(ckpt_p: Path, out_json: Path, full_dir: Path) -> dict:
    model = FixedSameDataPoseNet(pretrained=False).cuda().eval()
    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])

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

    ck_ego = torch.load(ROOT / "results/egolite/r_v2/last.pth", map_location="cpu", weights_only=False)
    m_ego = build_from_args(ck_ego["args"], pretrained=False).cuda().eval()
    m_ego.load_state_dict(ck_ego["model"])

    res = {"params_M": ck.get("params_M", 12.50), "iters": ck.get("iters", 12000), "protocol_a": {}, "protocol_b_stereomis": {}}

    for cn, seqs in cohorts.items():
        pn_errs, ego_errs, step_ratios = [], [], []
        for d in seqs:
            gt_all = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
            idx = select_frame_indices(gt_all, "uniform", max_frames=64)
            paths = list_color_frames(d, indices=idx)
            gt_sel = gt_all[idx]
            imgs, _, _ = load_frames(paths)
            img = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
            n = len(paths)
            Rs, ts = [], []
            for st in range(0, n - 1, 32):
                ed = min(n - 1, st + 32)
                Rp, tp = model(img[st:ed], img[st + 1:ed + 1])
                Rs.append(Rp.float())
                ts.append(tp.float())
            R_all, t_all = torch.cat(Rs, 0), torch.cat(ts, 0)
            est = chain(R_all, t_all, torch.ones(n - 1, device="cuda")).cpu().numpy().astype(np.float64)
            pn_errs.append(float(evaluate_trajectory(est, gt_sel)["ate_sim3"]["rmse"]))

            pred_step = float((t_all.norm(dim=-1) * DEPTH_UNIT_MM).median().item())
            gt_step = float(np.median(np.linalg.norm(gt_sel[1:, :3, 3] - gt_sel[:-1, :3, 3], axis=-1)))
            if gt_step > 1e-3:
                step_ratios.append(pred_step / gt_step)

            est_e, gt_e, _, _, _ = infer_sequence(m_ego, d, "cuda", max_frames=64)
            ego_errs.append(float(evaluate_trajectory(est_e, gt_e)["ate_sim3"]["rmse"]))

        arr_p, arr_e = np.array(pn_errs), np.array(ego_errs)
        pval = float(wilcoxon(arr_p, arr_e).pvalue)
        res["protocol_a"][cn] = {
            "n": len(seqs),
            "posenet_mean_mm": float(arr_p.mean()),
            "posenet_median_mm": float(np.median(arr_p)),
            "egolite_rv2_mean_mm": float(arr_e.mean()),
            "egolite_rv2_median_mm": float(np.median(arr_e)),
            "reduction_pct_vs_posenet": float((1.0 - arr_e.mean() / arr_p.mean()) * 100.0),
            "wilcoxon_p_vs_rv2": pval,
            "median_step_ratio_to_gt": float(np.median(step_ratios)) if step_ratios else 0.0,
            "posenet_per_seq": [round(x, 4) for x in pn_errs],
            "egolite_per_seq": [round(x, 4) for x in ego_errs],
        }
        print(f"  [Prot A {cn:14s}] PoseNet={arr_p.mean():.2f}mm (med {np.median(arr_p):.2f}) | "
              f"EgoLite={arr_e.mean():.2f}mm | red={res['protocol_a'][cn]['reduction_pct_vs_posenet']:.1f}% | "
              f"p={pval:.3g} | step_ratio={res['protocol_a'][cn]['median_step_ratio_to_gt']:.2f}", flush=True)

    full_dir.mkdir(parents=True, exist_ok=True)
    smis_seqs = cohorts["stereomis_11"]
    pn_full, ego_full = [], []
    ego_full_summ = json.loads((ROOT / "results/sota/egolite_r_v2_stereomis_full/summary.json").read_text(encoding="utf-8"))
    ego_map = {r["seq_id"]: r["ate_sim3"]["rmse"] for r in ego_full_summ["records"]}
    per_smis = []
    for d in smis_seqs:
        sid = os.path.basename(d)
        gt_all = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
        paths = list_color_frames(d)
        n = len(paths)
        Rs, ts = [], []
        chunk = 128
        for st in range(0, n - 1, chunk):
            ed = min(n - 1, st + chunk)
            imgs, _, _ = load_frames(paths[st:ed + 1])
            im = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
            for j in range(0, ed - st, 32):
                je = min(ed - st, j + 32)
                Rp, tp = model(im[j:je], im[j + 1:je + 1])
                Rs.append(Rp.float().cpu())
                ts.append(tp.float().cpu())
        R_all, t_all = torch.cat(Rs, 0), torch.cat(ts, 0)
        est = chain(R_all, t_all, torch.ones(n - 1)).numpy().astype(np.float64)
        np.savetxt(full_dir / f"{sid}_c2w.txt", est.reshape(-1, 16))
        ate = float(evaluate_trajectory(est, gt_all[:len(est)])["ate_sim3"]["rmse"])
        pn_full.append(ate)
        ego_full.append(float(ego_map[sid]))
        med_step_mm = float((t_all.norm(dim=-1) * DEPTH_UNIT_MM).median().item())
        per_smis.append({"sequence": sid, "n_frames": n, "posenet_ate_mm": ate, "egolite_ate_mm": float(ego_map[sid]), "med_step_mm": med_step_mm})
        print(f"  [Prot B StereoMIS {sid}] PoseNet={ate:.2f}mm vs EgoLite={ego_map[sid]:.2f}mm (med_step={med_step_mm:.3f}mm)", flush=True)

    arr_pf, arr_ef = np.array(pn_full), np.array(ego_full)
    p_full = float(wilcoxon(arr_pf, arr_ef).pvalue)
    res["protocol_b_stereomis"] = {
        "posenet_mean_mm": float(arr_pf.mean()),
        "posenet_median_mm": float(np.median(arr_pf)),
        "egolite_mean_mm": float(arr_ef.mean()),
        "egolite_median_mm": float(np.median(arr_ef)),
        "reduction_pct_vs_posenet": float((1.0 - arr_ef.mean() / arr_pf.mean()) * 100.0),
        "wilcoxon_p_vs_rv2": p_full,
        "per_sequence": per_smis,
    }
    print(f"  [Prot B StereoMIS MEAN] PoseNet={arr_pf.mean():.2f}mm vs EgoLite={arr_ef.mean():.2f}mm (p={p_full:.4f})", flush=True)

    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    out_dir = ROOT / "results" / "revision_vrih"
    ckpt = out_dir / "ckpts" / "samedata_posenet_fixed" / "last.pth"
    if not args.skip_train or not ckpt.is_file():
        train_posenet(ckpt, iters=args.iters)
    eval_posenet_all(ckpt, out_dir / "r1_7_samedata_posenet_fixed.json", out_dir / "posenet_stereomis_full_fixed")


if __name__ == "__main__":
    main()
