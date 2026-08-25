"""VGGT 在 EndoEgoSim 训练集上的轻量微调（M6: 训练收益验证）。

策略: 冻结骨干(DINO+aggregator 在 no_grad 下前向), 只微调 camera_head + depth_head
—— 直接校准域间差异(尤其单目尺度: zero-shot 时 ~29x 偏差)。

监督: 窗口内 GT 位姿(w2c, 首相机归一) + 度量深度, 尺度归一对齐 VGGT 训练约定
(normalize_camera_extrinsics_and_points_batch: 单位平均点距)。
损失: VGGT 官方 MultitaskLoss(camera=5.0 l1, depth=1.0 grad)。

用法:
  python scripts/finetune_vggt.py --iters 1000 --seq-len 16 --lr 1e-4 \
      --out results/finetune/vggt_endo
  # 之后评测: python scripts/baseline_sota.py --method vggt --ckpt <ckpt> ...
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

VGGT_ROOT = r"D:\vggt-main\vggt-main"

# VGGT 输入分辨率(14 的倍数, 640x512 等比缩放)
W_IN, H_IN = 518, 420


def find_vggt_ckpt() -> str:
    import glob as g
    for p in g.glob(os.path.expanduser(
            r"~\.cache\huggingface\hub\models--facebook--VGGT-1B\snapshots\*\model.safetensors")):
        return p
    raise FileNotFoundError("VGGT 权重未找到")


# ---------------------------------------------------------------------------
# 数据: 训练窗口采样
# ---------------------------------------------------------------------------

class SimWindowDataset:
    """从 sim train 序列采样窗口。

    sample_mode:
      dense  — 连续帧 (原行为)
      stride — 固定/随机步长, 覆盖更大基线
      mixed  — 60% dense + 40% stride (推荐, 兼顾平滑视频与 SCARED 关键帧)
    hard_frac: 按 reference_fraction 过采样低参照序列。
    """

    def __init__(self, root="sim_data/train", seq_len=16, min_len=None,
                 sample_mode="mixed", hard_frac=0.35, stride_range=(1, 6),
                 fail_frac=0.0):
        self.seq_dirs = sorted(glob.glob(os.path.join(root, "seq_*")))
        assert self.seq_dirs, f"无训练序列: {root}"
        self.seq_len = seq_len
        self.min_len = min_len or seq_len
        self.sample_mode = sample_mode
        self.hard_frac = float(hard_frac)
        self.fail_frac = float(fail_frac)
        self.stride_range = tuple(stride_range)
        self._rng = np.random.default_rng(1234)
        self._cache = {}
        self.hard_dirs, self.easy_dirs, self.fail_dirs = [], [], []
        for d in self.seq_dirs:
            meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
            refs = [r for r in meta.get("reference_fraction", []) if r is not None]
            mean_r = float(np.mean(refs)) if refs else 0.6
            mt = str(meta.get("motion_type") or "")
            # 与测试集灾难序列同型: 低参照 或 自由运动
            if mean_r < 0.30 or mt == "free":
                self.fail_dirs.append(d)
            (self.hard_dirs if mean_r < 0.45 else self.easy_dirs).append(d)
        if not self.hard_dirs:
            self.hard_dirs = list(self.seq_dirs)
        if not self.easy_dirs:
            self.easy_dirs = list(self.seq_dirs)
        if not self.fail_dirs:
            self.fail_dirs = list(self.hard_dirs)
        print(f"窗口采样: mode={sample_mode} fail={len(self.fail_dirs)} "
              f"hard={len(self.hard_dirs)} easy={len(self.easy_dirs)} "
              f"hard_frac={hard_frac} fail_frac={fail_frac}")

    def __len__(self):
        return 10 ** 9

    def _pick_dir(self):
        u = self._rng.random()
        if self.fail_frac > 0 and u < self.fail_frac:
            pool = self.fail_dirs
        elif u < self.fail_frac + self.hard_frac:
            pool = self.hard_dirs
        else:
            pool = self.easy_dirs
        return pool[int(self._rng.integers(0, len(pool)))]

    def sample(self):
        d = self._pick_dir()
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        n = meta["n_frames"]
        if n < self.min_len:
            return self.sample()
        if self.sample_mode == "dense":
            stride = 1
        elif self.sample_mode == "stride":
            stride = int(self._rng.integers(self.stride_range[0],
                                            self.stride_range[1] + 1))
        else:
            stride = 1 if self._rng.random() > 0.4 else int(
                self._rng.integers(self.stride_range[0], self.stride_range[1] + 1))
        span = (self.seq_len - 1) * stride + 1
        if span > n:
            stride = max(1, (n - 1) // max(self.seq_len - 1, 1))
            span = (self.seq_len - 1) * stride + 1
        a = 0 if n <= span else int(self._rng.integers(0, n - span + 1))
        idx = a + np.arange(self.seq_len) * stride
        idx = np.clip(idx, 0, n - 1)
        intr = json.load(open(os.path.join(d, "intrinsics.json")))
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        sx, sy = W_IN / intr["width"], H_IN / intr["height"]
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1.0]])

        poses = np.loadtxt(os.path.join(d, "pose_c2w.txt")).reshape(-1, 4, 4)
        T_wc = poses[idx]
        G = np.linalg.inv(T_wc[0])
        T_wc = G[None] @ T_wc                     # 窗口首相机归一
        w2c = np.linalg.inv(T_wc)

        imgs, depths = [], []
        for k in idx:
            img = cv2.imread(os.path.join(d, "color", f"{k:06d}.png"))
            img = cv2.resize(img, (W_IN, H_IN), interpolation=cv2.INTER_AREA)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            dep = cv2.imread(os.path.join(d, "depth", f"{k:06d}.png"),
                             cv2.IMREAD_UNCHANGED).astype(np.float32)
            dep = cv2.resize(dep, (W_IN, H_IN), interpolation=cv2.INTER_LINEAR)
            depths.append(dep)
        images = np.stack(imgs)                    # (S,H,W,3)
        depths = np.stack(depths)                  # (S,H,W) mm
        # 运动分解: 去掉独立运动物体像素的深度监督 (label 3);
        # 形变组织(2)与静态参照(1)保留 —— 相对位姿仍有效, 避免 M7c 式观测饥饿
        masks = []
        for ki, k in enumerate(idx):
            valid = depths[ki] > 0
            mp = os.path.join(d, "motion_mask", f"{int(k):06d}.png")
            if os.path.exists(mp):
                mm = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
                if mm is not None:
                    mm = cv2.resize(mm, (W_IN, H_IN), interpolation=cv2.INTER_NEAREST)
                    valid = valid & (mm != 3)
            masks.append(valid)
        masks = np.stack(masks)

        # 世界系点(首相机系): 用于 VGGT 尺度归一
        S = len(idx)
        u, v = np.meshgrid(np.arange(W_IN), np.arange(H_IN))
        cam_pts = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1],
                            np.ones_like(u, dtype=np.float32)], -1) * depths[..., None]
        world_pts = np.einsum('nij,nhwj->nhwi', T_wc[:, :3, :3], cam_pts) + T_wc[:, None, None, :3, 3]

        # 尺度归一只用稀疏点, 避免 16×420×518 的千万级 CPU 张量
        step = 8
        return dict(images=images, depths=depths, masks=masks, w2c=w2c,
                    cam_pts=cam_pts[:, ::step, ::step].astype(np.float32),
                    world_pts=world_pts[:, ::step, ::step].astype(np.float32),
                    masks_scale=masks[:, ::step, ::step], K=K)


def build_batch(sample, device):
    """numpy sample -> VGGT 训练 batch(含官方尺度归一)。"""
    sys.path.insert(0, VGGT_ROOT)
    sys.path.insert(0, os.path.join(VGGT_ROOT, "training"))
    from train_utils.normalization import normalize_camera_extrinsics_and_points_batch

    S = len(sample["w2c"])
    ext = torch.as_tensor(sample["w2c"], dtype=torch.float32)[None, :, :3, :]  # (1,S,3,4)
    depths = torch.as_tensor(sample["depths"], dtype=torch.float32)[None]  # (1,S,H,W)
    masks = torch.as_tensor(sample["masks"], dtype=torch.bool)[None]
    masks_s = torch.as_tensor(sample.get("masks_scale", sample["masks"]),
                              dtype=torch.bool)[None]
    world = torch.as_tensor(sample["world_pts"], dtype=torch.float32)[None]
    cam = torch.as_tensor(sample["cam_pts"], dtype=torch.float32)[None]
    ext_n, _, world_n, depth_n = normalize_camera_extrinsics_and_points_batch(
        ext, cam_points=cam, world_points=world, depths=depths,
        scale_by_points=True, point_masks=masks_s)
    batch = {
        "images": torch.as_tensor(sample["images"], dtype=torch.float32, device=device
                                  ).permute(0, 3, 1, 2)[None],              # (1,S,3,H,W)
        "extrinsics": ext_n.to(device),
        "intrinsics": torch.as_tensor(sample["K"], dtype=torch.float32
                                      )[None, None].repeat(1, S, 1, 1).to(device),
        "depths": depth_n.to(device),
        "point_masks": masks.to(device),
    }
    return batch


import torch  # noqa: E402  (build_batch 之上仅类型引用)


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="results/finetune/vggt_endo")
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze", default="heads_only",
                    choices=["heads_only", "dino", "global"],
                    help="heads_only=只训头; dino=冻结DINO其余全训; global=只训global_blocks+头(跨帧注意力)")
    ap.add_argument("--sample-mode", default="mixed",
                    choices=["dense", "stride", "mixed"],
                    help="dense=连续窗; stride=大基线; mixed=二者混合(推荐)")
    ap.add_argument("--hard-frac", type=float, default=0.35,
                    help="低参照序列过采样比例")
    ap.add_argument("--fail-frac", type=float, default=0.0,
                    help="灾难同型(低参照或free运动)额外过采样比例")
    ap.add_argument("--stride-lo", type=int, default=2)
    ap.add_argument("--stride-hi", type=int, default=6)
    ap.add_argument("--cam-weight", type=float, default=8.0,
                    help="位姿损失权重 (egomotion 主任务, 默认高于官方 5.0)")
    ap.add_argument("--init", default=None, help="从已有微调权重继续")
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"[Ours] 启动 freeze={args.freeze} iters={args.iters} "
          f"sample={args.sample_mode} hard_frac={args.hard_frac} "
          f"cam_w={args.cam_weight} seq_len={args.seq_len}", flush=True)

    sys.path.insert(0, VGGT_ROOT)
    sys.path.insert(0, os.path.join(VGGT_ROOT, "training"))
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    from training.loss import MultitaskLoss

    torch.manual_seed(args.seed)
    device = "cuda"
    model = VGGT()
    if args.init:
        sd = torch.load(args.init, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        model.load_state_dict(sd, strict=False)
        print(f"从 {args.init} 继续")
    else:
        model.load_state_dict(load_file(find_vggt_ckpt()))
    model = model.to(device).eval()
    if args.freeze in ("dino", "global"):
        # bf16 训练: 权重/梯度/优化器状态减半
        model = model.to(torch.bfloat16)

    # 冻结策略
    for name, p in model.named_parameters():
        if args.freeze == "heads_only":
            p.requires_grad_(name.startswith(("camera_head", "depth_head")))
        elif args.freeze == "global":
            # 只训跨帧注意力 + 头: 回传不经过 frame_blocks/patch_embed, 省显存省时
            p.requires_grad_(name.startswith(("aggregator.global_blocks",
                                              "camera_head", "depth_head")))
        else:  # dino: 冻结 aggregator.patch_embed, 其余(aggregator blocks + heads)全训
            p.requires_grad_(not name.startswith("aggregator.patch_embed"))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M")

    loss_fn = MultitaskLoss(camera=dict(weight=args.cam_weight, loss_type="l1"),
                            depth=dict(weight=1.0, gradient_loss_fn="grad",
                                       valid_range=0.98))
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters)

    dataset = SimWindowDataset(seq_len=args.seq_len, sample_mode=args.sample_mode,
                               hard_frac=args.hard_frac, fail_frac=args.fail_frac,
                               stride_range=(args.stride_lo, args.stride_hi))
    val_seqdirs = sorted(glob.glob("sim_data/val/seq_*"))[:args.val_n]

    def forward_loss(batch):
        """前向(官方损失)。heads_only: 聚合器 no_grad; dino: 聚合器带梯度。"""
        images = batch["images"]
        if args.freeze == "heads_only":
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                aggregated_tokens_list, patch_start_idx = model.aggregator(images)
            tokens = [t.detach() for t in aggregated_tokens_list]
        else:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                aggregated_tokens_list, patch_start_idx = model.aggregator(images)
            tokens = aggregated_tokens_list
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pose_enc_list = model.camera_head(tokens)
            depth, depth_conf = model.depth_head(tokens, images=images,
                                                 patch_start_idx=patch_start_idx)
            predictions = {"pose_enc_list": pose_enc_list,
                           "pose_enc": pose_enc_list[-1],
                           "depth": depth, "depth_conf": depth_conf}
        return loss_fn(predictions, batch)

    @torch.no_grad()
    def quick_val():
        """验证集窗口 ATE(Sim3, 归一尺度)。"""
        from endosim.eval.metrics import evaluate_trajectory
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        ates = []
        for d in val_seqdirs:
            s = sample_window(d, args.seq_len)
            batch = build_batch(s, device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                toks, psi = model.aggregator(batch["images"])
                pose_enc_list = model.camera_head(toks)
            ext, _ = pose_encoding_to_extri_intri(pose_enc_list[-1],
                                                  batch["images"].shape[-2:])
            def to44(m):  # (N,3,4) -> (N,4,4)
                out = np.tile(np.eye(4), (len(m), 1, 1))
                out[:, :3, :] = m
                return out
            est = ext.squeeze(0).float().cpu().numpy()          # (S,3,4) w2c
            gt = batch["extrinsics"][0].float().cpu().numpy()
            est_c2w = np.linalg.inv(to44(est))                  # w2c -> c2w
            gt_c2w = np.linalg.inv(to44(gt))
            res = evaluate_trajectory(est_c2w, gt_c2w)
            ates.append(res["ate_sim3"]["rmse"])
        return float(np.mean(ates))

    def sample_window(d, S):
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        n = meta["n_frames"]
        a = 0 if n <= S else int(np.random.default_rng(7).integers(0, n - S + 1))
        idx = np.arange(a, min(a + S, n))
        intr = json.load(open(os.path.join(d, "intrinsics.json")))
        sx, sy = W_IN / intr["width"], H_IN / intr["height"]
        K = np.array([[intr["fx"] * sx, 0, intr["cx"] * sx],
                      [0, intr["fy"] * sy, intr["cy"] * sy], [0, 0, 1.0]])
        poses = np.loadtxt(os.path.join(d, "pose_c2w.txt")).reshape(-1, 4, 4)
        T_wc = poses[idx]
        G = np.linalg.inv(T_wc[0])
        T_wc = G[None] @ T_wc
        imgs, depths = [], []
        for k in idx:
            img = cv2.resize(cv2.imread(os.path.join(d, "color", f"{k:06d}.png")),
                             (W_IN, H_IN), interpolation=cv2.INTER_AREA)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            dep = cv2.imread(os.path.join(d, "depth", f"{k:06d}.png"),
                             cv2.IMREAD_UNCHANGED).astype(np.float32)
            depths.append(cv2.resize(dep, (W_IN, H_IN), interpolation=cv2.INTER_LINEAR))
        images = np.stack(imgs)
        depths = np.stack(depths)
        masks = depths > 0
        u, v = np.meshgrid(np.arange(W_IN), np.arange(H_IN))
        cam_pts = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1],
                            np.ones_like(u, dtype=np.float32)], -1) * depths[..., None]
        world_pts = np.einsum('nij,nhwj->nhwi', T_wc[:, :3, :3], cam_pts) + T_wc[:, None, None, :3, 3]
        return dict(images=images, depths=depths, masks=masks,
                    w2c=np.linalg.inv(T_wc), cam_pts=cam_pts.astype(np.float32),
                    world_pts=world_pts.astype(np.float32), K=K)

    t0 = time.time()
    # 必须 train(): VGGT aggregator 只在 self.training 时做 gradient checkpoint;
    # eval() 会存下全部激活, 24GB 打满且每步数分钟。
    model.train()
    for m in model.modules():
        if m.__class__.__name__.lower().startswith("dropout"):
            m.eval()
    for it in range(1, args.iters + 1):
        sample = dataset.sample()
        batch = build_batch(sample, device)
        losses = forward_loss(batch)
        total = losses["objective"] if "objective" in losses else sum(
            v for v in losses.values() if isinstance(v, torch.Tensor))
        if not isinstance(total, torch.Tensor):
            total = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
        optim.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        sched.step()
        if it % 5 == 0 or it == 1:
            cam = losses.get("loss_camera", torch.tensor(0.0))
            dep = (losses.get("loss_conf_depth", torch.tensor(0.0))
                   + losses.get("loss_reg_depth", torch.tensor(0.0))
                   + losses.get("loss_grad_depth", torch.tensor(0.0)))
            print(f"[{it}/{args.iters}] loss={float(total.detach()):.4f} "
                  f"cam={float(cam.detach()) if torch.is_tensor(cam) else 0:.4f} "
                  f"depth={float(dep.detach()) if torch.is_tensor(dep) else dep:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if args.save_every and it % args.save_every == 0:
            mid = os.path.join(args.out, f"ckpt_{it}.pth")
            torch.save({"model": model.state_dict(), "iters": it,
                        "freeze": args.freeze}, mid)
        if args.val_every and (it % args.val_every == 0 or it == args.iters):
            model.eval()
            ate = quick_val()
            print(f"  [val] 窗口ATE(Sim3)={ate:.4f} (归一尺度)", flush=True)
            model.train()
            for m in model.modules():
                if m.__class__.__name__.lower().startswith("dropout"):
                    m.eval()

    ckpt_path = os.path.join(args.out, "vggt_endo_ft.pth")
    torch.save({"model": model.state_dict(), "iters": args.iters,
                "freeze": args.freeze, "recipe": "ours_m8_motion_mixed"},
               ckpt_path)
    print(f"已保存: {ckpt_path}")


if __name__ == "__main__":
    main()
