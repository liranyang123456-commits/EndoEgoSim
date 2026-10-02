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

from endosim.pseudoheight import pseudo_height_label  # noqa: E402

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
    href_frac: 过采样高参照 (>0.7), 对冲 DROID 在容易条上的中位数优势。
    barrel_policy:
      exclude     — 排除 RGB 与针孔稠密标签不对齐的旧桶形畸变序列（默认）。
      camera_only — 保留图像，但只监督相机与相邻相对位姿。
      legacy      — 复现历史 checkpoint；仍使用不对齐的稠密标签。
    """

    def __init__(self, root="sim_data/train", seq_len=16, min_len=None,
                 sample_mode="mixed", hard_frac=0.35, stride_range=(1, 6),
                 fail_frac=0.0, href_frac=0.0, barrel_policy="exclude"):
        self.root = root
        self.seq_len = seq_len
        self.min_len = min_len or seq_len
        self.sample_mode = sample_mode
        self.hard_frac = float(hard_frac)
        self.fail_frac = float(fail_frac)
        self.href_frac = float(href_frac)
        if barrel_policy not in {"exclude", "camera_only", "legacy"}:
            raise ValueError(f"unsupported barrel_policy: {barrel_policy}")
        self.barrel_policy = barrel_policy
        self.stride_range = tuple(stride_range)
        self._rng = np.random.default_rng(1234)
        self._cache = {}
        self.seq_dirs = []
        self.barrel_dirs = []
        self.hard_dirs, self.easy_dirs, self.fail_dirs, self.href_dirs = [], [], [], []
        self.reindex()

    def reindex(self):
        all_dirs = sorted(glob.glob(os.path.join(self.root, "seq_*")))
        assert all_dirs, f"无训练序列: {self.root}"
        self.seq_dirs, self.barrel_dirs = [], []
        self.hard_dirs, self.easy_dirs, self.fail_dirs, self.href_dirs = [], [], [], []
        for d in all_dirs:
            meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
            k1 = float(meta.get("appearance", {}).get("barrel_k1", 0.0) or 0.0)
            is_barrel = abs(k1) > 1e-8
            if is_barrel:
                self.barrel_dirs.append(d)
                if self.barrel_policy == "exclude":
                    continue
            self.seq_dirs.append(d)
            refs = [r for r in meta.get("reference_fraction", []) if r is not None]
            mean_r = float(np.mean(refs)) if refs else 0.6
            mt = str(meta.get("motion_type") or "")
            if mean_r < 0.30 or mt == "free":
                self.fail_dirs.append(d)
            if mean_r >= 0.70:
                self.href_dirs.append(d)
            (self.hard_dirs if mean_r < 0.45 else self.easy_dirs).append(d)
        if not self.hard_dirs:
            self.hard_dirs = list(self.seq_dirs)
        if not self.easy_dirs:
            self.easy_dirs = list(self.seq_dirs)
        if not self.fail_dirs:
            self.fail_dirs = list(self.hard_dirs)
        if not self.href_dirs:
            self.href_dirs = list(self.easy_dirs)
        print(f"窗口采样: mode={self.sample_mode} n={len(self.seq_dirs)} "
              f"fail={len(self.fail_dirs)} hard={len(self.hard_dirs)} "
              f"href={len(self.href_dirs)} easy={len(self.easy_dirs)} "
              f"barrel={len(self.barrel_dirs)} policy={self.barrel_policy} "
              f"fail_frac={self.fail_frac} hard_frac={self.hard_frac} "
              f"href_frac={self.href_frac}", flush=True)

    def __len__(self):
        return 10 ** 9

    def _pick_dir(self):
        u = self._rng.random()
        a, b, c = self.fail_frac, self.fail_frac + self.hard_frac, (
            self.fail_frac + self.hard_frac + self.href_frac)
        if self.fail_frac > 0 and u < a:
            pool = self.fail_dirs
        elif u < b:
            pool = self.hard_dirs
        elif self.href_frac > 0 and u < c:
            pool = self.href_dirs
        else:
            pool = self.easy_dirs
        return pool[int(self._rng.integers(0, len(pool)))]

    def sample(self):
        d = self._pick_dir()
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        barrel_k1 = float(meta.get("appearance", {}).get("barrel_k1", 0.0) or 0.0)
        camera_only_barrel = (
            self.barrel_policy == "camera_only" and abs(barrel_k1) > 1e-8
        )
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
            if camera_only_barrel:
                dep = np.zeros((H_IN, W_IN), dtype=np.float32)
            else:
                dep = cv2.imread(os.path.join(d, "depth", f"{k:06d}.png"),
                                 cv2.IMREAD_UNCHANGED).astype(np.float32)
                dep = cv2.resize(dep, (W_IN, H_IN), interpolation=cv2.INTER_LINEAR)
            depths.append(dep)
        images = np.stack(imgs)                    # (S,H,W,3)
        depths = np.stack(depths)                  # (S,H,W) mm
        ph = np.stack([pseudo_height_label(im) for im in imgs])  # (S,H,W) [0,1]
        if camera_only_barrel:
            masks = np.zeros_like(depths, dtype=bool)
            masks[:, H_IN // 2 - 16:H_IN // 2 + 16,
                  W_IN // 2 - 16:W_IN // 2 + 16] = True
            return dict(
                images=images,
                depths=depths,
                ph=ph,
                ml=np.full((len(idx), H_IN, W_IN), -100, np.int64),
                masks=masks,
                w2c=w2c,
                cam_pts=np.zeros((len(idx), 8, 8, 3), np.float32),
                world_pts=np.zeros((len(idx), 8, 8, 3), np.float32),
                masks_scale=masks[:, ::8, ::8],
                K=K,
                camera_only=True,
                source_kind="barrel_camera_only",
            )
        # 运动分解: 去掉独立运动物体像素的深度监督 (label 3);
        # 形变组织(2)与静态参照(1)保留 —— 相对位姿仍有效, 避免 M7c 式观测饥饿
        masks = []
        ml_labels = []
        for ki, k in enumerate(idx):
            valid = depths[ki] > 0
            mp = os.path.join(d, "motion_mask", f"{int(k):06d}.png")
            ml = np.full((H_IN, W_IN), -100, np.int64)  # -100 = 忽略（PyTorch 默认 ignore_index）
            if os.path.exists(mp):
                mm = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
                if mm is not None:
                    mm = cv2.resize(mm, (W_IN, H_IN), interpolation=cv2.INTER_NEAREST)
                    valid = valid & (mm != 3)
                    ml = np.where(np.isin(mm, [1, 2, 3]), mm - 1, -100).astype(np.int64)
            masks.append(valid)
            ml_labels.append(ml)
        masks = np.stack(masks)
        ml_labels = np.stack(ml_labels)            # (S,H,W) in {-1,0,1,2}

        # 世界系点(首相机系): 用于 VGGT 尺度归一
        S = len(idx)
        u, v = np.meshgrid(np.arange(W_IN), np.arange(H_IN))
        cam_pts = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1],
                            np.ones_like(u, dtype=np.float32)], -1) * depths[..., None]
        world_pts = np.einsum('nij,nhwj->nhwi', T_wc[:, :3, :3], cam_pts) + T_wc[:, None, None, :3, 3]

        # 尺度归一只用稀疏点, 避免 16×420×518 的千万级 CPU 张量
        step = 8
        return dict(images=images, depths=depths, ph=ph, ml=ml_labels, masks=masks, w2c=w2c,
                    cam_pts=cam_pts[:, ::step, ::step].astype(np.float32),
                    world_pts=world_pts[:, ::step, ::step].astype(np.float32),
                    masks_scale=masks[:, ::step, ::step], K=K,
                    camera_only=False)


class RealPoseWindowDataset:
    """C3VD 等路径索引序列: 只监督相机 (无仿真深度时不走 depth head)。"""

    def __init__(self, root="sim_data/real_refs", seq_len=16, stride_range=(2, 8)):
        from endosim.eval.protocol import list_color_frames
        self.seq_len = seq_len
        self.stride_range = tuple(stride_range)
        self._rng = np.random.default_rng(7)
        self.items = []
        if not os.path.isdir(root):
            return
        for name in sorted(os.listdir(root)):
            if name.startswith(("scared_", "stereomis_", "endomapper_")):
                continue
            d = os.path.join(root, name)
            pose_p = os.path.join(d, "pose_c2w.txt")
            idx_p = os.path.join(d, "color_index.json")
            if not os.path.isdir(d) or not os.path.isfile(pose_p):
                continue
            if os.path.isfile(idx_p):
                raw = json.load(open(idx_p, encoding="utf-8"))
                if isinstance(raw, dict) and raw.get("type") == "video":
                    continue
            poses = np.loadtxt(pose_p).reshape(-1, 4, 4)
            frames = list_color_frames(d)
            n = min(len(poses), len(frames))
            if n < seq_len:
                continue
            intr_p = os.path.join(d, "intrinsics.json")
            intr = json.load(open(intr_p, encoding="utf-8")) if os.path.isfile(intr_p) else {}
            self.items.append({
                "dir": d, "poses": poses[:n], "frames": frames[:n], "intr": intr,
            })
        print(f"真实域位姿窗: {len(self.items)} 条 @ {root}", flush=True)

    def __bool__(self):
        return bool(self.items)

    def sample(self):
        rec = self.items[int(self._rng.integers(0, len(self.items)))]
        n = len(rec["frames"])
        stride = int(self._rng.integers(self.stride_range[0], self.stride_range[1] + 1))
        span = (self.seq_len - 1) * stride + 1
        if span > n:
            stride = max(1, (n - 1) // max(self.seq_len - 1, 1))
            span = (self.seq_len - 1) * stride + 1
        a = 0 if n <= span else int(self._rng.integers(0, n - span + 1))
        idx = np.clip(a + np.arange(self.seq_len) * stride, 0, n - 1)
        T_wc = rec["poses"][idx]
        G = np.linalg.inv(T_wc[0])
        T_wc = G[None] @ T_wc
        imgs = []
        for k in idx:
            img = cv2.imread(rec["frames"][int(k)])
            if img is None:
                return self.sample()
            img = cv2.resize(img, (W_IN, H_IN), interpolation=cv2.INTER_AREA)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        images = np.stack(imgs)
        ph = np.stack([pseudo_height_label(im) for im in imgs])
        depths = np.zeros((len(idx), H_IN, W_IN), np.float32)
        masks = np.zeros_like(depths, dtype=bool)
        masks[:, H_IN // 2 - 16:H_IN // 2 + 16, W_IN // 2 - 16:W_IN // 2 + 16] = True
        intr = rec["intr"]
        fx = float(intr.get("fx", 0.9 * W_IN))
        fy = float(intr.get("fy", 0.9 * H_IN))
        cx = float(intr.get("cx", W_IN / 2))
        cy = float(intr.get("cy", H_IN / 2))
        iw = float(intr.get("width", W_IN)) or W_IN
        ih = float(intr.get("height", H_IN)) or H_IN
        sx, sy = W_IN / iw, H_IN / ih
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1.0]])
        return dict(images=images, depths=depths, ph=ph,
                    ml=np.full((len(idx), H_IN, W_IN), -100, np.int64),
                    masks=masks, w2c=np.linalg.inv(T_wc),
                    cam_pts=np.zeros((len(idx), 8, 8, 3), np.float32),
                    world_pts=np.zeros((len(idx), 8, 8, 3), np.float32),
                    masks_scale=masks[:, ::8, ::8], K=K, camera_only=True)


def build_batch(sample, device, ph_input: bool = False):
    """numpy sample -> VGGT 训练 batch(含官方尺度归一)。

    ph_input=True 时把 pseudo-height 拼为第 4 输入通道（A 方案）；
    否则 ph 仅作为辅助监督标签随 batch["ph"] 返回（B 方案）。"""
    sys.path.insert(0, VGGT_ROOT)
    sys.path.insert(0, os.path.join(VGGT_ROOT, "training"))
    from train_utils.normalization import normalize_camera_extrinsics_and_points_batch

    S = len(sample["w2c"])
    ext = torch.as_tensor(sample["w2c"], dtype=torch.float32)[None, :, :3, :]  # (1,S,3,4)
    images = torch.as_tensor(sample["images"], dtype=torch.float32, device=device
                             ).permute(0, 3, 1, 2)[None]
    ph = torch.as_tensor(sample["ph"], dtype=torch.float32, device=device)[None]  # (1,S,H,W)
    ml_arr = sample.get("ml")
    if ml_arr is None:
        S0, H0, W0 = sample["images"].shape[:3]
        ml_arr = np.full((S0, H0, W0), -100, np.int64)
    ml = torch.as_tensor(ml_arr, dtype=torch.long, device=device)[None]  # (1,S,H,W)
    if ph_input:
        images = torch.cat([images, ph[:, :, None]], dim=2)  # (1,S,4,H,W)
    if sample.get("camera_only"):
        t = ext[0, :, :3, 3]
        scale = t.norm(dim=-1).mean().clamp(min=1e-3)
        ext_n = ext.clone()
        ext_n[..., :3, 3] = ext_n[..., :3, 3] / scale
        masks = torch.as_tensor(sample["masks"], dtype=torch.bool)[None]
        batch = {
            "images": images,
            "extrinsics": ext_n.to(device),
            "intrinsics": torch.as_tensor(sample["K"], dtype=torch.float32
                                          )[None, None].repeat(1, S, 1, 1).to(device),
            "point_masks": masks.to(device),
            "ph": ph,
            "ml": ml,
            "camera_only": True,
        }
        return batch
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
        "images": images,
        "extrinsics": ext_n.to(device),
        "intrinsics": torch.as_tensor(sample["K"], dtype=torch.float32
                                      )[None, None].repeat(1, S, 1, 1).to(device),
        "depths": depth_n.to(device),
        "point_masks": masks.to(device),
        "ph": ph,
        "ml": ml,
        "camera_only": False,
    }
    return batch


def relative_se3_loss(pred_w2c, gt_w2c):
    """相邻帧相对位姿 L1(平移) + 测地旋转, pred/gt 为 (1,S,3,4) w2c。"""
    def to44(m):
        b, s = m.shape[:2]
        out = m.new_zeros(b, s, 4, 4)
        out[..., 3, 3] = 1
        out[..., :3, :] = m
        return out
    p, g = to44(pred_w2c), to44(gt_w2c)
    rel_p = p[:, 1:] @ torch.linalg.inv(p[:, :-1])
    rel_g = g[:, 1:] @ torch.linalg.inv(g[:, :-1])
    t = (rel_p[..., :3, 3] - rel_g[..., :3, 3]).abs().mean()
    r = rel_p[..., :3, :3] @ rel_g[..., :3, :3].transpose(-1, -2)
    tr = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
    ang = torch.acos(((tr - 1) * 0.5).clamp(-1 + 1e-5, 1 - 1e-5))
    return t + ang.mean()


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
    ap.add_argument("--href-frac", type=float, default=0.0,
                    help="高参照(>0.7)过采样, 拉近 DROID 中位数")
    ap.add_argument("--real-frac", type=float, default=0.0,
                    help="混入 C3VD 等真实域位姿窗的比例")
    ap.add_argument("--real-root", default="sim_data/real_refs")
    ap.add_argument("--rel-weight", type=float, default=0.0,
                    help="相邻相对 SE3 一致性损失权重")
    ap.add_argument("--rescan-every", type=int, default=0,
                    help="每隔 N step 重扫 train 目录(边生成边训)")
    ap.add_argument("--stride-lo", type=int, default=2)
    ap.add_argument("--stride-hi", type=int, default=6)
    ap.add_argument("--cam-weight", type=float, default=8.0,
                    help="位姿损失权重 (egomotion 主任务, 默认高于官方 5.0)")
    ap.add_argument(
        "--barrel-policy",
        choices=["exclude", "camera_only", "legacy"],
        default="exclude",
        help=("旧 RGB-only 桶形畸变序列：默认排除；camera_only 仅保留相机/"
              "相对位姿监督；legacy 仅用于复现历史权重"),
    )
    ap.add_argument("--init", default=None, help="从已有微调权重继续")
    ap.add_argument("--ph-weight", type=float, default=0.0,
                    help=">0 启用 pseudo-height 辅助监督头（B 方案），权重")
    ap.add_argument("--ph-input", action="store_true",
                    help="把 pseudo-height 拼为第 4 输入通道（A 方案）")
    ap.add_argument("--ml-weight", type=float, default=0.0,
                    help=">0 启用运动层分解辅助头（3 类分割，交叉熵）")
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"[MD-VGGT] 启动 freeze={args.freeze} iters={args.iters} "
          f"sample={args.sample_mode} hard_frac={args.hard_frac} "
          f"fail_frac={args.fail_frac} href_frac={args.href_frac} "
          f"real_frac={args.real_frac} rel_w={args.rel_weight} "
          f"cam_w={args.cam_weight} seq_len={args.seq_len} "
          f"barrel_policy={args.barrel_policy}", flush=True)

    sys.path.insert(0, VGGT_ROOT)
    sys.path.insert(0, os.path.join(VGGT_ROOT, "training"))
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    from training.loss import MultitaskLoss

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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

    # A 方案：patch_embed.proj 扩为 4 通道（前 3 通道继承，第 4 通道零初始化）
    if args.ph_input:
        pe = model.aggregator.patch_embed
        # dinov2: DinoVisionTransformer.patch_embed.proj；conv: PatchEmbed.proj
        proj_host = pe.patch_embed if hasattr(pe, "patch_embed") else pe
        old_proj = proj_host.proj
        new_proj = torch.nn.Conv2d(
            4, old_proj.out_channels, kernel_size=old_proj.kernel_size,
            stride=old_proj.stride, padding=old_proj.padding)
        with torch.no_grad():
            new_proj.weight[:, :3] = old_proj.weight
            new_proj.weight[:, 3:] = 0.0
            new_proj.bias.copy_(old_proj.bias)
        proj_host.proj = new_proj.to(device)
        print("pseudo-height 输入通道已启用（patch_embed.proj 3->4，零初始化）")

    # B 方案：pseudo-height 辅助监督头（仅训练正则，推理不使用）
    if args.ph_weight > 0:
        from vggt.heads.dpt_head import DPTHead
        model.ph_head = DPTHead(dim_in=2 * 1024, output_dim=2,
                                activation="relu", conf_activation="expp1")
        model.ph_head = model.ph_head.to(device)
        print(f"pseudo-height 辅助头已启用（weight={args.ph_weight}）")

    # 运动层分解头：网络显式输出 静态/变形/器械 三类概率（学习的运动分解）
    if args.ml_weight > 0:
        from vggt.heads.dpt_head import DPTHead
        model.ml_head = DPTHead(dim_in=2 * 1024, output_dim=4,
                                activation="linear", conf_activation="expp1")
        model.ml_head = model.ml_head.to(device)
        print(f"运动层分解头已启用（weight={args.ml_weight}）")

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
    # pseudo-height 模块始终可训
    if args.ph_weight > 0:
        for p in model.ph_head.parameters():
            p.requires_grad_(True)
    if args.ml_weight > 0:
        for p in model.ml_head.parameters():
            p.requires_grad_(True)
    if args.ph_input:
        _pe = model.aggregator.patch_embed
        _proj_host = _pe.patch_embed if hasattr(_pe, "patch_embed") else _pe
        for p in _proj_host.proj.parameters():
            p.requires_grad_(True)
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
                               href_frac=args.href_frac,
                               stride_range=(args.stride_lo, args.stride_hi),
                               barrel_policy=args.barrel_policy)
    real_ds = RealPoseWindowDataset(root=args.real_root, seq_len=args.seq_len) \
        if args.real_frac > 0 else None
    val_seqdirs = sorted(glob.glob("sim_data/val/seq_*"))
    if args.barrel_policy == "exclude":
        val_seqdirs = [
            d for d in val_seqdirs
            if abs(float(json.load(open(
                os.path.join(d, "meta.json"), encoding="utf-8"
            )).get("appearance", {}).get("barrel_k1", 0.0) or 0.0)) <= 1e-8
        ]
    val_seqdirs = val_seqdirs[:args.val_n]

    def forward_loss(batch):
        """前向(官方损失)。heads_only: 聚合器 no_grad; dino: 聚合器带梯度。"""
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
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
            predictions = {"pose_enc_list": pose_enc_list,
                           "pose_enc": pose_enc_list[-1]}
            if not batch.get("camera_only"):
                depth, depth_conf = model.depth_head(tokens, images=images,
                                                     patch_start_idx=patch_start_idx)
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
        losses = loss_fn(predictions, batch)
        if args.ph_weight > 0 and not batch.get("camera_only"):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ph_pred, ph_conf = model.ph_head(tokens, images=images,
                                                 patch_start_idx=patch_start_idx)
            ph_gt = batch["ph"].float()
            ph_pred = ph_pred.float()
            if ph_pred.dim() == 5 and ph_pred.shape[-1] == 1:
                ph_pred = ph_pred.squeeze(-1)      # (1,S,H,W,1) -> (1,S,H,W)
            elif ph_pred.dim() == 5:
                ph_pred = ph_pred.squeeze(2)       # (1,S,1,H,W) -> (1,S,H,W)
            ph_conf = ph_conf.float()
            while ph_conf.dim() > ph_gt.dim():
                ph_conf = ph_conf.squeeze(2) if ph_conf.shape[2] == 1 else ph_conf.squeeze(-1)
            err = (ph_pred - ph_gt).abs()
            # 与 depth 相同的 confidence 正则形式: c*e - 0.2*log(c)
            loss_ph = (ph_conf * err - 0.2 * torch.log(ph_conf.clamp(min=1e-6))).mean()
            losses["loss_ph"] = loss_ph
            losses["objective"] = losses["objective"] + args.ph_weight * loss_ph
        if args.ml_weight > 0 and not batch.get("camera_only"):
            import torch.nn.functional as F
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ml_pred, _ml_conf = model.ml_head(tokens, images=images,
                                                  patch_start_idx=patch_start_idx)
            ml_gt = batch["ml"]                       # (1,S,H,W) in {-1,0,1,2}
            ml_pred = ml_pred.float()
            if ml_pred.dim() == 5 and ml_pred.shape[-1] == 3:
                logits = ml_pred[0].permute(0, 3, 1, 2)   # (S,H,W,3)->(S,3,H,W)
            elif ml_pred.dim() == 5 and ml_pred.shape[2] == 3:
                logits = ml_pred[0]                       # (1,S,3,H,W)->(S,3,H,W)
            else:
                raise RuntimeError(f"unexpected ml_pred shape {tuple(ml_pred.shape)}")
            tgt = ml_gt.squeeze(0)
            # 防御：越界值归 ignore，打印异常样本
            bad = ((tgt < 0) & (tgt != -100)) | (tgt > 2)
            if bad.any():
                print(f"[ml] BAD target values {torch.unique(tgt).tolist()} -> clamp to ignore")
                tgt = tgt.clone()
                tgt[bad] = -100
            if logits.shape[2:] != tgt.shape[1:]:
                print(f"[ml] shape mismatch logits {tuple(logits.shape)} tgt {tuple(tgt.shape)}")
            loss_ml = F.cross_entropy(logits.clamp(-20, 20), tgt, ignore_index=-100)
            losses["loss_ml"] = loss_ml
            losses["objective"] = losses["objective"] + args.ml_weight * loss_ml
        if args.rel_weight > 0:
            ext, _ = pose_encoding_to_extri_intri(pose_enc_list[-1],
                                                  images.shape[-2:])
            rel = relative_se3_loss(ext.float(), batch["extrinsics"].float())
            losses["loss_rel"] = rel
            losses["objective"] = losses["objective"] + args.rel_weight * rel
        return losses

    @torch.no_grad()
    def quick_val():
        """验证集窗口 ATE(Sim3, 归一尺度)。"""
        from endosim.eval.metrics import evaluate_trajectory
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        ates = []
        for d in val_seqdirs:
            s = sample_window(d, args.seq_len)
            batch = build_batch(s, device, ph_input=args.ph_input)
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
        ph = np.stack([pseudo_height_label(im) for im in imgs])
        masks = depths > 0
        u, v = np.meshgrid(np.arange(W_IN), np.arange(H_IN))
        cam_pts = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1],
                            np.ones_like(u, dtype=np.float32)], -1) * depths[..., None]
        world_pts = np.einsum('nij,nhwj->nhwi', T_wc[:, :3, :3], cam_pts) + T_wc[:, None, None, :3, 3]
        return dict(images=images, depths=depths, ph=ph, masks=masks,
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
        if args.rescan_every and it % args.rescan_every == 0:
            dataset.reindex()
        if real_ds and np.random.random() < args.real_frac:
            sample = real_ds.sample()
        else:
            sample = dataset.sample()
        batch = build_batch(sample, device, ph_input=args.ph_input)
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
                        "freeze": args.freeze,
                        "barrel_policy": args.barrel_policy,
                        "seed": args.seed}, mid)
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
                "freeze": args.freeze,
                "recipe": "ours_v4_href_real_rel",
                "barrel_policy": args.barrel_policy,
                "seed": args.seed},
               ckpt_path)
    print(f"已保存: {ckpt_path}")


if __name__ == "__main__":
    main()
