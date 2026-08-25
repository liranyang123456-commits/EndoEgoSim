"""真实纹理库: 从本地真实内窥镜图像构建可平铺组织纹理。

数据源（只读路径引用, 绝不拷贝）:
- CholecSeg8k: D:\\datasets\\CholecSeg8k\\images (8080帧, 854x480) + masks(器械过滤)
- SCARED d1_key1: D:\\LumenGSLAM\\dataset\\SCARED\\d1_key1\\color (40帧)
- MIS 视频: D:\\Exp_MIS_ChessBox_Datas\\MIS_Videso_1_6\\extracted_frames*\\extracted_frames

纹理构造: 随机裁剪 -> 3x3 马赛克拼贴(软融合) -> HSV 抖动 -> 输出 uint8 纹理。
纹理源按"源视频ID"划分 train/test 防泄漏。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

CHOLEC_DIR = r"D:\datasets\CholecSeg8k"
CHOLEC_IMG = os.path.join(CHOLEC_DIR, "images")
CHOLEC_MASK = os.path.join(CHOLEC_DIR, "masks")
SCARED_COLOR = r"D:\LumenGSLAM\dataset\SCARED\d1_key1\color"
MIS_ROOT = r"D:\Exp_MIS_ChessBox_Datas\MIS_Videso_1_6"
# E盘真实数据集（下消化道域, 纹理多样性）
HYPERKVASIR_IMG = (r"E:\World_Agent_Enoscopy\datasets\HyperKvasir"
                   r"\hyper-kvasir-labeled-images\labeled-images\lower-gi-tract")


@dataclass
class TextureBank:
    """按 split 管理的纹理源池。"""
    train_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    _cache: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"train_textures": len(self.train_files),
                "test_textures": len(self.test_files)}


def _list_cholec() -> list[tuple[str, str | None]]:
    """返回 (image_path, mask_path|None)，按文件名排序。"""
    out = []
    if not os.path.isdir(CHOLEC_IMG):
        return out
    for fn in sorted(os.listdir(CHOLEC_IMG)):
        if not fn.endswith(".png"):
            continue
        img = os.path.join(CHOLEC_IMG, fn)
        # mask 同名规则探测
        mask = None
        if os.path.isdir(CHOLEC_MASK):
            for cand in (fn, fn.replace("_endo.png", "_endo_mask.png"),
                         re.sub(r"_endo\.png$", "_mask.png", fn)):
                p = os.path.join(CHOLEC_MASK, cand)
                if os.path.exists(p):
                    mask = p
                    break
        out.append((img, mask))
    return out


def _list_scared() -> list[tuple[str, None]]:
    if not os.path.isdir(SCARED_COLOR):
        return []
    return [(os.path.join(SCARED_COLOR, fn), None)
            for fn in sorted(os.listdir(SCARED_COLOR)) if fn.endswith(".png")]


def _list_mis() -> list[tuple[str, None]]:
    out = []
    if not os.path.isdir(MIS_ROOT):
        return out
    for i in range(1, 7):
        d = os.path.join(MIS_ROOT, f"extracted_frames{i}", "extracted_frames")
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".jpg", ".png")):
                out.append((os.path.join(d, fn), None))
    return out


def _list_hyperkvasir() -> list[tuple[str, None]]:
    """HyperKvasir 下消化道标注图像（cecum/rectum/sigmoid等, 与结肠仿真同域）。"""
    out = []
    if not os.path.isdir(HYPERKVASIR_IMG):
        return out
    for root, _, files in os.walk(HYPERKVASIR_IMG):
        if "findings" in root:  # 病变图像也可作纹理, 但保守排除标注框污染
            continue
        for fn in sorted(files):
            if fn.lower().endswith(".jpg"):
                out.append((os.path.join(root, fn), None))
    return out


def build_bank(train_frac: float = 0.7, seed: int = 0) -> TextureBank:
    """汇总所有纹理源并按源视频划分 train/test。

    防泄漏原则: CholecSeg8k 的同一 video 的帧必须在同一 split。
    """
    rng = np.random.default_rng(seed)
    bank = TextureBank()

    cholec = _list_cholec()
    # video01_frame_100_endo.png -> video01
    by_video: dict[str, list] = {}
    for img, mask in cholec:
        m = re.match(r"(video\d+)", os.path.basename(img))
        key = m.group(1) if m else os.path.basename(img)[:6]
        by_video.setdefault(key, []).append((img, mask))
    vids = sorted(by_video)
    rng.shuffle(vids)
    n_train_v = max(1, int(round(len(vids) * train_frac)))
    for v in vids[:n_train_v]:
        bank.train_files.extend(img for img, _ in by_video[v])
    for v in vids[n_train_v:]:
        bank.test_files.extend(img for img, _ in by_video[v])

    scared = _list_scared()
    mis = _list_mis()
    hk = _list_hyperkvasir()
    # SCARED/MIS/HyperKvasir 数量控制: 随机采样上限, 避免单一来源主导
    rng2 = np.random.default_rng(seed + 1)
    if len(hk) > 4000:
        idx = rng2.choice(len(hk), 4000, replace=False)
        hk = [hk[i] for i in idx]
    # HyperKvasir 按目录(解剖部位)划分 train/test
    hk_by_part: dict[str, list] = {}
    for img, _ in hk:
        part = os.path.basename(os.path.dirname(img))
        hk_by_part.setdefault(part, []).append(img)
    parts = sorted(hk_by_part)
    rng.shuffle(parts)
    n_train_p = max(1, int(round(len(parts) * train_frac)))
    for pt in parts[:n_train_p]:
        bank.train_files.extend(hk_by_part[pt])
    for pt in parts[n_train_p:]:
        bank.test_files.extend(hk_by_part[pt])
    # SCARED/MIS 数量少, 全部给 train（真实锚定测试另有专门协议）
    bank.train_files.extend(img for img, _ in scared)
    bank.train_files.extend(img for img, _ in mis)
    return bank


def _load_rgb(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _instrument_fraction(mask_path: str | None, crop_box: tuple) -> float:
    """裁剪区域中器械像素占比（CholecSeg8k 13类中器械类别id>0）。"""
    if mask_path is None:
        return 0.0
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return 0.0
    x, y, w, h = crop_box
    if m.shape[0] < y + h or m.shape[1] < x + w:
        return 0.0
    patch = m[y:y + h, x:x + w]
    return float((patch > 0).mean())


def make_texture(rng: np.random.Generator, bank: TextureBank, split: str = "train",
                 size: int = 1024, crop: int = 256,
                 hsv_jitter: float = 0.06,
                 instrument_max_frac: float = 0.05,
                 max_tries: int = 40) -> np.ndarray:
    """生成一张 size x size 可平铺组织纹理（3x3 马赛克 + 软融合 + HSV抖动）。"""
    files = bank.train_files if split == "train" else bank.test_files
    if not files:
        # 无真实源时回退: 程序化粉红噪声纹理
        return _procedural_texture(rng, size)
    cell = size // 3
    mosaic = np.zeros((size, size, 3), dtype=np.float32)
    weight = np.zeros((size, size, 1), dtype=np.float32)
    # 每个马赛克格取一张随机源图裁剪
    for gy in range(3):
        for gx in range(3):
            patch = None
            for _ in range(max_tries):
                f_idx = rng.integers(0, len(files))
                img, mask = files[f_idx], None
                # bank 只存了 image 路径; mask 由同名规则重探测
                m = re.match(r"(video\d+)", os.path.basename(img))
                mask = None
                if m and os.path.isdir(CHOLEC_MASK):
                    for cand in (os.path.basename(img),
                                 os.path.basename(img).replace("_endo.png", "_endo_mask.png")):
                        p = os.path.join(CHOLEC_MASK, cand)
                        if os.path.exists(p):
                            mask = p
                            break
                rgb = _cache_load(bank, img)
                if rgb is None:
                    continue
                H, W = rgb.shape[:2]
                if H < crop or W < crop:
                    continue
                x = int(rng.integers(0, W - crop + 1))
                y = int(rng.integers(0, H - crop + 1))
                if _instrument_fraction(mask, (x, y, crop, crop)) > instrument_max_frac:
                    continue
                patch = cv2.resize(rgb[y:y + crop, x:x + crop].astype(np.float32),
                                   (cell, cell), interpolation=cv2.INTER_AREA)
                break
            if patch is None:
                patch = _procedural_texture(rng, cell).astype(np.float32)
            y0, x0 = gy * cell, gx * cell
            mosaic[y0:y0 + cell, x0:x0 + cell] = patch
            weight[y0:y0 + cell, x0:x0 + cell] = 1.0
    # 软融合: 高斯模糊权重后归一化拼接（消格缝）
    wf = cv2.GaussianBlur(weight, (0, 0), sigmaX=cell * 0.25)
    blur = cv2.GaussianBlur(mosaic, (0, 0), sigmaX=cell * 0.25)
    tex = np.where(weight > 0.5, mosaic, blur)
    tex = np.clip(tex, 0, 255).astype(np.uint8)

    # HSV 抖动（域随机化）
    hsv = cv2.cvtColor(tex, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-25, 25) * hsv_jitter * 10) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + rng.uniform(-hsv_jitter, hsv_jitter)), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + rng.uniform(-hsv_jitter, hsv_jitter)), 0, 255)
    tex = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return tex


def _cache_load(bank: TextureBank, path: str) -> np.ndarray | None:
    if path in bank._cache:
        return bank._cache[path]
    img = _load_rgb(path)
    if len(bank._cache) > 256:
        bank._cache.clear()
    bank._cache[path] = img
    return img


def _procedural_texture(rng: np.random.Generator, size: int = 1024) -> np.ndarray:
    """程序化组织纹理: 粉红底 + 分形斑驳 + 血管线。"""
    base = np.array([205, 138, 128], dtype=np.float32)
    noise = np.zeros((size, size), dtype=np.float32)
    for o in range(5):
        f = 2 ** o
        gy = rng.uniform(0, 2 * np.pi)
        gx = rng.uniform(0, 2 * np.pi)
        ys = np.arange(size) / size * f
        xs = np.arange(size) / size * f
        noise += (0.5 ** o) * np.sin(2 * np.pi * ys[:, None] + gy) * np.cos(2 * np.pi * xs[None] + gx)
    noise = (noise / noise.max() * 0.5 + 0.5)
    tex = base[None, None] * (0.82 + 0.30 * noise[..., None])
    # 血管: 随机游走暗红线
    n_ves = rng.integers(6, 14)
    for _ in range(n_ves):
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        ang = rng.uniform(0, 2 * np.pi)
        pts = [(x, y)]
        for _ in range(rng.integers(20, 60)):
            ang += rng.normal(0, 0.25)
            x += np.cos(ang) * 6
            y += np.sin(ang) * 6
            pts.append((x, y))
        pts = np.array(pts, np.int32)
        cv2.polylines(tex, [pts], False, (120, 55, 60), thickness=int(rng.integers(1, 3)))
    return np.clip(tex, 0, 255).astype(np.uint8)
