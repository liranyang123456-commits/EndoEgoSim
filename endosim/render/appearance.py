"""外观级域随机化: 色彩迁移 / 圆形渐晕 / 桶形畸变。

色彩迁移与光学退化只作用于 RGB, 深度/位姿/光流 GT 仍按针孔模型计算
——这是有意的 sim-to-real 外观扰动, 与传感器噪声同层。
真实参照图通过 TextureBank 路径访问, 不拷贝像素到本仓库。
"""
from __future__ import annotations

import cv2
import numpy as np


def pick_ref_image(bank, split: str, rng: np.random.Generator,
                   min_size: int = 128) -> np.ndarray | None:
    """从纹理库抽一张真实内窥镜图 (float RGB [0,1]), 路径只读。"""
    from ..scene.texture import _load_rgb
    files = bank.train_files if split == "train" else bank.test_files
    if not files:
        return None
    for _ in range(16):
        path = files[int(rng.integers(0, len(files)))]
        img = _load_rgb(path)
        if img is None or min(img.shape[:2]) < min_size:
            continue
        return img.astype(np.float32) / 255.0
    return None


def reinhard_color_transfer(src: np.ndarray, ref: np.ndarray,
                            strength: float = 0.5) -> np.ndarray:
    """Lab 空间 Reinhard 均值/方差对齐, strength∈[0,1] 混合原图。

    src/ref: float RGB [0,1] 或 uint8; 返回 float RGB [0,1]。
    """
    def _u8(x):
        if x.dtype == np.uint8:
            return x
        return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)

    src_u, ref_u = _u8(src), _u8(ref)
    if ref_u.shape[:2] != src_u.shape[:2]:
        ref_u = cv2.resize(ref_u, (src_u.shape[1], src_u.shape[0]),
                           interpolation=cv2.INTER_AREA)
    src_lab = cv2.cvtColor(src_u, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_u, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = src_lab.copy()
    s = float(np.clip(strength, 0.0, 1.0))
    for c in range(3):
        sm, ss = float(src_lab[..., c].mean()), float(src_lab[..., c].std()) + 1e-6
        rm, rs = float(ref_lab[..., c].mean()), float(ref_lab[..., c].std()) + 1e-6
        mapped = (src_lab[..., c] - sm) * (rs / ss) + rm
        out[..., c] = src_lab[..., c] * (1.0 - s) + mapped * s
    rgb = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return rgb.astype(np.float32) / 255.0


def circular_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """内窥镜圆形视场渐晕 (中心亮、边缘暗)。img: float [0,1]。"""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xs - (w - 1) * 0.5) / max(w * 0.5, 1.0)) ** 2
                + ((ys - (h - 1) * 0.5) / max(h * 0.5, 1.0)) ** 2)
    vig = 1.0 - float(strength) * np.clip(r, 0.0, 1.2) ** 2
    return np.clip(img * vig[..., None], 0.0, 1.0)


def barrel_distort(img: np.ndarray, k1: float, fx: float, fy: float,
                   cx: float, cy: float) -> np.ndarray:
    """轻度径向桶形畸变 (仅 RGB)。k1<0 为桶形, 典型内窥镜。"""
    if abs(k1) < 1e-8:
        return img
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    x = (xs - cx) / max(fx, 1e-6)
    y = (ys - cy) / max(fy, 1e-6)
    r2 = x * x + y * y
    scale = 1.0 + float(k1) * r2
    map_x = (x * scale) * fx + cx
    map_y = (y * scale) * fy + cy
    return cv2.remap(img.astype(np.float32), map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def apply_optics(img: np.ndarray, cam, vignette: float = 0.0,
                 k1: float = 0.0) -> np.ndarray:
    """渐晕 + 桶形 (先畸变再渐晕, 符合透镜-光阑顺序)。"""
    out = barrel_distort(img, k1, cam.fx, cam.fy, cam.cx, cam.cy)
    return circular_vignette(out, vignette)
