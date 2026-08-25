"""传感器级图像退化仿真（域随机化/真实感）。

- 光子散粒噪声(泊松) + 读出噪声(高斯)
- 运动模糊: 有 blur_dir 时沿图像流做方向核, 否则各向同性高斯
- 伽马编码(2.2) + 曝光抖动 + 白平衡漂移
- 轻度雾/烟（低对比度散射, 可关）

输入输出: float [0,1] RGB (H,W,3)。
"""
from __future__ import annotations

import cv2
import numpy as np


def directional_blur(img: np.ndarray, length: float, direction: np.ndarray) -> np.ndarray:
    """沿 direction=(dx,dy) 像素的线段卷积核运动模糊。"""
    k = int(np.clip(round(length), 1, 21)) | 1
    if k < 3:
        return img
    d = np.asarray(direction, dtype=np.float64).reshape(-1)[:2]
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-6:
        return cv2.GaussianBlur(img, (k, k), 0)
    d = d / nrm
    kernel = np.zeros((k, k), np.float32)
    c = k // 2
    for i in range(k):
        t = i - c
        x = int(round(c + t * d[0]))
        y = int(round(c + t * d[1]))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0
    s = float(kernel.sum())
    if s < 1.0:
        return cv2.GaussianBlur(img, (k, k), 0)
    kernel /= s
    return cv2.filter2D(img, -1, kernel)


def apply_sensor(img: np.ndarray, rng: np.random.Generator,
                 shot_noise: float = 0.5,
                 read_noise: float = 0.004,
                 blur_px: float = 0.0,
                 exposure_jitter: float = 0.15,
                 wb_jitter: float = 0.04,
                 gamma: float = 2.2,
                 haze: float = 0.0,
                 blur_dir: np.ndarray | None = None) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    H, W = img.shape[:2]

    # ---- 雾/烟: 距离相关的散射（近似: 常数低频雾 + 轻微去对比度） ----
    if haze > 0:
        fog = rng.uniform(0.5, 1.0) * haze
        img = img * (1 - fog) + fog * np.array([0.55, 0.55, 0.58])[None, None]

    # ---- 曝光与白平衡 ----
    if exposure_jitter > 0:
        gain = 1.0 + rng.uniform(-exposure_jitter, exposure_jitter)
        img = img * gain
    if wb_jitter > 0:
        gains = 1.0 + rng.normal(0, wb_jitter, 3)
        img = img * gains[None, None]

    # ---- 运动模糊 ----
    if blur_px > 0.3:
        if blur_dir is not None and float(np.linalg.norm(blur_dir)) > 0.15:
            img = directional_blur(img, blur_px, blur_dir)
        else:
            k = int(np.clip(round(blur_px), 1, 21)) | 1
            img = cv2.GaussianBlur(img, (k, k), 0)

    # ---- 光子散粒噪声: variance ∝ signal / full_well ----
    if shot_noise > 0:
        full_well = 4000.0 / max(shot_noise, 1e-3)
        sig = np.clip(img, 0, 1)
        noisy = sig + rng.normal(0, 1, sig.shape) * np.sqrt(np.maximum(sig, 0) / full_well)
        img = noisy
    # ---- 读出噪声 ----
    if read_noise > 0:
        img = img + rng.normal(0, read_noise, img.shape)

    img = np.clip(img, 0.0, 1.0)
    # ---- 伽马编码 ----
    if gamma > 0:
        img = img ** (1.0 / gamma)
    return img
