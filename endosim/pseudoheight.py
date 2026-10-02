"""Pseudo-height 表征（Mesh-RTS 风格梯度伪高度的轻量版）。

供训练（辅助监督/辅助输入通道）与推理共用，保证两侧标签一致。
与 Mesh-RTS 完整管线（bilateral + Sobel + p68 + 形态学闭运算）的差异：
此处省略 bilateral 与闭运算，作为学习用标签足够平滑且计算更轻。
"""
from __future__ import annotations

import cv2
import numpy as np


def pseudo_height_label(img: np.ndarray) -> np.ndarray:
    """chi(|grad I| > p68) * |grad I|，clip 到 [0,1]。

    img: (H,W,3) float32 [0,1]（RGB）。返回 (H,W) float32。
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    thr = np.percentile(mag, 68)
    h = np.where(mag > thr, mag, 0.0)
    return np.clip(h, 0.0, 1.0).astype(np.float32)
