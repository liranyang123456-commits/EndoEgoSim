"""相机内参模型与投影工具（针孔模型，OpenCV 惯例）。"""
from __future__ import annotations

import numpy as np


class PinholeCamera:
    """针孔相机。

    fx, fy, cx, cy 单位为像素。分辨率 (width, height)。
    默认参数对齐 SCARED d1_key1（fx≈1035 @1280x1024 缩放到 640x512）。
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 width: int, height: int):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.width, self.height = int(width), int(height)
        self.K = np.array([[self.fx, 0.0, self.cx],
                           [0.0, self.fy, self.cy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)

    @classmethod
    def from_fov(cls, hfov_deg: float, width: int, height: int,
                 vfov_deg: float | None = None) -> "PinholeCamera":
        vfov = vfov_deg if vfov_deg is not None else hfov_deg * height / width
        fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        fy = (height / 2.0) / np.tan(np.deg2rad(vfov) / 2.0)
        return cls(fx, fy, width / 2.0, height / 2.0, width, height)

    @classmethod
    def scared_like(cls, width: int = 640, height: int = 512,
                    jitter: float = 0.0, rng: np.random.Generator | None = None
                    ) -> "PinholeCamera":
        """SCARED d1_key1 内参（1280x1024: fx=1035.31, fy=1035.09, cx=596.96, cy=520.41）
        等比缩放到目标分辨率，可选主点/焦距抖动（域随机化）。"""
        sx, sy = width / 1280.0, height / 1024.0
        fx, fy = 1035.3081 * sx, 1035.0876 * sy
        cx, cy = 596.9550 * sx, 520.4100 * sy
        if jitter > 0 and rng is not None:
            fx *= 1.0 + rng.normal(0, jitter)
            fy *= 1.0 + rng.normal(0, jitter)
            cx += rng.normal(0, jitter * 20)
            cy += rng.normal(0, jitter * 20)
        return cls(fx, fy, cx, cy, width, height)

    @property
    def fov_deg(self) -> tuple[float, float]:
        h = 2 * np.rad2deg(np.arctan(self.width / (2 * self.fx)))
        v = 2 * np.rad2deg(np.arctan(self.height / (2 * self.fy)))
        return float(h), float(v)

    def project(self, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """投影相机系点 (N,3) -> 像素坐标 (N,2) 与深度 (N,)。"""
        pts = np.asarray(pts_cam, dtype=np.float64)
        z = pts[:, 2]
        safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = self.fx * pts[:, 0] / safe + self.cx
        v = self.fy * pts[:, 1] / safe + self.cy
        return np.stack([u, v], axis=1), z

    def unproject(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """像素 (N,2) + 深度 (N,) -> 相机系点 (N,3)。"""
        uv = np.asarray(uv, dtype=np.float64)
        d = np.asarray(depth, dtype=np.float64)
        x = (uv[:, 0] - self.cx) / self.fx * d
        y = (uv[:, 1] - self.cy) / self.fy * d
        return np.stack([x, y, d], axis=1)

    def to_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "width": self.width, "height": self.height,
                "K": self.K.tolist(),
                "fov_hfov_deg": self.fov_deg[0], "fov_vfov_deg": self.fov_deg[1]}

    def __repr__(self) -> str:
        h, v = self.fov_deg
        return (f"PinholeCamera({self.width}x{self.height}, fx={self.fx:.1f}, "
                f"fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}, "
                f"FOV={h:.1f}°x{v:.1f}°)")
