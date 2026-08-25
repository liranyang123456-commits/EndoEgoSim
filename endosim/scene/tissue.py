"""程序化组织腔道（管腔隧道）网格生成。

模拟结肠镜/胃镜视野:
- 中心线: 缓变 3D 曲线（Catmull-Rom 采样）
- 半径: 基础半径 + 皱襞（轴向半周期隆起，模拟结肠袋）+ 周向低频起伏 + fBm 细节
- 远端盲端封口（逼真"顶到组织壁"场景），近端开口
- UV: u = 周向 [0,1), v = 轴向（按弧长/纹理尺度）
- 绕序保证内表面法线朝向管腔轴心（相机所在侧）

单位: mm。
"""
from __future__ import annotations

import numpy as np

from ..render.rasterizer import SimMesh


def _catmull_rom(points: np.ndarray, n_samples: int) -> np.ndarray:
    P = np.asarray(points, dtype=np.float64)
    K = len(P)
    if K < 2:
        raise ValueError("中心线至少需要2个控制点")
    if K == 2:
        t = np.linspace(0, 1, n_samples)[:, None]
        return P[0][None] * (1 - t) + P[1][None] * t
    E = np.concatenate([(2 * P[0] - P[1])[None], P, (2 * P[-1] - P[-2])[None]], axis=0)
    segs = K - 1
    out = []
    for i in range(segs):
        n_loc = n_samples if i == segs - 1 else max(int(round(n_samples / segs)), 1)
        for j in range(n_loc):
            u = j / max(n_loc - 1, 1) if n_loc > 1 else 0.0
            p0, p1, p2, p3 = E[i], E[i + 1], E[i + 2], E[i + 3]
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    return np.asarray(out)


def _parallel_transport_frames(cl: np.ndarray):
    """沿中心线的旋转最小标架（tangent/right/up, 每点一组）。

    固定参考轴方案在 |dot(tangent, ref)| 跨阈值时会硬切换参考轴,
    导致标架瞬间翻转 ~90-180°（相机轨迹出现不可能的大转角）。
    平行传输（投影法）保证标架沿弧长连续变化。
    """
    n = len(cl)
    tangents = np.zeros_like(cl)
    for i in range(n):
        i0, i1 = max(i - 1, 0), min(i + 1, n - 1)
        t = cl[i1] - cl[i0]
        tangents[i] = t / np.linalg.norm(t)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(tangents[0] @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    right = np.zeros_like(cl)
    up = np.zeros_like(cl)
    right[0] = np.cross(ref, tangents[0])
    right[0] /= np.linalg.norm(right[0])
    up[0] = np.cross(tangents[0], right[0])
    for i in range(1, n):
        r = right[i - 1] - (right[i - 1] @ tangents[i]) * tangents[i]
        nr = np.linalg.norm(r)
        if nr < 1e-9:  # 防御: 切向与 right 平行时改用 up 叉乘
            r = np.cross(tangents[i], up[i - 1])
            nr = np.linalg.norm(r)
        right[i] = r / nr
        up[i] = np.cross(tangents[i], right[i])
    return tangents, right, up


def _fbm_1d(n: int, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """1D 带限噪声（多倍频正弦叠加）。"""
    x = np.linspace(0, 1, n)
    out = np.zeros(n)
    for o in range(octaves):
        freq = 2 ** (o + 1)
        amp = 0.5 ** o
        phase = rng.uniform(0, 2 * np.pi)
        out += amp * np.sin(2 * np.pi * freq * x + phase)
    out /= sum(0.5 ** o for o in range(octaves))
    return out


def _fbm_2d(h: int, w: int, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """2D 带限噪声（用于半径扰动场），行列均周期化以保持周向无缝。"""
    ys = np.linspace(0, 1, h)
    xs = np.linspace(0, 1, w)
    out = np.zeros((h, w))
    for o in range(octaves):
        fy, fx = 2 ** (o + 1), 2 ** (o + 1)
        amp = 0.5 ** o
        py = rng.uniform(0, 2 * np.pi)
        px_ = rng.uniform(0, 2 * np.pi)
        out += amp * np.sin(2 * np.pi * fy * ys[:, None] + py) * np.cos(2 * np.pi * fx * xs[None] + px_)
    out /= sum(0.5 ** o for o in range(octaves))
    return out


class TissueTunnel:
    """组织腔道（世界系）。Z 轴为插入主方向，近端 z=0，远端 z≈length。"""

    def __init__(self, rng: np.random.Generator,
                 length: float = 500.0,
                 radius_base: float = 28.0,
                 radius_range: float = 12.0,
                 curvature: float = 0.25,
                 n_rings: int = 160,
                 n_sector: int = 64,
                 fold_amplitude: float = 0.22,
                 fold_count: int = 6,
                 detail_amplitude: float = 0.08,
                 cap_end: bool = True,
                 seed: int | None = None):
        self.rng = rng
        self.length = float(length)
        self.radius_base = float(radius_base)
        self.cap_end = bool(cap_end)

        # ---- 中心线: 从 (0,0,0) 到 (dx, dy, length) 缓弯 ----
        ctrl = np.array([
            [0.0, 0.0, 0.0],
            [rng.uniform(-curvature, curvature) * length * 0.25,
             rng.uniform(-curvature, curvature) * length * 0.25, length * 0.33],
            [rng.uniform(-curvature, curvature) * length * 0.5,
             rng.uniform(-curvature, curvature) * length * 0.5, length * 0.66],
            [rng.uniform(-curvature, curvature) * length * 0.6,
             rng.uniform(-curvature, curvature) * length * 0.6, length],
        ])
        self.centerline = _catmull_rom(ctrl, n_rings)
        self._tangents, self._pt_right, self._pt_up = _parallel_transport_frames(self.centerline)

        # ---- 半径场 ----
        r = self.radius_base + rng.uniform(0, radius_range) * _fbm_1d(n_rings, rng, octaves=3)
        # 皱襞: 轴向半波隆起（结肠袋状褶皱）
        fold_phase = rng.uniform(0, 2 * np.pi)
        r *= (1.0 + fold_amplitude * np.abs(np.sin(np.pi * fold_count * np.linspace(0, 1, n_rings) + fold_phase)))
        # 细节噪声（周向+轴向 2D）
        detail = detail_amplitude * _fbm_2d(n_rings, n_sector, rng)
        self.radius_axial = r
        self.radius_detail = detail

        self.n_rings, self.n_sector = n_rings, n_sector
        self._build_mesh()

    # ------------------------------------------------------------------
    def _frame_at(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """中心线第 i 环处的切向/法向坐标系（平行传输, 沿弧长连续）。"""
        return self._tangents[i], self._pt_right[i], self._pt_up[i]

    def _build_mesh(self) -> None:
        n_r, n_s = self.n_rings, self.n_sector
        theta = np.linspace(0, 2 * np.pi, n_s, endpoint=False)
        verts = np.zeros((n_r * n_s, 3))
        uvs = np.zeros((n_r * n_s, 2))

        # 累计弧长（v 坐标）
        seg = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
        arc = np.concatenate([[0], np.cumsum(seg)])
        self.arc = arc

        for i in range(n_r):
            c = self.centerline[i]
            _, right, up = self._frame_at(i)
            rad = self.radius_axial[i] * (1.0 + self.radius_detail[i])
            ring = (c[None] + rad[:, None] * (np.cos(theta)[:, None] * right[None]
                                             + np.sin(theta)[:, None] * up[None]))
            verts[i * n_s:(i + 1) * n_s] = ring
            uvs[i * n_s:(i + 1) * n_s, 0] = theta / (2 * np.pi)
            uvs[i * n_s:(i + 1) * n_s, 1] = arc[i] / 120.0  # 纹理每120mm重复

        faces = []
        for i in range(n_r - 1):
            for j in range(n_s):
                jn = (j + 1) % n_s
                a, b = i * n_s + j, i * n_s + jn
                c_, d = (i + 1) * n_s + j, (i + 1) * n_s + jn
                # 绕序: 从管内看逆时针 -> 法线朝轴心（见 orient 检查）
                faces.append([a, b, c_])
                faces.append([b, d, c_])
        faces = np.asarray(faces, dtype=np.int64)

        # ---- 远端盲端封口（半球帽） ----
        if self.cap_end:
            cap_center_idx = len(verts)
            verts = np.vstack([verts, self.centerline[-1][None]])
            end_ring = verts[(n_r - 1) * n_s: n_r * n_s]
            apex_idx = len(verts)
            apex = self.centerline[-1] + self._frame_at(n_r - 1)[0] * max(self.radius_axial[-1] * 0.7, 5.0)
            verts = np.vstack([verts, apex[None]])
            for j in range(n_s):
                jn = (j + 1) % n_s
                faces = np.vstack([faces, [[(n_r - 1) * n_s + j, (n_r - 1) * n_s + jn, apex_idx]]])
            self.cap_center_idx = cap_center_idx
            # 盲端中心的 UV
            uvs = np.vstack([uvs, [[0.5, arc[-1] / 120.0], [0.5, (arc[-1] + 20.0) / 120.0]]])

        self.mesh = SimMesh(verts, faces, uvs=uvs)
        self.mesh.compute_normals()
        self._orient_inward()

    def _orient_inward(self) -> None:
        """确保内表面法线朝向管腔轴心（光栅化背面剔除依赖）。"""
        tri = self.mesh.verts[self.mesh.faces]
        centroid = tri.mean(axis=1)
        # 每个三角形找最近中心线点
        cl = self.centerline
        cl_idx = np.argmin(
            np.linalg.norm(centroid[:, None, :] - cl[::max(len(cl) // 20, 1)][None], axis=2),
            axis=1) * max(len(cl) // 20, 1)
        radial = cl[cl_idx] - centroid
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        inward_frac = np.mean(np.einsum('ij,ij->i', fn, radial) > 0)
        if inward_frac < 0.5:
            self.mesh.flip_winding()

    # ------------------------------------------------------------------
    def sample_axis_point(self, s: float) -> np.ndarray:
        """弧长 s (mm) 处的中心线点。"""
        arc = self.arc
        s = np.clip(s, arc[0], arc[-1])
        i = np.searchsorted(arc, s) - 1
        i = int(np.clip(i, 0, len(arc) - 2))
        t = (s - arc[i]) / max(arc[i + 1] - arc[i], 1e-9)
        return self.centerline[i] * (1 - t) + self.centerline[i + 1] * t

    def axis_pose_at(self, s: float, roll: float = 0.0) -> np.ndarray:
        """弧长 s 处"沿轴向看"的位姿（c2w），供轨迹生成使用。

        朝向取平行传输标架（沿弧长连续, 无参考轴切换跳变）。
        """
        from ..geometry.se3 import rot_trans, so3_exp
        i = int(np.clip(np.searchsorted(self.arc, s) - 1, 0, len(self.centerline) - 2))
        tangent = self._tangents[i]
        R = np.stack([self._pt_right[i], self._pt_up[i], tangent], axis=1)
        if roll != 0.0:
            R = R @ so3_exp(tangent * np.deg2rad(roll))
        return rot_trans(R, self.sample_axis_point(s))

    def to_dict(self) -> dict:
        return {"length": self.length, "radius_base": self.radius_base,
                "n_rings": self.n_rings, "n_sector": self.n_sector,
                "cap_end": self.cap_end, "arc_length": float(self.arc[-1])}
