"""组织非刚性形变场（时变顶点位移）——内窥镜现实：形变是常态。

三类生理模式的随机组合:
- peristalsis: 沿管腔轴向行进的收缩环（结肠蠕动波, 行波）
- pulse:       心跳频率的局灶隆起（邻近血管搏动）
- harmonic:    全域低频谐波（呼吸 + 慢漂移）
- contact:     器械尖端附近的局部压陷（工具接触形变）

位移方向: 沿径向指向/背离管腔轴心（收缩/舒张）。
形变只作用于组织网格; 相机/器械保持刚体 -> egomotion GT 始终良定义。

模型: d(v, t) = strength * radial(v) * Σ_k a_k φ_k(v) ψ_k(t)
"""
from __future__ import annotations

import numpy as np


class TissueDeformation:
    def __init__(self, tunnel, rng: np.random.Generator,
                 strength: float = 1.0,
                 n_modes: int = 5,
                 breath_hz: float = 0.25,
                 pulse_hz: float = 1.4,
                 fps: float = 10.0,
                 mode_mix: tuple | None = None,
                 contact_point: np.ndarray | None = None):
        """strength=0 完全刚性; 1=强形变（径向位移可达半径~20%）。

        mode_mix: (peristalsis, pulse, harmonic) 的权重随机来源;
        contact_point: 世界系接触点（器械尖端附近组织压陷）。
        """
        self.tunnel = tunnel
        self.rng = rng
        self.strength = float(strength)
        self.fps = float(fps)
        self.arc_len = float(tunnel.arc[-1])
        # 模式权重
        if mode_mix is None:
            mode_mix = rng.dirichlet([1.2, 1.0, 1.0])
        self.mode_mix = np.asarray(mode_mix, float)

        # ---- 蠕动波: 1~2 个行波（沿弧长行进的收缩环） ----
        self.peri_waves = []
        for _ in range(int(rng.integers(1, 3))):
            self.peri_waves.append({
                "speed": rng.uniform(3.0, 12.0),          # mm/s 行进速度
                "sigma": rng.uniform(15.0, 45.0),         # 波宽 (mm)
                "amp": rng.uniform(0.6, 1.0),
                "s0": rng.uniform(0.0, self.arc_len),     # 初始位置
                "k_theta": int(rng.integers(1, 3)),       # 收缩的周向模态
                "theta_phase": rng.uniform(0, 2 * np.pi),
                "dir": rng.choice([-1.0, 1.0]),
            })

        # ---- 局灶搏动: 1~2 个高斯凸起以心跳频率起伏 ----
        self.pulses = []
        for _ in range(int(rng.integers(1, 3))):
            self.pulses.append({
                "s_c": rng.uniform(0.05, 0.95) * self.arc_len,
                "sigma": rng.uniform(8.0, 25.0),
                "theta_c": rng.uniform(0, 2 * np.pi),
                "sigma_theta": rng.uniform(0.5, 1.5),
                "amp": rng.uniform(0.5, 1.0),
                "freq": rng.uniform(0.85, 1.15) * pulse_hz,
                "phase": rng.uniform(0, 2 * np.pi),
            })

        # ---- 全域谐波（呼吸+慢漂移） ----
        self.harm_modes = []
        for k in range(int(rng.integers(2, 5))):
            self.harm_modes.append({
                "amp": rng.uniform(0.3, 1.0),
                "k_theta": int(rng.integers(1, 4)),
                "k_z": int(rng.integers(1, 3)),
                "phase": rng.uniform(0, 2 * np.pi),
                "phase_z": rng.uniform(0, 2 * np.pi),
                "freq": rng.uniform(0.8, 1.2) * (breath_hz if k % 2 == 0 else pulse_hz * 0.5),
            })

        # ---- 接触压陷 ----
        self.contact_point = (np.asarray(contact_point, float)
                              if contact_point is not None else None)
        self.contact_sigma = rng.uniform(6.0, 14.0)
        self.contact_amp = rng.uniform(0.5, 1.0)
        self.contact_freq = rng.uniform(0.4, 1.2)  # 周期性按压

        self._precompute()

    # ------------------------------------------------------------------
    def _precompute(self) -> None:
        verts = self.tunnel.mesh.verts
        n_s = self.tunnel.n_sector
        n_r = self.tunnel.n_rings
        n = len(verts)
        theta = np.zeros(n)
        s_arc = np.zeros(n)
        radial = np.zeros((n, 3))
        for i in range(n_r):
            c = self.tunnel.centerline[i]
            # 平行传输标架（tissue.py 已修复参考轴硬切换, 保证 theta 沿环连续）
            _, right, up = self.tunnel._frame_at(i)
            for j in range(n_s):
                idx = i * n_s + j
                v = verts[idx] - c
                theta[idx] = np.arctan2(v @ up, v @ right)
                radial[idx] = -v / max(np.linalg.norm(v), 1e-9)  # 指向轴心
                s_arc[idx] = self.tunnel.arc[i]
        for idx in range(n_r * n_s, n):
            i = int(np.argmin(np.linalg.norm(self.tunnel.centerline - verts[idx], axis=1)))
            c = self.tunnel.centerline[i]
            v = verts[idx] - c
            theta[idx] = np.arctan2(v[1], v[0])
            radial[idx] = -v / max(np.linalg.norm(v), 1e-9)
            s_arc[idx] = self.tunnel.arc[i]
        self.theta = theta
        self.s_arc = s_arc
        self.radial = radial

        # 接触压陷的空间权重（到接触点的测地近似: 弧长差+周向差）
        if self.contact_point is not None:
            i_c = int(np.argmin(np.linalg.norm(self.tunnel.centerline - self.contact_point, axis=1)))
            s_c = self.tunnel.arc[i_c]
            th_c = np.arctan2(self.contact_point[1] - self.tunnel.centerline[i_c][1],
                              self.contact_point[0] - self.tunnel.centerline[i_c][0])
            dth = np.angle(np.exp(1j * (self.theta - th_c)))
            ds = self.s_arc - s_c
            self.contact_weight = np.exp(-(ds ** 2 + (dth * 25.0) ** 2)
                                         / (2 * self.contact_sigma ** 2))
        else:
            self.contact_weight = None

    # ------------------------------------------------------------------
    def _spatial_temporal_amp(self, t: float) -> np.ndarray:
        n = len(self.theta)
        amp = np.zeros(n)
        w_peri, w_pulse, w_harm = self.mode_mix

        # 蠕动行波: 高斯收缩环沿弧长行进
        if w_peri > 0:
            for w in self.peri_waves:
                s_wave = (w["s0"] + w["dir"] * w["speed"] * t) % (1.3 * self.arc_len)
                ds = self.s_arc - s_wave
                ring = np.exp(-ds ** 2 / (2 * w["sigma"] ** 2))
                ring -= 0.35 * np.exp(-(ds - 2.4 * w["sigma"]) ** 2 / (2 * w["sigma"] ** 2))
                ring -= 0.35 * np.exp(-(ds + 2.4 * w["sigma"]) ** 2 / (2 * w["sigma"] ** 2))
                angular = 0.6 + 0.4 * np.cos(w["k_theta"] * self.theta + w["theta_phase"])
                amp += w_peri * w["amp"] * ring * angular

        # 局灶搏动
        if w_pulse > 0:
            for p in self.pulses:
                ds = self.s_arc - p["s_c"]
                dth = np.angle(np.exp(1j * (self.theta - p["theta_c"])))
                bump = np.exp(-ds ** 2 / (2 * p["sigma"] ** 2)) * \
                       np.exp(-dth ** 2 / (2 * p["sigma_theta"] ** 2))
                beat = 0.5 - 0.5 * np.cos(2 * np.pi * p["freq"] * t + p["phase"])
                amp += w_pulse * p["amp"] * bump * beat

        # 全域谐波
        if w_harm > 0:
            for m in self.harm_modes:
                spatial = np.sin(m["k_theta"] * self.theta + m["phase"]) * \
                          np.cos(2 * np.pi * m["k_z"] * self.s_arc / max(self.arc_len, 1)
                                 + m["phase_z"])
                temporal = np.sin(2 * np.pi * m["freq"] * t + m["phase"])
                amp += w_harm * m["amp"] * spatial * temporal

        # 接触压陷（周期性按压, 负向=压陷）
        if self.contact_weight is not None:
            press = max(0.0, np.sin(2 * np.pi * self.contact_freq * t))
            amp -= self.contact_amp * press * self.contact_weight

        return amp

    def displacement(self, frame: int) -> np.ndarray:
        if self.strength <= 0:
            return np.zeros((len(self.tunnel.mesh.verts), 3))
        t = frame / self.fps
        amp = self._spatial_temporal_amp(t)
        scale = self.strength * 0.14 * self.tunnel.radius_base
        return scale * amp[:, None] * self.radial

    def apply(self, frame: int) -> np.ndarray:
        return self.tunnel.mesh.verts + self.displacement(frame)

    def to_dict(self) -> dict:
        return {
            "strength": self.strength, "fps": self.fps,
            "mode_mix": self.mode_mix.tolist(),
            "n_peristalsis_waves": len(self.peri_waves),
            "peristalsis_speeds": [w["speed"] for w in self.peri_waves],
            "n_pulses": len(self.pulses),
            "pulse_freqs": [p["freq"] for p in self.pulses],
            "n_harmonic_modes": len(self.harm_modes),
            "contact": self.contact_point is not None,
        }


class MeshHarmonicDeformation:
    """通用网格谐波形变（适用于任意器官几何, 如 C3VD 真实器官网格）。

    d(v, t) = strength · scale · Σ_k a_k sin(f_k·v + φ_k) · ψ_k(t)
    3D行波+驻波混合: 模拟蠕动/搏动在任意解剖形状上的传播。
    """

    def __init__(self, mesh_verts: np.ndarray, rng: np.random.Generator,
                 strength: float = 1.0,
                 fps: float = 10.0,
                 n_modes: int = 6,
                 ref_radius: float = 30.0):
        self.strength = float(strength)
        self.fps = float(fps)
        self.ref_radius = float(ref_radius)
        self.n_verts = len(mesh_verts)
        lo, hi = mesh_verts.min(0), mesh_verts.max(0)
        self.extent = np.maximum(hi - lo, 1.0)
        self.modes = []
        for k in range(n_modes):
            # 空间频率: 每模式1~3个周期跨器官
            freq = rng.uniform(0.8, 3.0) * 2 * np.pi / self.extent
            self.modes.append({
                "freq": freq,
                "phase": rng.uniform(0, 2 * np.pi, 3),
                "amp": rng.uniform(0.3, 1.0) / n_modes,
                "t_freq": rng.uniform(0.2, 1.6),        # Hz: 慢(蠕动)到快(搏动)
                "t_phase": rng.uniform(0, 2 * np.pi),
                "wave_dir": rng.uniform(-1, 1, 3),       # 行波方向
            })

    def displacement(self, frame: int) -> np.ndarray:
        if self.strength <= 0:
            return np.zeros((self.n_verts, 3))
        t = frame / self.fps
        v = self.verts_ref
        amp = np.zeros(len(v))
        for m in self.modes:
            spatial = np.sin(v * m["freq"][None, :] + m["phase"][None, :]).prod(axis=1)
            proj = v @ (m["wave_dir"] / max(np.linalg.norm(m["wave_dir"]), 1e-9))
            wave = np.sin(proj * np.linalg.norm(m["freq"]) + 2 * np.pi * m["t_freq"] * t
                          + m["t_phase"])
            temporal = 0.5 * (np.sin(2 * np.pi * m["t_freq"] * t + m["t_phase"]) + wave)
            amp += m["amp"] * spatial * temporal
        # 方向: 沿表面法线(膨胀/收缩), 与组织生理运动一致
        return self.strength * 0.08 * self.ref_radius * amp[:, None] * self.normals_ref

    def apply(self, frame: int) -> np.ndarray:
        return self.verts_ref + self.displacement(frame)

    def set_reference(self, verts: np.ndarray, normals: np.ndarray | None = None) -> None:
        self.verts_ref = np.asarray(verts, dtype=np.float64)
        if normals is not None:
            self.normals_ref = np.asarray(normals, dtype=np.float64)
        elif getattr(self, "normals_ref", None) is None:
            self.normals_ref = np.zeros_like(self.verts_ref)

    def to_dict(self) -> dict:
        return {"kind": "mesh_harmonic", "strength": self.strength,
                "fps": self.fps, "n_modes": len(self.modes),
                "t_freqs": [m["t_freq"] for m in self.modes]}
