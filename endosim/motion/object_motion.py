"""目标物体运动模型（器械/异物）。

三种模型:
- insertion: 从视野外沿路径进入, 到达工作区后小幅操摆
- manipulation: 绕腕部支点周期摆动（模拟夹持牵拉）
- free: 平滑随机 6D

输出: (N,4,4) T_wo（物体在世界系的位姿序列）。
物体局部系约定: 器械尖端在原点, 杆沿 -Z。
"""
from __future__ import annotations

import numpy as np

from ..geometry.se3 import catmull_rom_se3, look_at, rot_trans, so3_exp


def _smooth_noise(n: int, rng: np.random.Generator, cutoff: float = 0.25) -> np.ndarray:
    x = rng.normal(0, 1, n)
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n)
    fft[freqs > cutoff] = 0
    out = np.fft.irfft(fft, n)
    s = out.std()
    return out / s if s > 1e-9 else out


def object_motion(rng: np.random.Generator, model: str,
                  n_frames: int, scene_center: np.ndarray,
                  approach_dir: np.ndarray | None = None,
                  work_radius: float = 15.0) -> np.ndarray:
    """生成物体 T_wo 序列。

    scene_center: 工作区中心（世界系, 通常为腔道轴上一点）
    approach_dir: 进入方向（世界系单位向量, 默认从相机后方）
    """
    if model == "insertion":
        return _obj_insertion(rng, n_frames, scene_center, approach_dir, work_radius)
    if model == "manipulation":
        return _obj_manipulation(rng, n_frames, scene_center, work_radius)
    if model == "static":
        T = look_at(scene_center + np.array([0, 0, 8.0]), scene_center,
                    np.array([0.0, 1.0, 0.0]))
        return np.repeat(T[None], n_frames, axis=0)
    if model == "free":
        return _obj_free(rng, n_frames, scene_center, work_radius)
    raise ValueError(f"未知物体运动模型: {model}")


def _obj_insertion(rng, n_frames, center, approach_dir, work_radius):
    """从远处沿接近方向进入工作区, 停留后小幅操摆, 再退出。"""
    d = (np.asarray(approach_dir, float)
         if approach_dir is not None else np.array([0.2, -0.1, -1.0]))
    d /= np.linalg.norm(d)
    t = np.linspace(0, 1, n_frames)
    # 进入(0~0.35) 停留操摆(0.35~0.8) 退出(0.8~1)
    s = np.where(t < 0.35, t / 0.35,
                 np.where(t < 0.8, 1.0, 1.0 - (t - 0.8) / 0.2))
    s = s * s * (3 - 2 * s)  # smoothstep
    depth = 1.0 - s  # 1=远处, 0=工作位
    entry_offset = rng.uniform(60, 110)
    wob = _smooth_noise(n_frames, rng, 0.2)
    poses = []
    base_roll = rng.uniform(0, 2 * np.pi)
    for i in range(n_frames):
        pos = center + d * (entry_offset * depth[i]) + np.array([wob[i] * 2, wob[(i + n_frames // 2) % n_frames] * 2, 0])
        # 器械姿态: 尖端朝目标, 即 -Z_local(杆) 反向 -> 杆指向 d
        T = look_at(pos, center, np.array([0.0, 1.0, 0.0]))
        # 器械尖端在原点, 故平移取"尖端位置"= pos - d*0 (尖端即 pos)
        roll = so3_exp(np.array([0.0, 0.0, 1.0]) * (base_roll + 0.15 * np.sin(2 * np.pi * i / n_frames)))
        T = T @ rot_trans(roll, np.zeros(3))
        poses.append(T)
    return np.stack(poses)


def _obj_manipulation(rng, n_frames, center, work_radius):
    """尖端固定在工作区, 绕支点周期摆动（模拟牵拉组织）。"""
    pivot = np.asarray(center, float)
    amp_p = rng.uniform(8, 25)   # 度
    amp_y = rng.uniform(8, 25)
    f_p, f_y = rng.uniform(0.3, 0.8, 2)
    ph = rng.uniform(0, 2 * np.pi, 2)
    entry_dir = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), -1.0])
    entry_dir /= np.linalg.norm(entry_dir)
    poses = []
    for i in range(n_frames):
        u = i / max(n_frames - 1, 1)
        pitch = amp_p * np.sin(2 * np.pi * f_p * u + ph[0])
        yaw = amp_y * np.sin(2 * np.pi * f_y * u + ph[1])
        # 尖端位置固定, 姿态绕尖端摆动
        T = look_at(pivot, pivot - entry_dir, np.array([0.0, 1.0, 0.0]))
        R_w = so3_exp(np.array([np.deg2rad(pitch), np.deg2rad(yaw), 0.0]))
        poses.append(T @ rot_trans(R_w, np.zeros(3)))
    return np.stack(poses)


def _obj_free(rng, n_frames, center, work_radius):
    """自由 6D 漂移。"""
    n_wp = max(n_frames // 15, 3)
    wps = [rot_trans(np.eye(3), center.copy())]
    p = center.copy()
    R = np.eye(3)
    for _ in range(n_wp - 1):
        p = p + rng.normal(0, work_radius * 0.5, 3)
        R = R @ so3_exp(rng.normal(0, np.deg2rad(15), 3))
        wps.append(rot_trans(R, p))
    return catmull_rom_se3(wps, n_frames)
