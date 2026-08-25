"""相机轨迹生成器。

基础运动:
1. insertion/retraction: 沿腔道轴前进-后退（梯形速度）
2. orbital scan: 原地旋转扫查（俯仰/偏航摆动）
3. free 6D: SE(3) Catmull-Rom 样条
4. retrace: 沿同一路径进镜再退回（回环/重访）
5. keyframe: 从密采样进镜轨迹贪心抽大基线关键帧（对齐 SCARED 稀疏关键帧 25–160mm）
6. tremor: 高频小幅扰动（可开关, 消融用）

输出: (N,4,4) c2w 位姿序列。
所有物理量单位 mm / 度。
"""
from __future__ import annotations

import numpy as np

from .se3 import (catmull_rom_se3, interp_se3, look_at, relative, rot_trans,
                  so3_exp, so3_log, slerp)


def _bandlimited_noise(n: int, rng: np.random.Generator, freq_cutoff: float = 0.2) -> np.ndarray:
    """[0,1) 采样序列的低通随机噪声（FFT 滤波）。"""
    x = rng.normal(0, 1, n)
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n)
    fft[freqs > freq_cutoff] = 0
    out = np.fft.irfft(fft, n)
    s = out.std()
    return out / s if s > 1e-9 else out


def _trapezoid(n: int, forward_frac: float = 0.7) -> np.ndarray:
    """梯形速度剖面归一化位移曲线（先前进后小幅回退）。"""
    t = np.linspace(0, 1, n)
    fwd = np.clip(t / forward_frac, 0, 1)
    # 平滑梯形: s形积分
    v = np.clip(np.sin(np.pi * np.clip(t / forward_frac, 0, 1)) ** 0.7, 0, None)
    v[t > forward_frac] = 0
    s_fwd = np.cumsum(v)
    if s_fwd[-1] <= 0:
        s_fwd = np.zeros(n)
    else:
        s_fwd = s_fwd / s_fwd[-1]
    # 回退段
    back_t = np.clip((t - forward_frac) / max(1 - forward_frac, 1e-6), 0, 1)
    back = 0.5 - 0.5 * np.cos(np.pi * back_t)
    return s_fwd - 0.25 * back * (t > forward_frac)


def generate_trajectory(rng: np.random.Generator, tunnel=None,
                        n_frames: int = 60,
                        motion_type: str = "insertion",
                        step_mm: float = 2.0,
                        rot_deg: float = 2.0,
                        tremor_mm: float = 0.0,
                        tremor_deg: float = 0.0,
                        start_arc: float | None = None,
                        travel_mm: float | None = None,
                        max_rot_per_frame: float = 15.0,
                        normalize_first: bool = True,
                        keyframe_hop_mm: tuple | None = None) -> np.ndarray:
    """生成一轨迹（含物理合理性保险: 帧间旋转超过上限自动重采样）。

    normalize_first=False 时返回场景坐标系下的原始位姿(渲染用);
    True 时归一化到首帧相机系(导出GT用)。
    """
    def _gen():
        if motion_type == "insertion":
            return _traj_insertion(rng, tunnel, n_frames, step_mm, rot_deg,
                                   tremor_mm, tremor_deg, start_arc, travel_mm)
        if motion_type == "orbital":
            return _traj_orbital(rng, tunnel, n_frames, step_mm, rot_deg,
                                 tremor_mm, tremor_deg, start_arc)
        if motion_type == "free":
            return _traj_free(rng, n_frames, step_mm, rot_deg, tremor_mm, tremor_deg)
        if motion_type == "retrace":
            return _traj_retrace(rng, tunnel, n_frames, step_mm, rot_deg,
                                 tremor_mm, tremor_deg, start_arc, travel_mm)
        if motion_type == "keyframe":
            return _traj_keyframe(rng, tunnel, n_frames, step_mm, rot_deg,
                                  tremor_mm, tremor_deg, start_arc, travel_mm,
                                  keyframe_hop_mm)
        raise ValueError(f"未知运动类型: {motion_type}")

    poses = _gen()
    for _ in range(5):
        rel = np.stack([relative(poses[i], poses[i + 1]) for i in range(len(poses) - 1)])
        rots = np.array([np.linalg.norm(so3_log(r[:3, :3])) for r in rel])
        if np.rad2deg(rots.max()) <= max_rot_per_frame:
            break
        poses = _gen()
    if normalize_first:
        poses = _normalize_first(poses)
    return poses


def _apply_tremor(poses: np.ndarray, rng: np.random.Generator,
                  tremor_mm: float, tremor_deg: float) -> np.ndarray:
    if tremor_mm <= 0 and tremor_deg <= 0:
        return poses
    n = len(poses)
    out = poses.copy()
    for ax in range(3):
        noise_t = _bandlimited_noise(n, rng, freq_cutoff=0.35) * tremor_mm
        for i in range(n):
            out[i, :3, 3] = out[i, :3, 3] + noise_t[i] * out[i, :3, ax]
    for ax in range(3):
        noise_r = _bandlimited_noise(n, rng, freq_cutoff=0.35) * np.deg2rad(tremor_deg)
        for i in range(n):
            out[i] = out[i] @ rot_trans(so3_exp(out[i, :3, ax] * noise_r[i]), np.zeros(3))
    return out


def _normalize_first(poses: np.ndarray) -> np.ndarray:
    """把世界系设为第0帧相机系。"""
    T0_inv = np.linalg.inv(poses[0])
    return np.einsum('ij,njk->nik', T0_inv, poses)


def _traj_insertion(rng, tunnel, n_frames, step_mm, rot_deg, tremor_mm,
                    tremor_deg, start_arc, travel_mm):
    travel = travel_mm if travel_mm is not None else step_mm * n_frames * 0.6
    arc_len = tunnel.arc[-1] if tunnel is not None else 500.0
    s0 = start_arc if start_arc is not None else rng.uniform(20.0, max(arc_len * 0.25, 40.0))
    s0 = min(s0, max(arc_len - travel - 10.0, 10.0))
    s_t = s0 + travel * _trapezoid(n_frames, forward_frac=rng.uniform(0.6, 0.85))
    # 视角: 轴向位姿 + 随机滚转 + 低频俯仰/偏航漂移（内窥镜师晃动, 频率固定采样一次）
    roll0 = rng.uniform(0, 360)
    wobble = rng.normal(0, 1, (3,)) * rot_deg
    f1 = rng.uniform(0.5, 1.5)
    f2 = rng.uniform(0.5, 1.5)
    poses = []
    for i in range(n_frames):
        T = tunnel.axis_pose_at(float(s_t[i]), roll=roll0 + wobble[2] * i / n_frames)
        # 附加俯仰/偏航扰动（绕相机自身轴）
        pitch = wobble[0] * np.sin(2 * np.pi * f1 * i / n_frames)
        yaw = wobble[1] * np.cos(2 * np.pi * f2 * i / n_frames)
        R_off = so3_exp(np.array([np.deg2rad(pitch), np.deg2rad(yaw), 0.0]))
        poses.append(T @ rot_trans(R_off, np.zeros(3)))
    poses = np.stack(poses)
    return _apply_tremor(poses, rng, tremor_mm, tremor_deg)


def _traj_orbital(rng, tunnel, n_frames, step_mm, rot_deg, tremor_mm,
                  tremor_deg, start_arc):
    """定点环视: 位置基本固定, 大幅俯仰/偏航扫查。"""
    arc_len = tunnel.arc[-1] if tunnel is not None else 500.0
    s0 = start_arc if start_arc is not None else rng.uniform(30.0, arc_len * 0.5)
    T_base = tunnel.axis_pose_at(float(s0))
    amp_pitch = rng.uniform(0.4, 1.0) * rot_deg * 8
    amp_yaw = rng.uniform(0.4, 1.0) * rot_deg * 8
    f1 = rng.uniform(0.5, 1.5)
    f2 = rng.uniform(0.5, 1.5)
    ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
    poses = []
    for i in range(n_frames):
        u = i / max(n_frames - 1, 1)
        pitch = amp_pitch * np.sin(2 * np.pi * f1 * u + ph1)
        yaw = amp_yaw * np.sin(2 * np.pi * f2 * u + ph2)
        R = so3_exp(np.array([np.deg2rad(pitch), np.deg2rad(yaw), 0.0]))
        poses.append(T_base @ rot_trans(R, np.zeros(3)))
    poses = np.stack(poses)
    return _apply_tremor(poses, rng, tremor_mm, tremor_deg)


def _look_along_arc(tunnel, s_t, rng, rot_deg):
    """沿弧长采样轴向位姿 + 滚转/俯仰/偏航低频晃动。"""
    roll0 = rng.uniform(0, 360)
    wobble = rng.normal(0, 1, (3,)) * rot_deg
    f1 = rng.uniform(0.5, 1.5)
    f2 = rng.uniform(0.5, 1.5)
    n = len(s_t)
    poses = []
    for i in range(n):
        T = tunnel.axis_pose_at(float(s_t[i]), roll=roll0 + wobble[2] * i / max(n, 1))
        pitch = wobble[0] * np.sin(2 * np.pi * f1 * i / max(n, 1))
        yaw = wobble[1] * np.cos(2 * np.pi * f2 * i / max(n, 1))
        R_off = so3_exp(np.array([np.deg2rad(pitch), np.deg2rad(yaw), 0.0]))
        poses.append(T @ rot_trans(R_off, np.zeros(3)))
    return np.stack(poses)


def _traj_retrace(rng, tunnel, n_frames, step_mm, rot_deg, tremor_mm,
                  tremor_deg, start_arc, travel_mm):
    """沿同一腔道路径前进再原路退回 (结肠镜回撤/重访)。"""
    n_fwd = max(n_frames // 2, 3)
    n_back = max(n_frames - n_fwd, 2)
    travel = travel_mm if travel_mm is not None else step_mm * n_fwd * 0.7
    arc_len = tunnel.arc[-1] if tunnel is not None else 500.0
    s0 = start_arc if start_arc is not None else rng.uniform(20.0, max(arc_len * 0.25, 40.0))
    s0 = min(s0, max(arc_len - travel - 10.0, 10.0))
    s_fwd = s0 + travel * np.linspace(0.0, 1.0, n_fwd)
    s_back = np.linspace(s_fwd[-1], s0, n_back)
    s_t = np.concatenate([s_fwd, s_back])[:n_frames]
    poses = _look_along_arc(tunnel, s_t, rng, rot_deg)
    return _apply_tremor(poses, rng, tremor_mm, tremor_deg)


def _traj_keyframe(rng, tunnel, n_frames, step_mm, rot_deg, tremor_mm,
                   tremor_deg, start_arc, travel_mm, keyframe_hop_mm):
    """密采样进镜轨迹上贪心抽大基线关键帧, 帧间旋转仍受 15° 约束。"""
    hop = keyframe_hop_mm or (max(step_mm * 8.0, 25.0), max(step_mm * 40.0, 160.0))
    hop_min, hop_max = float(hop[0]), float(hop[1])
    if hop_max < hop_min:
        hop_min, hop_max = hop_max, hop_min
    dense_n = max(n_frames * 10, 80)
    dense_travel = travel_mm if travel_mm is not None else hop_max * (n_frames - 1) * 0.85
    dense = _traj_insertion(rng, tunnel, dense_n, step_mm, rot_deg,
                            tremor_mm, tremor_deg, start_arc, dense_travel)
    selected = [0]
    for i in range(1, len(dense)):
        rel = relative(dense[selected[-1]], dense[i])
        trans = float(np.linalg.norm(rel[:3, 3]))
        rot = float(np.rad2deg(np.linalg.norm(so3_log(rel[:3, :3]))))
        if trans < hop_min:
            continue
        if rot > 14.5:
            continue
        if trans > hop_max and i > selected[-1] + 1:
            # 已越过 hop_max: 取前一帧 (该帧相对更近)
            cand = i - 1
            if cand not in selected:
                rel2 = relative(dense[selected[-1]], dense[cand])
                if np.rad2deg(np.linalg.norm(so3_log(rel2[:3, :3]))) <= 14.5:
                    selected.append(cand)
            continue
        selected.append(i)
        if len(selected) >= n_frames:
            break
    if selected[-1] != len(dense) - 1 and len(selected) < n_frames:
        selected.append(len(dense) - 1)
    if len(selected) < n_frames:
        extra = np.linspace(0, len(dense) - 1, n_frames).round().astype(int)
        selected = sorted(set(selected) | set(extra.tolist()))
    idx = np.array(selected[:n_frames], dtype=int)
    if len(idx) < n_frames:
        pad = np.full(n_frames - len(idx), idx[-1])
        idx = np.concatenate([idx, pad])
    return dense[idx]


def _traj_free(rng, n_frames, step_mm, rot_deg, tremor_mm, tremor_deg):
    """自由 6D: 随机路标点 SE(3) Catmull-Rom。"""
    n_wp = max(n_frames // 12, 3)
    wps = [np.eye(4)]
    p = np.zeros(3)
    R = np.eye(3)
    for _ in range(n_wp - 1):
        p = p + rng.normal(0, step_mm * 10, 3)
        R = R @ so3_exp(rng.normal(0, np.deg2rad(rot_deg * 5), 3))
        wps.append(rot_trans(R, p))
    poses = catmull_rom_se3(wps, n_frames)
    return _apply_tremor(poses, rng, tremor_mm, tremor_deg)


def motion_stats(poses: np.ndarray) -> dict:
    """轨迹统计（帧间步长/转角），供数据集质检与分层。"""
    rel = np.stack([relative(poses[i], poses[i + 1]) for i in range(len(poses) - 1)])
    trans = np.linalg.norm(rel[:, :3, 3], axis=1)
    from .se3 import so3_log
    angles = np.array([np.rad2deg(np.linalg.norm(so3_log(r[:3, :3]))) for r in rel])
    return {"step_mm_mean": float(trans.mean()), "step_mm_max": float(trans.max()),
            "rot_deg_mean": float(angles.mean()), "rot_deg_max": float(angles.max()),
            "path_length_mm": float(trans.sum())}
