"""SE(3) 刚体变换工具。

约定（全项目统一）:
- 位姿 T = [R | t; 0 0 0 1]，4x4 齐次矩阵，float64。
- T_wc: camera-to-world（相机在世界系下的位姿）。
- 相机系: OpenCV 惯例（+X 右, +Y 下, +Z 前）。
- 单位: 平移 mm，角度 rad（对外接口用度时单独注明）。
"""
from __future__ import annotations

import numpy as np


def hat(v: np.ndarray) -> np.ndarray:
    """3向量 -> 反对称矩阵。"""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def vee(m: np.ndarray) -> np.ndarray:
    """反对称矩阵 -> 3向量。"""
    return np.array([m[2, 1], m[0, 2], m[1, 0]], dtype=np.float64)


def so3_exp(w: np.ndarray) -> np.ndarray:
    """SO(3) 指数映射（Rodrigues）。w: 旋转向量 (3,)。"""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = hat(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) 对数映射 -> 旋转向量。"""
    R = np.asarray(R, dtype=np.float64)
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_t))
    if theta < 1e-10:
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-6:
        # 接近 pi，从对称部分恢复轴
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, 1.0))
        k = axis / max(np.linalg.norm(axis), 1e-12)
        return k * theta
    return theta * vee(R - R.T) / (2.0 * np.sin(theta))


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """se(3) 指数映射。xi = [w(3), v(3)]。"""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    w, v = xi[:3], xi[3:]
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        T = np.eye(4)
        T[:3, 3] = v
        return T
    k = w / theta
    K = hat(k)
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    # V = I + ((1-cos t)/t) K + ((t-sin t)/t^2) K^2
    V = (np.eye(3) + (1.0 - np.cos(theta)) / theta * K
         + (theta - np.sin(theta)) / theta ** 2 * (K @ K))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """se(3) 对数映射 -> xi = [w(3), v(3)]。"""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    w = so3_log(R)
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        return np.concatenate([np.zeros(3), t])
    k = w / theta
    K = hat(k)
    # V = I + ((1-cos t)/t) K + ((t-sin t)/t^2) K^2  (单位轴)
    V = (np.eye(3) + (1.0 - np.cos(theta)) / theta * K
         + (theta - np.sin(theta)) / theta ** 2 * (K @ K))
    v = np.linalg.inv(V) @ t
    return np.concatenate([w, v])


def inverse(T: np.ndarray) -> np.ndarray:
    """齐次位姿求逆（解析，比 np.linalg.inv 快且稳）。"""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def relative(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """T_ab = T_a^{-1} @ T_b：b 在 a 坐标系下的位姿。"""
    return inverse(T_a) @ T_b


def compose(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    return np.asarray(T_a) @ np.asarray(T_b)


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """变换点集 (N,3)。"""
    pts = np.asarray(pts, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    return pts @ R.T + t


def rot_trans(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """构造 c2w：相机位于 eye，光轴指向 target（OpenCV 相机系 +Z 朝向 target）。"""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    z = target - eye
    n = np.linalg.norm(z)
    if n < 1e-9:
        return np.eye(4)
    z = z / n
    x = np.cross(up, z)
    nx = np.linalg.norm(x)
    if nx < 1e-9:  # up 与 z 平行，换一个
        up = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        nx = np.linalg.norm(x)
    x = x / nx
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)  # 列向量为相机轴在世界系的表示
    return rot_trans(R, eye)


def slerp(R0: np.ndarray, R1: np.ndarray, u: float) -> np.ndarray:
    """旋转插值。"""
    R_rel = R0.T @ R1
    w = so3_log(R_rel)
    return R0 @ so3_exp(w * u)


def interp_se3(T0: np.ndarray, T1: np.ndarray, u: float) -> np.ndarray:
    """SE(3) 插值（旋转 slerp + 平移线性；对密插值足够平滑）。"""
    R = slerp(T0[:3, :3], T1[:3, :3], u)
    t = (1.0 - u) * T0[:3, 3] + u * T1[:3, 3]
    return rot_trans(R, t)


def catmull_rom_se3(poses: list[np.ndarray], num_samples: int) -> np.ndarray:
    """在 SE(3) 路标点序列上做 Catmull-Rom 样条（旋转 slerp、平移 Catmull-Rom）。

    poses: K 个 4x4；返回恰好 (num_samples, 4, 4)，首尾路标点包含在结果中。
    """
    P = np.stack([np.asarray(p) for p in poses])
    K = len(P)
    if K == 1:
        return np.repeat(P, num_samples, axis=0)
    # 构造虚拟端点
    pads = [2 * P[0] - P[1], 2 * P[-1] - P[-2]]
    E = np.concatenate([pads[0][None], P, pads[1][None]], axis=0)  # (K+2, 4, 4)
    segs = max(K - 1, 1)
    # 每段样本数: 均分, 最后一段补余, 总和恰为 num_samples
    per = max(num_samples // segs, 1)
    counts = [per] * (segs - 1) + [max(num_samples - per * (segs - 1), 1)]
    out = []
    for i in range(segs):
        p0, p1, p2, p3 = E[i], E[i + 1], E[i + 2], E[i + 3]
        for j in range(counts[i]):
            u = j / max(counts[i] - 1, 1) if counts[i] > 1 else 0.0
            # 平移 Catmull-Rom
            t0, t1, t2, t3 = p0[:3, 3], p1[:3, 3], p2[:3, 3], p3[:3, 3]
            t = (0.5 * ((2 * t1) + (-t0 + t2) * u
                        + (2 * t0 - 5 * t1 + 4 * t2 - t3) * u ** 2
                        + (-t0 + 3 * t1 - 3 * t2 + t3) * u ** 3))
            # 旋转: 在 p1->p2 之间 slerp（样条平滑由路标密度保证）
            R = slerp(p1[:3, :3], p2[:3, :3], u)
            out.append(rot_trans(R, t))
    return np.stack(out[:num_samples])


def quat_from_R(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 四元数 [x, y, z, w]（TUM 顺序）。"""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def R_from_quat(q: np.ndarray) -> np.ndarray:
    """四元数 [x,y,z,w] -> 旋转矩阵。"""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """欧拉角（XYZ外旋，度）-> 旋转矩阵。"""
    def ax(a, axis):
        a = np.deg2rad(a)
        c, s = np.cos(a), np.sin(a)
        if axis == 0:
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == 1:
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return ax(rz, 2) @ ax(ry, 1) @ ax(rx, 0)


def orthogonalize(R: np.ndarray) -> np.ndarray:
    """把数值漂移的旋转矩阵投影回 SO(3)（SVD 正交化，保行列式为正）。"""
    U, _, Vt = np.linalg.svd(R)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    return U @ D @ Vt


def ensure_se3(T: np.ndarray) -> np.ndarray:
    T = np.array(T, dtype=np.float64, copy=True)
    T[:3, :3] = orthogonalize(T[:3, :3])
    T[3, :] = [0.0, 0.0, 0.0, 1.0]
    return T
