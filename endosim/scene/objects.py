"""仿真目标物体: BOP 网格加载 + 程序化手术器械。

所有物体统一为 SimMesh + Material；BOP 物体用顶点色，
程序化器械用材质着色（金属高光）。
单位: mm。
"""
from __future__ import annotations

import os

import numpy as np

from ..render.rasterizer import Material, SimMesh

BOP_MODELS_DIR = r"D:\bop\lm_models\models"  # 本地数据, 只读引用不拷贝


# ---------------------------------------------------------------------------
# BOP LineMod 网格
# ---------------------------------------------------------------------------

def load_bop_object(obj_id: int, scale: float = 1.0) -> tuple[SimMesh, dict]:
    """加载 BOP LineMod 物体 (obj_id 1..15)。

    PLY 为 ASCII 顶点色格式（毫米单位），居中到几何中心。
    返回 (mesh, info)。
    """
    path = os.path.join(BOP_MODELS_DIR, f"obj_{obj_id:06d}.ply")
    if not os.path.exists(path):
        raise FileNotFoundError(f"BOP 模型不存在: {path}")
    verts, colors, faces = [], [], []
    with open(path, "r") as f:
        n_v = n_f = 0
        in_v = in_f = False
        for line in f:
            if line.startswith("element vertex"):
                n_v = int(line.split()[-1])
            elif line.startswith("element face"):
                n_f = int(line.split()[-1])
            elif line.startswith("end_header"):
                in_v = True
                continue
            if in_v and not in_f:
                parts = line.split()
                if len(parts) == 10:
                    verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    colors.append([int(parts[6]) / 255.0, int(parts[7]) / 255.0, int(parts[8]) / 255.0])
                    if len(verts) == n_v:
                        in_v = False
                        in_f = True
            elif in_f:
                parts = line.split()
                if len(parts) == 4 and parts[0] == "3":
                    faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
                    if len(faces) == n_f:
                        break
    verts = np.asarray(verts, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    verts = (verts - verts.mean(axis=0)) * float(scale)  # 居中+缩放
    info = {"source": "bop_linemod", "obj_id": obj_id, "scale": scale,
            "n_verts": len(verts), "n_faces": len(faces)}
    return SimMesh(verts, faces, colors=colors), info


# ---------------------------------------------------------------------------
# 程序化手术器械
# ---------------------------------------------------------------------------

def _cylinder(p0: np.ndarray, dir_: np.ndarray, radius: float, length: float,
              n_seg: int = 16, n_ring: int = 2, caps: bool = True):
    """沿 dir_ 的圆柱（含端盖），返回 (verts, faces)。"""
    d = np.asarray(dir_, float)
    d /= np.linalg.norm(d)
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(ref, d); u /= np.linalg.norm(u)
    w = np.cross(d, u)
    ang = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    verts, faces = [], []
    for k in range(n_ring + 1):
        t = k / n_ring
        c = p0 + d * (length * t)
        r = radius * (1.0 if k in (0, n_ring) else 1.0)
        for a in ang:
            verts.append(c + r * (np.cos(a) * u + np.sin(a) * w))
    for k in range(n_ring):
        for j in range(n_seg):
            jn = (j + 1) % n_seg
            a, b = k * n_seg + j, k * n_seg + jn
            c_, dd = (k + 1) * n_seg + j, (k + 1) * n_seg + jn
            faces.append([a, c_, b]); faces.append([b, c_, dd])
    if caps:
        ci0 = len(verts); verts.append(p0.copy())
        ci1 = len(verts); verts.append(p0 + d * length)
        for j in range(n_seg):
            jn = (j + 1) % n_seg
            faces.append([ci0, j, jn])
            faces.append([ci1, n_ring * n_seg + jn, n_ring * n_seg + j])
    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


def _box(center: np.ndarray, half: np.ndarray, rot_axis: np.ndarray, angle_deg: float):
    c = np.asarray(center, float)
    h = np.asarray(half, float)
    from ..geometry.se3 import so3_exp
    R = so3_exp(np.asarray(rot_axis, float) * np.deg2rad(angle_deg))
    corners = np.array([[sx, sy, sz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)], float) * h
    corners = corners @ R.T + c
    faces = np.array([
        [0, 1, 3], [0, 3, 2],   # z-
        [4, 7, 5], [4, 6, 7],   # z+
        [0, 4, 5], [0, 5, 1],   # y-
        [2, 3, 7], [2, 7, 6],   # y+
        [0, 2, 6], [0, 6, 4],   # x-
        [1, 5, 7], [1, 7, 3],   # x+
    ], dtype=np.int64)
    return corners, faces


def make_instrument(rng: np.random.Generator | None = None,
                    shaft_radius: float = 3.0,
                    shaft_length: float = 160.0,
                    jaw_length: float = 18.0,
                    jaw_open: float = 25.0) -> tuple[SimMesh, dict]:
    """程序化腹腔镜抓钳: 圆柱杆 + 腕部 + 两片锥形夹爪。

    器械局部坐标系: 杆沿 -Z（尖端在 +Z=0 处, 杆身伸向 -Z）。
    即尖端 (tip) 在原点附近, 柄在 (0,0,-shaft_length)。
    """
    rng = rng or np.random.default_rng()
    all_v, all_f = [], []

    def add(v, f):
        if len(all_v):
            f = f + len(all_v)
        all_v.append(v)
        all_f.append(f)

    # 杆: 从 z=0 到 z=-shaft_length
    v, f = _cylinder(np.array([0, 0, -jaw_length * 0.2]), np.array([0, 0, -1.0]),
                     shaft_radius, shaft_length)
    add(v, f)
    # 腕部球
    n_sph = 12
    ang = np.linspace(0, 2 * np.pi, 2 * n_sph, endpoint=False)
    lat = np.linspace(0, np.pi, n_sph)
    vs = []
    for la in lat:
        for ao in ang[:n_sph * 2 // 2]:
            vs.append([np.sin(la) * np.cos(ao), np.sin(la) * np.sin(ao), np.cos(la)])
    vs = np.asarray(vs) * (shaft_radius * 1.35)
    fs = []
    n_azi = n_sph
    for i in range(n_sph - 1):
        for j in range(n_azi):
            jn = (j + 1) % n_azi
            a, b = i * n_azi + j, i * n_azi + jn
            c_, d = (i + 1) * n_azi + j, (i + 1) * n_azi + jn
            fs.append([a, b, c_]); fs.append([b, d, c_])
    add(vs + np.array([0, 0, -jaw_length * 0.2]), np.asarray(fs, dtype=np.int64))

    # 两片夹爪（细长盒子, 张开角度对称）
    for sign in (+1, -1):
        pivot = np.array([0, sign * shaft_radius * 0.9, -jaw_length * 0.1])
        rot_axis = np.array([1.0, 0.0, 0.0])
        angle = sign * jaw_open
        c, f = _box(pivot + np.array([0, 0, jaw_length / 2 + 2]), np.array([1.6, 2.6, jaw_length / 2]),
                    rot_axis, angle)
        add(c, f)

    verts = np.vstack(all_v)
    faces = np.vstack(all_f).astype(np.int64)
    mesh = SimMesh(verts, faces)
    mesh.compute_normals()
    info = {"source": "procedural_instrument", "shaft_radius": shaft_radius,
            "shaft_length": shaft_length, "jaw_length": jaw_length,
            "jaw_open_deg": jaw_open, "n_faces": len(faces)}
    return mesh, info


# ---------------------------------------------------------------------------
# 显式参照物: 贴壁棋盘格标记（仿手术钛夹/标定片, 呼应用户棋盘格标定工作）
# ---------------------------------------------------------------------------

def make_marker_patch(size_mm: float = 14.0, n_cells: int = 6,
                      cell_px: int = 16) -> tuple[SimMesh, np.ndarray, dict]:
    """棋盘格标记贴片。

    局部系: 平面在 z=0, x/y 在 [-size/2, size/2]; +z 为外法线。
    返回 (mesh, texture, info)。纹理为黑白棋盘格。
    """
    h = size_mm / 2.0
    verts = np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]])
    # 加一点厚度变成薄盒（避免单面剔除问题）
    th = 0.4
    v = []
    for z in (0.0, th):
        for yy in (-h, h):
            for xx in (-h, h):
                v.append([xx, yy, z])
    v = np.asarray(v, float)  # 0-3: z=0底面; 4-7: z=th顶面(棋盘格面)
    faces = np.array([
        [0, 3, 2], [0, 2, 1],          # 底面 -z
        [4, 5, 6], [4, 6, 7],          # 顶面 +z (棋盘格)
        [0, 1, 5], [0, 5, 4],          # y-
        [2, 3, 7], [2, 7, 6],          # y+
        [0, 4, 7], [0, 7, 3],          # x-
        [1, 2, 6], [1, 6, 5],          # x+
    ], dtype=np.int64)
    uvs = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
    ], dtype=np.float64)
    # 棋盘格纹理
    tex = np.zeros((cell_px, cell_px * n_cells, 3), dtype=np.uint8)
    for i in range(n_cells):
        for j in range(n_cells):
            c = 235 if (i + j) % 2 == 0 else 25
            tex[j * cell_px:(j + 1) * cell_px, i * cell_px:(i + 1) * cell_px] = c
    tex = np.ascontiguousarray(tex)
    mesh = SimMesh(v, faces, uvs=uvs)
    mesh.compute_normals()
    # 保证棋盘面(+z面)朝外: 检查face 4-7 的法线
    tri = mesh.verts[mesh.faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    info = {"source": "checkerboard_marker", "size_mm": size_mm,
            "n_cells": n_cells, "n_faces": len(faces)}
    return mesh, tex, info


MARKER_MATERIAL = Material(albedo=(1.0, 1.0, 1.0), ambient=0.12, ks=0.4,
                           shininess=60.0, d_ref=50.0, atten_max=6.0)


# ---------------------------------------------------------------------------
# 材质库
# ---------------------------------------------------------------------------

TISSUE_MATERIAL = Material(albedo=(0.92, 0.72, 0.66), ambient=0.08, ks=0.22,
                           shininess=36.0, d_ref=50.0, atten_max=6.0)

INSTRUMENT_MATERIAL = Material(albedo=(0.62, 0.63, 0.66), ambient=0.10, ks=0.85,
                               shininess=90.0, spec_color=(1.0, 0.98, 0.94),
                               d_ref=50.0, atten_max=8.0)

BOP_MATERIAL = Material(albedo=(1.0, 1.0, 1.0), ambient=0.10, ks=0.35,
                        shininess=60.0, d_ref=50.0, atten_max=6.0)


def available_bop_ids() -> list[int]:
    ids = []
    for i in range(1, 16):
        if os.path.exists(os.path.join(BOP_MODELS_DIR, f"obj_{i:06d}.ply")):
            ids.append(i)
    return ids
