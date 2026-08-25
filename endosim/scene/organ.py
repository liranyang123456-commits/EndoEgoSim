"""真实器官几何场景：C3VD 器官覆盖网格 / 模具 STL 作为仿真组织几何。

数据源（本地路径引用，不拷贝）:
- C3VD coverage_mesh.obj（与C3VD视频/位姿同坐标系）:
  E:\\World_Agent_Enoscopy\\datasets\\C3VD\\cecum_t1_a\\cecum_t1_a\\coverage_mesh.obj
- C3VD 器官模具 STL（结肠段真实解剖形状）:
  E:\\World_Agent_Enoscopy\\datasets\\C3VD\\{organ}_mold\\{organ}_mold\\{organ}_mold_core.stl

处理管线: 加载 -> 四边形简化(40k面) -> 统一绕序朝内 -> 主轴切片质心路径 ->
提供与 TissueTunnel 兼容的接口(mesh/centerline/arc/axis_pose_at)。
纹理: 三平面(triplanar)映射，无需UV。
"""
from __future__ import annotations

import os

import numpy as np

from ..render.rasterizer import SimMesh

C3VD_ROOT = r"E:\World_Agent_Enoscopy\datasets\C3VD"

# 可用的真实器官几何（名称 -> (网格路径, 可选C3VD位姿文件)）
ORGAN_MESHES = {
    "cecum_mold": (os.path.join(C3VD_ROOT, "cecum_mold", "cecum_mold", "cecum_mold_core.stl"), None),
    "sigmoid_mold": (os.path.join(C3VD_ROOT, "sigmoid_mold", "sigmoid_mold", "sigmoid_mold_core.stl"), None),
    "trans_mold": (os.path.join(C3VD_ROOT, "trans_mold", "trans_mold", "trans_mold_core.stl"), None),
    "desc_mold": (os.path.join(C3VD_ROOT, "desc_mold", "desc_mold", "desc_mold_core.stl"), None),
    "cecum_t1_a": (os.path.join(C3VD_ROOT, "cecum_t1_a", "cecum_t1_a", "coverage_mesh.obj"),
                   os.path.join(C3VD_ROOT, "cecum_t1_a", "cecum_t1_a", "pose.txt")),
}


def available_organs() -> list[str]:
    return [k for k, (mesh, _) in ORGAN_MESHES.items() if os.path.exists(mesh)]


def _resample_path(path: np.ndarray, n_out: int = 200,
                   max_turn_deg: float = 2.0, min_step: float = 0.5) -> np.ndarray:
    """路径去重 + 迭代平滑 + 弧长均匀重采样。

    原始路径点距不均（C3VD 轨迹含停顿/快进段, 切片质心稀疏且质心跳变）,
    直接按索引取标架会使相机在小位移内快速翻旋（帧间旋转超限）。
    停顿点簇内的方向抖动会被重采样压缩到相邻点上, 必须先按最小步长去重;
    再迭代平滑直到相邻段切向转角 <= max_turn_deg, 最后均匀重采样。
    """
    p = np.asarray(path, dtype=np.float64)
    # 1) 去重: 剔除与上一保留点距离 < min_step 的点（停顿簇）
    if len(p) > 2:
        keep = [0]
        for i in range(1, len(p) - 1):
            if np.linalg.norm(p[i] - p[keep[-1]]) >= min_step:
                keep.append(i)
        keep.append(len(p) - 1)
        p = p[keep]

    def _tangent_turns(q):
        tan = q[2:] - q[:-2]
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)
        dots = np.clip(np.einsum('ij,ij->i', tan[:-1], tan[1:]), -1, 1)
        return np.rad2deg(np.arccos(dots))

    # 2) 迭代平滑
    k = 9
    for _ in range(12):
        if len(p) > k > 2:
            pad = np.pad(p, ((k // 2, k // 2), (0, 0)), mode="edge")
            p = np.stack([np.convolve(pad[:, d], np.ones(k) / k, mode="valid")
                          for d in range(3)], 1)
        if _tangent_turns(p).max() <= max_turn_deg:
            break
    # 3) 弧长均匀重采样
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    if arc[-1] < 1e-6:
        return p
    s_new = np.linspace(0, arc[-1], n_out)
    return np.stack([np.interp(s_new, arc, p[:, d]) for d in range(3)], 1)


def _orient_inward(mesh: SimMesh, path_pts: np.ndarray) -> None:
    """统一绕序使内表面法线朝向管腔路径（相机侧）。"""
    tri = mesh.verts[mesh.faces]
    centroid = tri.mean(axis=1)
    # 每面最近路径点
    d = np.linalg.norm(centroid[:, None, :] - path_pts[None, ::max(len(path_pts) // 30, 1)],
                       axis=2)
    near = path_pts[::max(len(path_pts) // 30, 1)][np.argmin(d, axis=1)]
    radial = near - centroid
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    inward_frac = float(np.mean(np.einsum('ij,ij->i', fn, radial) > 0))
    if inward_frac < 0.5:
        mesh.flip_winding()


def _slice_centroid_path(verts: np.ndarray, faces: np.ndarray,
                         n_slices: int = 40) -> np.ndarray:
    """主轴切片质心路径: 沿最大特征向量方向切片, 每片取截面多边形加权质心。"""
    center = verts.mean(axis=0)
    X = verts - center
    # 主方向
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis_dir = Vt[0]
    t = X @ axis_dir
    t0, t1 = t.min(), t.max()
    # 对非凸/弯曲器官: 迭代细化 —— 用截平面法向逐段推进, 质心连线
    # 第一遍: 等间距切片质心
    slices = np.linspace(t0, t1, n_slices)
    pts = []
    for s in slices:
        # 该截面附近的顶点
        band = np.abs(t - s) < (t1 - t0) / n_slices * 1.5
        if band.sum() < 3:
            continue
        pts.append(X[band].mean(axis=0) + center)
    pts = np.asarray(pts)
    # 平滑
    if len(pts) >= 5:
        k = 5
        kernel = np.ones(k) / k
        for d in range(3):
            pts[:, d] = np.convolve(np.pad(pts[:, d], (k // 2, k // 2), mode="edge"),
                                    kernel, mode="valid")
    return pts


class OrganMeshScene:
    """真实器官几何场景（接口与 TissueTunnel 兼容）。

    pose_txt: 可选 C3VD pose.txt —— 提供时用真实相机轨迹作管腔路径
    （保证路径在腔内, 且支持"真实轨迹重放"式仿真）; 否则用切片质心路径。
    """

    def __init__(self, mesh_path: str, rng: np.random.Generator,
                 decimate_faces: int = 40000,
                 n_slices: int = 40,
                 tex_scale: float = 0.02,
                 scale: float = 1.0,
                 pose_txt: str | None = None):
        import trimesh
        # process=True: 合并顶点统一绕序(STL来自CAD常为逐面独立法线, 必须先统一)
        tm = trimesh.load(mesh_path, process=True)
        if len(tm.faces) > decimate_faces:
            try:
                tm = tm.simplify_quadric_decimation(face_count=decimate_faces)
            except Exception:
                pass  # 保持原网格(渲染会慢)
        try:
            tm.fix_normals()  # 一致绕序(水密网格→外法线)
        except Exception:
            pass
        verts = np.asarray(tm.vertices, dtype=np.float64) * scale
        faces = np.asarray(tm.faces, dtype=np.int64)

        # 路径: 优先真实C3VD轨迹, 否则切片质心(取中段70%避免端部跑出);
        # 统一做平滑+均匀重采样(原始点距不均/质心跳变会致帧间旋转超限)
        if pose_txt and os.path.exists(pose_txt):
            P = np.loadtxt(pose_txt, delimiter=",")
            T = P.reshape(-1, 4, 4).transpose(0, 2, 1)  # c2w
            # 子采样: 每4帧取一个, 平滑
            path = T[::4][:, :3, 3]
            k = 7
            kernel = np.ones(k) / k
            for d in range(3):
                path[:, d] = np.convolve(np.pad(path[:, d], (k // 2, k // 2), mode="edge"),
                                         kernel, mode="valid")
            self.path_source = "c3vd_poses"
        else:
            path = _slice_centroid_path(verts, faces, n_slices)
            lo, hi = int(len(path) * 0.15), int(len(path) * 0.85)
            path = path[lo:hi]
            self.path_source = "slice_centroid"
        path = _resample_path(path)
        self.centerline = path
        # 平行传输标架: 器官路径扭曲多变, 固定参考轴会在 |dot(t,ref)| 跨阈值时翻转
        from .tissue import _parallel_transport_frames
        self._tangents, self._pt_right, self._pt_up = _parallel_transport_frames(path)
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.arc = np.concatenate([[0], np.cumsum(seg)])
        self.length = float(self.arc[-1])

        # 网格 + 朝内绕序
        self.mesh = SimMesh(verts, faces, triplanar=True, tex_scale=tex_scale)
        self.mesh.compute_normals()
        _orient_inward(self.mesh, path)

        # 近似间隙半径(路径到表面平均距离) —— 供物体放置/曝光参考
        d_surf = np.linalg.norm(verts[:, None, :] - path[None, ::4], axis=2).min(axis=1)
        self.radius_base = float(np.median(d_surf))
        self.radius_axial = np.full(len(path), self.radius_base)
        self.n_rings = len(path)
        self.n_sector = 0
        self.cap_end = False
        self.tex_scale = tex_scale

    # -- 与 TissueTunnel 兼容的接口 ------------------------------------
    def sample_axis_point(self, s: float) -> np.ndarray:
        s = np.clip(s, self.arc[0], self.arc[-1])
        i = int(np.clip(np.searchsorted(self.arc, s) - 1, 0, len(self.arc) - 2))
        t = (s - self.arc[i]) / max(self.arc[i + 1] - self.arc[i], 1e-9)
        return self.centerline[i] * (1 - t) + self.centerline[i + 1] * t

    def axis_pose_at(self, s: float, roll: float = 0.0) -> np.ndarray:
        from ..geometry.se3 import rot_trans, so3_exp
        i = int(np.clip(np.searchsorted(self.arc, s) - 1, 0, len(self.centerline) - 2))
        tangent = self._tangents[i]
        R = np.stack([self._pt_right[i], self._pt_up[i], tangent], axis=1)
        if roll != 0.0:
            R = R @ so3_exp(tangent * np.deg2rad(roll))
        return rot_trans(R, self.sample_axis_point(s))

    def to_dict(self) -> dict:
        return {"kind": "organ_mesh", "length": self.length,
                "radius_base": self.radius_base, "n_faces": self.mesh.n_faces,
                "path_points": len(self.centerline),
                "path_source": self.path_source}
