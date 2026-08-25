"""GPU 光栅化器（nvdiffrast + torch）—— CPU 软件光栅化器的高性能等价实现。

与 endosim.render.rasterizer 的语义对齐:
- 背面剔除: 相机空间几何测试 (fn·centroid<0), 剔除后映射回原始面 ID
- 近平面: clip 空间 z 映射 z_ndc=(z-near*2)/z, 由 nvdiffrast 裁剪
- 深度/属性: 透视校正插值(与 CPU 调和插值/属性 b/z 一致), 深度单位 mm
- 着色: 内窥镜同轴点光 (ambient + 1/d² 衰减 diffuse + 同轴 Phong 高光)
- 纹理: 三平面/UV/顶点色/常量反照率, 手工双线性(带 wrap, 与 CPU 逐位一致)
- 对应关系: 原始面 ID + 透视校正重心 (B0,B1) —— 光流/运动分解 GT 用

坐标系: OpenCV 相机系(+X右 +Y下 +Z前)。clip 映射:
  clip = [2fx·x+(2cx−W)z, (H−2cy)z−2fy·y, z−2·near, z]  (w=z)
  使像素中心 (col+.5, row+.5) 与 CPU 完全对齐, y 翻转适配 OpenGL 约定。
"""
from __future__ import annotations

import numpy as np
import torch

from .rasterizer import Material, SimMesh

_GLCTX = None


def get_glctx():
    """单例 CUDA 光栅化上下文。"""
    global _GLCTX
    if _GLCTX is None:
        import nvdiffrast.torch as dr
        _GLCTX = dr.RasterizeCudaContext()
    return _GLCTX


class MeshGPU:
    """网格的 GPU 缓存（顶点/面 float32/int32）。形变网格每帧更新 verts。"""

    def __init__(self, mesh: SimMesh, device="cuda"):
        self.device = device
        self.faces = torch.as_tensor(mesh.faces, dtype=torch.int32, device=device)
        self.faces_long = self.faces.long()
        self.n_faces = len(mesh.faces)
        self.base_verts_np = np.ascontiguousarray(mesh.verts, dtype=np.float32)
        self.triplanar = bool(mesh.triplanar)
        self.tex_scale = float(mesh.tex_scale)
        # 静态属性(物体本地系, 不随位姿变化): colors + uvs
        # 三平面纹理的本地位置/法线需逐帧计算(形变网格), 不在此缓存
        has_col = mesh.colors is not None
        has_uv = mesh.uvs is not None
        parts = []
        if has_col:
            parts.append(np.asarray(mesh.colors, dtype=np.float32))
        if has_uv:
            parts.append(np.asarray(mesh.uvs, dtype=np.float32))
        self.attr_local = (torch.as_tensor(np.concatenate(parts, 1), device=device)
                           if parts else None)
        self.attr_has_colors = has_col
        self.static_normals = (torch.as_tensor(np.asarray(mesh.normals, np.float32), device=device)
                               if mesh.normals is not None else None)

    def verts_with_disp(self, disp: np.ndarray | None):
        v = self.base_verts_np
        if disp is not None:
            v = v + disp.astype(np.float32)
        return torch.as_tensor(np.ascontiguousarray(v), device=self.device)


def _compute_normals_torch(verts: torch.Tensor, faces_long: torch.Tensor) -> torch.Tensor:
    """顶点法线(面积加权), 与 SimMesh.compute_normals 一致。"""
    v0 = verts[faces_long[:, 0]]
    v1 = verts[faces_long[:, 1]]
    v2 = verts[faces_long[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = torch.zeros_like(verts)
    for k in range(3):
        normals.index_add_(0, faces_long[:, k], fn)
    normals = normals / torch.clamp(normals.norm(dim=1, keepdim=True), min=1e-12)
    return normals


def _tex_bilinear_torch(texf: torch.Tensor, uu: torch.Tensor, vv: torch.Tensor) -> torch.Tensor:
    """双线性纹理采样, 坐标 wrap —— 与 CPU _tex_bilinear 逐位一致。
    texf: (th,tw,3) float; uu/vv: (N,) 像素单位纹理坐标。"""
    th, tw = texf.shape[:2]
    map_u = uu - 0.5
    map_v = vv - 0.5
    u0f = torch.floor(map_u).long()
    v0f = torch.floor(map_v).long()
    du = (map_u - u0f.to(map_u.dtype))[:, None]
    dv = (map_v - v0f.to(map_v.dtype))[:, None]
    u0 = u0f % tw
    v0 = v0f % th
    u1 = (u0f + 1) % tw
    v1 = (v0f + 1) % th
    c00 = texf[v0, u0]
    c01 = texf[v0, u1]
    c10 = texf[v1, u0]
    c11 = texf[v1, u1]
    return (c00 * (1 - du) * (1 - dv) + c01 * du * (1 - dv)
            + c10 * (1 - du) * dv + c11 * du * dv)


def render_mesh_gpu(glctx, bufs: dict, mesh: MeshGPU,
                    verts_local: torch.Tensor, normals_local: torch.Tensor | None,
                    cam, T_co: np.ndarray, instance_id: int,
                    material: Material, texture: torch.Tensor | None,
                    near: float):
    """把一个网格渲染并按深度合成进全局缓冲 bufs。

    bufs: {color:(H,W,3), depth:(H,W), instance:(H,W) int32, tri:(H,W) int32, bary:(H,W,2)}
    （torch cuda, depth 初始 +inf, tri 初始 -1）
    """
    import nvdiffrast.torch as dr
    H, W = bufs["color"].shape[:2]
    R = torch.as_tensor(T_co[:3, :3], dtype=torch.float32, device=verts_local.device)
    t = torch.as_tensor(T_co[:3, 3], dtype=torch.float32, device=verts_local.device)
    verts_cam = verts_local @ R.T + t
    if normals_local is None:
        normals_local = _compute_normals_torch(verts_local, mesh.faces_long)
    nrm_cam = normals_local @ R.T

    F = mesh.faces_long
    tri_cam = verts_cam[F]                       # (F,3,3)
    fn = torch.cross(tri_cam[:, 1] - tri_cam[:, 0], tri_cam[:, 2] - tri_cam[:, 0], dim=1)
    centroid = tri_cam.mean(dim=1)
    facing = (fn * centroid).sum(dim=1) < 0.0    # 与 CPU 几何背面剔除一致
    if not bool(facing.any()):
        return
    facing_idx = torch.nonzero(facing, as_tuple=False).reshape(-1)

    # clip 空间 (w=z, y 翻转适配 nvdiffrast OpenGL 约定: NDC_y=-1 为图像顶部)
    # NDC_x = 2u/W-1, NDC_y = 2v/H-1 (u,v=像素坐标) => clip = NDC*w
    x, y, z = verts_cam[:, 0], verts_cam[:, 1], verts_cam[:, 2]
    zs = torch.clamp(z, min=near)                # 退化保护
    clip = torch.stack([(2 * cam.fx / W) * x + (2 * cam.cx - W) / W * zs,
                        (2 * cam.fy / H) * y + (2 * cam.cy - H) / H * zs,
                        z - 2.0 * near, z], dim=1)[None]
    tri_sub = mesh.faces[facing_idx]              # (F',3) 本版本不接受批维

    rast, _ = dr.rasterize(glctx, clip.contiguous(), tri_sub.contiguous(),
                           resolution=[H, W])
    rast = rast[0]
    hit = rast[:, :, 3] > 0
    if not bool(hit.any()):
        return
    hit_idx = torch.nonzero(hit.reshape(-1), as_tuple=False).reshape(-1)
    rast_hit = rast.reshape(-1, 4)[hit_idx].float()
    py = (hit_idx // W).long()
    px = (hit_idx % W).long()
    tri_hit = rast_hit[:, 3].long() - 1          # nvdiffrast 三角形ID从1开始
    orig_face = facing_idx[tri_hit]              # 映射回原始面ID(对应关系GT)

    # rast(u,v) 即透视校正重心 (B0,B1) —— 与 CPU fb.bary 语义一致(已数值验证)
    sub_tri = tri_sub[tri_hit]                    # 面顶点索引(局部)
    B0 = rast_hit[:, 0]
    B1 = rast_hit[:, 1]
    B2 = 1.0 - B0 - B1

    # 属性插值: 位置/法线(相机系) + 本地属性(纹理用)
    vc = tri_cam[orig_face]
    P = B0[:, None] * vc[:, 0] + B1[:, None] * vc[:, 1] + B2[:, None] * vc[:, 2]
    nrm_f = nrm_cam[sub_tri]
    N = B0[:, None] * nrm_f[:, 0] + B1[:, None] * nrm_f[:, 1] + B2[:, None] * nrm_f[:, 2]
    N = N / torch.clamp(N.norm(dim=1, keepdim=True), min=1e-12)

    depth = P[:, 2]

    # ---- 反照率 ----
    albedo = None
    A = None
    if mesh.attr_local is not None:
        A = (B0[:, None] * mesh.attr_local[sub_tri[:, 0]]
             + B1[:, None] * mesh.attr_local[sub_tri[:, 1]]
             + B2[:, None] * mesh.attr_local[sub_tri[:, 2]])
    n_col = 3 if mesh.attr_has_colors else 0
    if mesh.triplanar and texture is not None:
        # 本地位置/法线逐帧插值(与 CPU 一致: CPU 读逐帧形变后的 mesh.verts/normals)
        Pl = (B0[:, None] * verts_local[sub_tri[:, 0]]
              + B1[:, None] * verts_local[sub_tri[:, 1]]
              + B2[:, None] * verts_local[sub_tri[:, 2]])
        Nl = (B0[:, None] * normals_local[sub_tri[:, 0]]
              + B1[:, None] * normals_local[sub_tri[:, 1]]
              + B2[:, None] * normals_local[sub_tri[:, 2]])
        an = Nl.abs()
        wgt = an ** 4
        wgt = wgt / torch.clamp(wgt.sum(dim=1, keepdim=True), min=1e-12)
        sc = 1.0 / max(mesh.tex_scale, 1e-6)
        cX = _tex_bilinear_torch(texture, Pl[:, 1] * sc, Pl[:, 2] * sc)
        cY = _tex_bilinear_torch(texture, Pl[:, 0] * sc, Pl[:, 2] * sc)
        cZ = _tex_bilinear_torch(texture, Pl[:, 0] * sc, Pl[:, 1] * sc)
        albedo = cX * wgt[:, 0:1] + cY * wgt[:, 1:2] + cZ * wgt[:, 2:3]
    elif A is not None and mesh.attr_has_colors:
        albedo = A[:, :3].clamp(0.0, 1.0)
    elif A is not None and texture is not None:
        th, tw = texture.shape[:2]
        albedo = _tex_bilinear_torch(texture, A[:, 0] * tw, A[:, 1] * th)
    if albedo is None:
        albedo = torch.as_tensor(material.albedo, dtype=torch.float32,
                                 device=P.device)[None, :].expand(len(P), 3)

    # ---- 内窥镜同轴点光 (与 CPU 公式一致) ----
    d = torch.clamp(P.norm(dim=1), min=1e-3)
    l = -P / d[:, None]
    ndl = (N * l).sum(dim=1).clamp(0.0, 1.0)
    atten = (material.d_ref ** 2 / d ** 2).clamp(0.0, material.atten_max)
    diffuse = albedo * (ndl * atten)[:, None]
    rv = (2.0 * ndl ** 2 - 1.0).clamp(0.0, 1.0)
    spec = material.ks * rv ** material.shininess * atten
    spec_color = torch.as_tensor(material.spec_color, dtype=torch.float32,
                                 device=P.device)
    color = material.ambient * albedo + diffuse + spec[:, None] * spec_color[None, :]
    color = color.clamp(0.0, 1.0)

    # ---- 深度合成进全局缓冲 ----
    old_depth = bufs["depth"][py, px]
    upd = depth < old_depth
    if not bool(upd.any()):
        return
    py, px = py[upd], px[upd]
    bufs["color"][py, px] = color[upd]
    bufs["depth"][py, px] = depth[upd]
    bufs["instance"][py, px] = instance_id
    if "normal" in bufs:
        bufs["normal"][py, px] = N[upd]
    if "tri" in bufs:
        bufs["tri"][py, px] = orig_face[upd].to(torch.int32)
        bufs["bary"][py, px, 0] = B0[upd]
        bufs["bary"][py, px, 1] = B1[upd]


compute_normals_gpu = _compute_normals_torch


def shade_background_gpu(bufs: dict, color=(0.02, 0.01, 0.015),
                         vignette: float = 0.7) -> None:
    """未命中像素填充内窥镜暗腔背景（带渐晕）—— 与 CPU shade_background 一致。"""
    H, W = bufs["color"].shape[:2]
    empty = ~torch.isfinite(bufs["depth"])
    if not bool(empty.any()):
        return
    yy, xx = torch.meshgrid(torch.arange(H, device=bufs["color"].device, dtype=torch.float32),
                            torch.arange(W, device=bufs["color"].device, dtype=torch.float32),
                            indexing="ij")
    r = torch.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = (1.0 - vignette * torch.clamp(r - 0.35, min=0.0) ** 1.5).clamp(0.0, 1.0)
    base = torch.tensor(color, dtype=torch.float32, device=bufs["color"].device)
    bufs["color"] = torch.where(empty[..., None], base[None, None, :] * vig[..., None],
                                bufs["color"])


def new_buffers(height: int, width: int, device="cuda", track=True,
                track_normals=False) -> dict:
    bufs = {
        "color": torch.zeros(height, width, 3, dtype=torch.float32, device=device),
        "depth": torch.full((height, width), float("inf"), dtype=torch.float32, device=device),
        "instance": torch.zeros(height, width, dtype=torch.int32, device=device),
    }
    if track:
        bufs["tri"] = torch.full((height, width), -1, dtype=torch.int32, device=device)
        bufs["bary"] = torch.zeros(height, width, 2, dtype=torch.float32, device=device)
    if track_normals:
        bufs["normal"] = torch.zeros(height, width, 3, dtype=torch.float32, device=device)
    return bufs


def buffers_to_numpy(bufs: dict) -> dict:
    out = {"color": bufs["color"].cpu().numpy().astype(np.float64),
           "depth": bufs["depth"].cpu().numpy().astype(np.float64),
           "instance": bufs["instance"].cpu().numpy()}
    if "tri" in bufs:
        out["tri"] = bufs["tri"].cpu().numpy()
        out["bary"] = bufs["bary"].cpu().numpy().astype(np.float64)
    if "normal" in bufs:
        out["normal"] = bufs["normal"].cpu().numpy().astype(np.float64)
    return out
