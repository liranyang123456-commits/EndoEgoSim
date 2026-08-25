"""向量化软件光栅化器（numpy 实现）。

算法:
- 三角形按屏幕 bbox 面积分块; 块内候选像素打包 (像素ID, 量化深度) 为 int64 key,
  stable argsort 后取每像素最后一个(最近深度)为胜者, 再与全局 z-buffer 归并
  —— 与精确 z-buffer 等价。
- 透视校正插值(属性 b/z), 深度按屏幕空间线性插值(平面方程精确)。
- 近平面 Sutherland-Hodgman 裁剪(相机空间, 属性随边线性插值)。
- 背面剔除(几何测试: 保留朝向相机的面, 依赖一致绕序)。
- 支持顶点色 / UV 纹理 / 常量反照率; 内窥镜同轴点光 + Phong 高光。

坐标系: OpenCV 相机系(+X右 +Y下 +Z前); 像素中心 (col+0.5, row+0.5)。
所有渲染输出的深度单位为 mm。
"""
from __future__ import annotations

import numpy as np

from .camera import PinholeCamera


class SimMesh:
    """三角形网格(物体本地坐标)。

    triplanar=True 时用三平面(局部坐标)纹理映射, 无需UV —— 适用于
    真实器官网格(STL/OBJ无UV)。tex_scale: 纹理世界尺寸(mm/纹理重复)。
    """

    def __init__(self, verts, faces, normals=None, colors=None, uvs=None,
                 triplanar=False, tex_scale=0.02):
        self.verts = np.ascontiguousarray(verts, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.normals = (np.ascontiguousarray(normals, dtype=np.float64)
                        if normals is not None else None)
        self.colors = (np.ascontiguousarray(colors, dtype=np.float64)
                       if colors is not None else None)
        self.uvs = (np.ascontiguousarray(uvs, dtype=np.float64)
                    if uvs is not None else None)
        self.triplanar = bool(triplanar)
        self.tex_scale = float(tex_scale)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def compute_normals(self) -> np.ndarray:
        v0, v1, v2 = (self.verts[self.faces[:, i]] for i in range(3))
        fn = np.cross(v1 - v0, v2 - v0)
        normals = np.zeros_like(self.verts)
        for k in range(3):
            np.add.at(normals, self.faces[:, k], fn)
        nrm = np.linalg.norm(normals, axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1.0
        self.normals = normals / nrm
        return self.normals

    def flip_winding(self) -> None:
        self.faces = self.faces[:, [0, 2, 1]]
        if self.normals is not None:
            self.normals = -self.normals


class Material:
    """内窥镜近场点光材质。

    光照模型(光源与相机光心同轴):
      atten = min(d_ref^2 / d^2, atten_max)     # 平方反比衰减(归一化到 d_ref)
      diffuse = albedo * max(n·l,0) * atten
      spec    = ks * max(2(n·l)^2-1, 0)^shininess * atten   # 同轴 Phong
      color   = ambient*albedo + diffuse + spec*spec_color
    """

    def __init__(self, albedo=(1.0, 1.0, 1.0), ambient=0.06,
                 ks=0.25, shininess=40.0, spec_color=(1.0, 1.0, 1.0),
                 d_ref=50.0, atten_max=6.0):
        self.albedo = np.asarray(albedo, dtype=np.float64)
        self.ambient = float(ambient)
        self.ks = float(ks)
        self.shininess = float(shininess)
        self.spec_color = np.asarray(spec_color, dtype=np.float64)
        self.d_ref = float(d_ref)
        self.atten_max = float(atten_max)

    def to_dict(self) -> dict:
        return {"albedo": self.albedo.tolist(), "ambient": self.ambient,
                "ks": self.ks, "shininess": self.shininess,
                "d_ref": self.d_ref, "atten_max": self.atten_max}


class FrameBuffer:
    def __init__(self, width: int, height: int, track_correspondence: bool = False,
                 track_normals: bool = False):
        self.width, self.height = int(width), int(height)
        self.color = np.zeros((height, width, 3), dtype=np.float64)
        self.depth = np.full((height, width), np.inf, dtype=np.float64)
        self.instance = np.zeros((height, width), dtype=np.int32)
        # 对应关系缓冲: 命中像素属于哪个三角形(网格局部ID) + 重心坐标(b0,b1)
        # 用于稠密光流/运动分解GT (材料点级对应)
        self.track_correspondence = track_correspondence
        if track_correspondence:
            self.tri = np.full((height, width), -1, dtype=np.int32)
            self.bary = np.zeros((height, width, 2), dtype=np.float64)
        else:
            self.tri = None
            self.bary = None
        # 着色法线缓冲(相机系, 单位向量) —— 法线图模态导出用
        self.track_normals = track_normals
        if track_normals:
            self.normal = np.zeros((height, width, 3), dtype=np.float64)
        else:
            self.normal = None

    def reset(self) -> None:
        self.color[:] = 0.0
        self.depth[:] = np.inf
        self.instance[:] = 0
        if self.track_correspondence:
            self.tri[:] = -1
            self.bary[:] = 0.0
        if self.track_normals:
            self.normal[:] = 0.0

    @property
    def coverage(self) -> float:
        return float(np.isfinite(self.depth).mean())


# ---------------------------------------------------------------------------
# 近平面裁剪
# ---------------------------------------------------------------------------

def _clip_triangles_near(tri, attr, nrm, near, face_ids=None):
    """裁剪横跨近平面的三角形。返回 (tri, attr, nrm, face_ids) 新三角形列表。"""
    z = tri[:, :, 2]
    n_front = (z >= near).sum(axis=1)
    crossing = (n_front > 0) & (n_front < 3)
    empty = (np.zeros((0, 3, 3)),
             np.zeros((0, 3, attr.shape[2] if attr is not None else 0)),
             np.zeros((0, 3, 3)),
             np.zeros(0, dtype=np.int64))
    if not crossing.any():
        return empty
    out_t, out_a, out_n, out_f = [], [], [], []
    for i in np.flatnonzero(crossing):
        poly = list(tri[i])
        poly_a = list(attr[i]) if attr is not None else None
        poly_n = list(nrm[i])
        new_p, new_a, new_n = [], [], []
        for e in range(3):
            p_cur, p_nxt = poly[e], poly[(e + 1) % 3]
            a_cur = poly_a[e] if poly_a is not None else None
            a_nxt = poly_a[(e + 1) % 3] if poly_a is not None else None
            n_cur, n_nxt = poly_n[e], poly_n[(e + 1) % 3]
            c_in, n_in = p_cur[2] >= near, p_nxt[2] >= near
            if c_in:
                new_p.append(p_cur)
                new_n.append(n_cur)
                if poly_a is not None:
                    new_a.append(a_cur)
            if c_in != n_in:
                t = (near - p_cur[2]) / (p_nxt[2] - p_cur[2])
                pi = p_cur + t * (p_nxt - p_cur)
                pi[2] = near
                new_p.append(pi)
                new_n.append(n_cur + t * (n_nxt - n_cur))
                if poly_a is not None:
                    new_a.append(a_cur + t * (a_nxt - a_cur))
        m = len(new_p)
        for k in range(1, m - 1):
            out_t.append([new_p[0], new_p[k], new_p[k + 1]])
            out_n.append([new_n[0], new_n[k], new_n[k + 1]])
            if poly_a is not None:
                out_a.append([new_a[0], new_a[k], new_a[k + 1]])
            out_f.append(face_ids[i] if face_ids is not None else i)
    if not out_t:
        return empty
    return (np.asarray(out_t), np.asarray(out_a) if out_a else None,
            np.asarray(out_n), np.asarray(out_f, dtype=np.int64))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def render_mesh(fb: FrameBuffer, mesh: SimMesh, cam: PinholeCamera,
                T_co: np.ndarray, instance_id: int,
                material: Material | None = None,
                texture: np.ndarray | None = None,
                near: float = 1.5, cull_backface: bool = True,
                max_chunk_pixels: int = 3_000_000) -> None:
    """把网格以 T_co(物体->相机, 即 cTo) 渲染进帧缓冲。

    texture: uint8 (th,tw,3)，需 mesh.uvs；否则用顶点色/材质常量反照率。
    """
    if material is None:
        material = Material()
    R, t = T_co[:3, :3], T_co[:3, 3]
    verts_cam = mesh.verts @ R.T + t
    if mesh.normals is None:
        mesh.compute_normals()
    nrm_cam = mesh.normals @ R.T

    F = mesh.faces
    tri = verts_cam[F]
    nrm = nrm_cam[F]
    face_ids = np.arange(len(F), dtype=np.int64)  # 原始面索引(对应关系GT用)

    # 顶点属性 (N,3,A): colors + uvs + [triplanar: 本地位置(3) + 本地法线(3)]
    attr = None
    attr_has_colors = mesh.colors is not None
    if attr_has_colors or mesh.uvs is not None or (mesh.triplanar and texture is not None):
        cols = []
        if mesh.colors is not None:
            cols.append(mesh.colors[F])
        if mesh.uvs is not None:
            cols.append(mesh.uvs[F])
        if mesh.triplanar and texture is not None:
            # 本地位置/法线(用于三平面纹理, 固定于物体坐标系避免纹理游动)
            cols.append(mesh.verts[F])
            cols.append(mesh.normals[F] if mesh.normals is not None
                        else np.zeros((len(F), 3, 3)))
        attr = np.concatenate(cols, axis=2)

    # ---- 背面剔除 ----
    if cull_backface:
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        centroid = tri.mean(axis=1)
        facing = np.einsum('ij,ij->i', fn, centroid) < 0.0
        tri, nrm, face_ids = tri[facing], nrm[facing], face_ids[facing]
        if attr is not None:
            attr = attr[facing]

    # ---- 近平面 ----
    z_min = tri[:, :, 2].min(axis=1)
    front = z_min >= near
    if not front.all():
        clipped = _clip_triangles_near(tri[~front], attr[~front] if attr is not None else None,
                                       nrm[~front], near, face_ids[~front])
        tri = np.concatenate([tri[front], clipped[0]])
        nrm = np.concatenate([nrm[front], clipped[2]])
        face_ids = np.concatenate([face_ids[front], clipped[3]])
        if attr is not None:
            if clipped[1] is not None and len(clipped[1]):
                attr = np.concatenate([attr[front], clipped[1]])
            else:
                attr = attr[front]
    if len(tri) == 0:
        return

    _rasterize_scanline(fb, cam, tri, attr, nrm, instance_id,
                        material, texture, attr_has_colors, near,
                        face_ids=face_ids,
                        triplanar=bool(getattr(mesh, "triplanar", False)) and texture is not None,
                        tex_scale=float(getattr(mesh, "tex_scale", 0.02)))


def shade_background(fb: FrameBuffer, color=(0.02, 0.01, 0.015),
                     vignette: float = 0.7) -> None:
    """未命中像素填充内窥镜暗腔背景（带渐晕）。"""
    empty = ~np.isfinite(fb.depth)
    if not empty.any():
        return
    H, W = fb.height, fb.width
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = np.clip(1.0 - vignette * np.clip(r - 0.35, 0, None) ** 1.5, 0.0, 1.0)
    base = np.asarray(color, dtype=np.float64)
    fb.color[empty] = base[None, :] * vig[empty][:, None]


def _rasterize_scanline(fb: FrameBuffer, cam: PinholeCamera,
                        tri: np.ndarray, attr: np.ndarray | None,
                        nrm: np.ndarray, instance_id: int,
                        material: Material, texture: np.ndarray | None,
                        attr_has_colors: bool, near: float,
                        face_ids: np.ndarray | None = None,
                        triplanar: bool = False,
                        tex_scale: float = 0.02) -> None:
    """扫描线光栅化主过程（行线性边函数, 无分块）。

    候选像素 = 每行三角形精确x区间内的像素（凸性保证区间唯一）。
    """
    H, W = fb.height, fb.width
    n = len(tri)
    if n == 0:
        return

    z = np.maximum(tri[:, :, 2], near)
    u = cam.fx * (tri[:, :, 0] / z) + cam.cx
    v = cam.fy * (tri[:, :, 1] / z) + cam.cy

    denom = (u[:, 1] - u[:, 0]) * (v[:, 2] - v[:, 0]) - (v[:, 1] - v[:, 0]) * (u[:, 2] - u[:, 0])
    ok = np.abs(denom) > 1e-9
    if not ok.all():
        keep = ok
        tri, nrm, denom = tri[keep], nrm[keep], denom[keep]
        u, v, z = u[keep], v[keep], z[keep]
        if attr is not None:
            attr = attr[keep]
        n = len(tri)
        if n == 0:
            return

    # ---- 每三角形边函数系数: w_k(x,y) = C_k + P_k*x + Q_k*y ----
    ax, ay = u[:, 0], v[:, 0]
    bx, by = u[:, 1], v[:, 1]
    cx, cy = u[:, 2], v[:, 2]
    C0 = bx * cy - by * cx; P0 = by - cy; Q0 = cx - bx
    C1 = cx * ay - cy * ax; P1 = cy - ay; Q1 = ax - cx
    C2 = ax * by - ay * bx; P2 = ay - by; Q2 = bx - ax
    aden = np.abs(denom)

    # ---- 行范围 ----
    ry0 = np.ceil(v.min(axis=1) - 0.5)
    ry1 = np.floor(v.max(axis=1) - 0.5)
    np.clip(ry0, 0, H - 1, out=ry0)
    np.clip(ry1, 0, H - 1, out=ry1)
    rows_cnt = np.maximum(ry1 - ry0 + 1, 0).astype(np.int64)
    total_rows = int(rows_cnt.sum())
    if total_rows == 0:
        return

    tri_idx = np.repeat(np.arange(n), rows_cnt)
    offs = np.zeros(n, dtype=np.int64)
    if n > 1:
        np.cumsum(rows_cnt[:-1], out=offs[1:])
    within = np.arange(total_rows, dtype=np.int64) - np.repeat(offs, rows_cnt)
    ys = ry0[tri_idx] + within + 0.5

    # ---- 每行精确x区间（边与扫描线求交, 凸性 -> 单区间） ----
    INF = 1e18
    xs_lo = np.full(total_rows, INF)
    xs_hi = np.full(total_rows, -INF)
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        e_ay = v[tri_idx, i]; e_by = v[tri_idx, j]
        e_ax = u[tri_idx, i]; e_bx = u[tri_idx, j]
        span = ((e_ay <= ys) & (ys <= e_by)) | ((e_by <= ys) & (ys <= e_ay))
        horiz = span & (e_ay == e_by)
        slant = span & (e_ay != e_by)
        t = np.zeros(total_rows)
        np.divide(ys - e_ay, e_by - e_ay, out=t, where=slant)
        xc = e_ax + t * (e_bx - e_ax)
        lo = np.where(slant, xc, INF)
        hi = np.where(slant, xc, -INF)
        lo = np.minimum(lo, np.where(horiz, np.minimum(e_ax, e_bx), INF))
        hi = np.maximum(hi, np.where(horiz, np.maximum(e_ax, e_bx), -INF))
        np.minimum(xs_lo, lo, out=xs_lo)
        np.maximum(xs_hi, hi, out=xs_hi)

    # 微膨胀抗缝隙（z-buffer 消解重叠）
    xs_lo -= 0.02
    xs_hi += 0.02
    it_x0 = np.ceil(xs_lo - 0.5).astype(np.int64)
    it_x1 = np.floor(xs_hi - 0.5).astype(np.int64)
    np.clip(it_x0, 0, W - 1, out=it_x0)
    np.clip(it_x1, 0, W - 1, out=it_x1)
    it_y = (ys - 0.5).astype(np.int64)
    keep = (it_x1 >= it_x0) & (xs_hi > -INF)
    if not keep.any():
        return
    it_x0, it_x1, it_y = it_x0[keep], it_x1[keep], it_y[keep]
    tri_idx = tri_idx[keep]

    # ---- 展开候选像素 ----
    counts = it_x1 - it_x0 + 1
    total = int(counts.sum())
    if total == 0:
        return
    item_rep = np.repeat(np.arange(len(it_x0)), counts)
    offs2 = np.zeros(len(it_x0), dtype=np.int64)
    if len(it_x0) > 1:
        np.cumsum(counts[:-1], out=offs2[1:])
    within2 = np.arange(total, dtype=np.int64) - np.repeat(offs2, counts)
    px = np.repeat(it_x0, counts) + within2
    py = np.repeat(it_y, counts)
    del within2, offs2

    # ---- 行线性边函数求重心权重 ----
    pxs = px + 0.5
    t_r = tri_idx[item_rep]
    y_r = py + 0.5
    a0 = C0[t_r] + Q0[t_r] * y_r
    a1 = C1[t_r] + Q1[t_r] * y_r
    a2 = C2[t_r] + Q2[t_r] * y_r
    b0 = b1 = None  # 占位
    w0 = a0 + P0[t_r] * pxs
    w1 = a1 + P1[t_r] * pxs
    w2 = a2 + P2[t_r] * pxs
    den_r = denom[t_r]
    be0 = w0 / den_r
    be1 = w1 / den_r
    be2 = w2 / den_r
    del w0, w1, w2, a0, a1, a2, pxs, y_r

    # 有效性掩码: 排除区间边界外的重心外推候选（退化/掠射三角形）
    valid = (be0 >= -0.01) & (be1 >= -0.01) & (be2 >= -0.01)
    if not valid.all():
        px, py, t_r = px[valid], py[valid], t_r[valid]
        be0, be1, be2, den_r = be0[valid], be1[valid], be2[valid], den_r[valid]
        if len(px) == 0:
            return

    # ---- 深度(调和插值) ----
    z0r, z1r, z2r = z[t_r, 0], z[t_r, 1], z[t_r, 2]
    S = be0 / z0r + be1 / z1r + be2 / z2r
    Ssafe = np.where(np.abs(S) < 1e-12, 1e-12, S)
    depth = 1.0 / Ssafe
    pixel_id = py * W + px
    del z0r, z1r, z2r, S, Ssafe, be0, be1, be2

    # ---- 每像素最近深度胜者（打包key稳定排序） ----
    depth_q = np.clip(np.round(depth * 1024.0), 0, 2 ** 31 - 2).astype(np.int64)
    key = pixel_id * np.int64(2 ** 31) + (np.int64(2 ** 31 - 1) - depth_q)
    del depth_q
    order = np.argsort(key, kind='stable')
    del key
    pid_s = pixel_id[order]
    is_last = np.empty(len(pid_s), dtype=bool)
    is_last[:-1] = pid_s[1:] != pid_s[:-1]
    is_last[-1] = True
    win = order[is_last]
    del order, pid_s, is_last
    if len(win) == 0:
        return

    win_pid = pixel_id[win]
    win_depth = depth[win]
    upd = win_depth < fb.depth.reshape(-1)[win_pid] - 1e-6
    if not upd.any():
        return
    win_pid, win_depth = win_pid[upd], win_depth[upd]
    win_px = (win_pid % W).astype(np.int64)
    win_py = (win_pid // W).astype(np.int64)
    wsel = win[upd]
    del win, upd, pixel_id, depth

    # ---- 胜者属性插值（透视校正） ----
    wt = t_r[wsel]
    pxw = win_px + 0.5
    pyw = win_py + 0.5
    w0w = C0[wt] + P0[wt] * pxw + Q0[wt] * pyw
    w1w = C1[wt] + P1[wt] * pxw + Q1[wt] * pyw
    w2w = C2[wt] + P2[wt] * pxw + Q2[wt] * pyw
    den_w = denom[wt]
    be0w = w0w / den_w
    be1w = w1w / den_w
    be2w = w2w / den_w
    iz0 = 1.0 / z[wt, 0]
    iz1 = 1.0 / z[wt, 1]
    iz2 = 1.0 / z[wt, 2]
    bsum = be0w * iz0 + be1w * iz1 + be2w * iz2
    bsum = np.where(np.abs(bsum) < 1e-12, 1e-12, bsum)
    B0 = be0w * iz0 / bsum
    B1 = be1w * iz1 / bsum
    B2 = be2w * iz2 / bsum

    P = (B0[:, None] * tri[wt, 0] + B1[:, None] * tri[wt, 1] + B2[:, None] * tri[wt, 2])
    nrm_p = (B0[:, None] * nrm[wt, 0] + B1[:, None] * nrm[wt, 1] + B2[:, None] * nrm[wt, 2])
    nn = np.linalg.norm(nrm_p, axis=1, keepdims=True)
    nn[nn < 1e-12] = 1.0
    nrm_p = nrm_p / nn

    # ---- 反照率 ----
    A = None
    if attr is not None:
        A = (B0[:, None] * attr[wt, 0] + B1[:, None] * attr[wt, 1] + B2[:, None] * attr[wt, 2])
    def _tex_bilinear(texf, uu, vv):
        th, tw = texf.shape[:2]
        map_u = uu - 0.5
        map_v = vv - 0.5
        u0f = np.floor(map_u).astype(np.int64)
        v0f = np.floor(map_v).astype(np.int64)
        du, dv = map_u - u0f, map_v - v0f
        u0, v0 = np.mod(u0f, tw), np.mod(v0f, th)
        u1, v1 = np.mod(u0f + 1, tw), np.mod(v0f + 1, th)
        return (texf[v0, u0] * ((1 - du) * (1 - dv))[:, None]
                + texf[v0, u1] * (du * (1 - dv))[:, None]
                + texf[v1, u0] * ((1 - du) * dv)[:, None]
                + texf[v1, u1] * (du * dv)[:, None]).astype(np.float64)

    n_col = 3 if attr_has_colors else 0
    if triplanar and A is not None and texture is not None:
        # 三平面: 本地位置/法线 -> 三主平面纹理加权混合(权重|n|^4锐化)
        pos_l = A[:, n_col:n_col + 3]
        n_l = A[:, n_col + 3:n_col + 6]
        texf = texture.astype(np.float32) / 255.0
        th, tw = texture.shape[:2]
        an = np.abs(n_l)
        w = an ** 4
        wsum = np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        w = w / wsum
        sc = 1.0 / max(tex_scale, 1e-6)
        cX = _tex_bilinear(texf, pos_l[:, 1] * sc, pos_l[:, 2] * sc)
        cY = _tex_bilinear(texf, pos_l[:, 0] * sc, pos_l[:, 2] * sc)
        cZ = _tex_bilinear(texf, pos_l[:, 0] * sc, pos_l[:, 1] * sc)
        albedo = (cX * w[:, 0:1] + cY * w[:, 1:2] + cZ * w[:, 2:3])
    elif A is not None and attr_has_colors:
        albedo = np.clip(A[:, :3], 0, 1)
    elif A is not None and texture is not None:
        uv = A[:, :2]
        texf = texture.astype(np.float32) / 255.0
        th, tw = texture.shape[:2]
        albedo = _tex_bilinear(texf, uv[:, 0] * tw, uv[:, 1] * th)
    else:
        albedo = np.tile(material.albedo, (len(win_pid), 1))

    # ---- 内窥镜同轴点光 ----
    d = np.maximum(np.linalg.norm(P, axis=1), 1e-3)
    l = -P / d[:, None]
    ndl = np.clip(np.einsum('ij,ij->i', nrm_p, l), 0.0, 1.0)
    atten = np.clip(material.d_ref ** 2 / d ** 2, 0.0, material.atten_max)
    diffuse = albedo * (ndl * atten)[:, None]
    rv = np.clip(2.0 * ndl ** 2 - 1.0, 0.0, 1.0)
    spec = material.ks * rv ** material.shininess * atten
    color = material.ambient * albedo + diffuse + spec[:, None] * material.spec_color[None, :]

    fb.color[win_py, win_px] = np.clip(color, 0.0, 1.0)
    fb.depth[win_py, win_px] = win_depth
    fb.instance[win_py, win_px] = instance_id
    if fb.track_normals:
        fb.normal[win_py, win_px] = nrm_p
    if fb.track_correspondence and face_ids is not None:
        fb.tri[win_py, win_px] = face_ids[wt]
        fb.bary[win_py, win_px, 0] = B0
        fb.bary[win_py, win_px, 1] = B1


