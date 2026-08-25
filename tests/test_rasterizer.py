"""光栅化器测试: 球体解析深度验证 + 纹理平面验证。"""
import sys, time
sys.path.insert(0, r'D:\ego_motiion_Camera')

import numpy as np
from endosim.render.rasterizer import SimMesh, Material, FrameBuffer, render_mesh, shade_background
from endosim.render.camera import PinholeCamera


def make_sphere(r=30.0, center=(0, 0, 100), n_lat=48, n_lon=64):
    lat = np.linspace(0, np.pi, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    L, O = np.meshgrid(lat, lon, indexing='ij')
    x = r * np.sin(L) * np.cos(O) + center[0]
    y = r * np.sin(L) * np.sin(O) + center[1]
    z = r * np.cos(L) + center[2]
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    idx = np.arange(n_lat * n_lon).reshape(n_lat, n_lon)
    f = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            a, b, c = idx[i, j], idx[i + 1, j], idx[i + 1, (j + 1) % n_lon]
            d = idx[i, (j + 1) % n_lon]
            f.append([a, b, c]); f.append([a, c, d])
    return SimMesh(verts, np.asarray(f))


def ray_sphere_depth(cam, px, py, r=30.0, c=(0, 0, 100)):
    """解析解: 像素光线与球最近交点的 z 深度(与深度缓冲同口径)。"""
    du = (px + 0.5 - cam.cx) / cam.fx
    dv = (py + 0.5 - cam.cy) / cam.fy
    d = np.array([du, dv, 1.0])  # z分量=1, 参数t即深度
    c = np.asarray(c, float)
    b = d @ c
    disc = b * b - (d @ d) * (c @ c - r * r)
    if disc < 0:
        return None
    return (b - np.sqrt(disc)) / (d @ d)


def main():
    cam = PinholeCamera(320.0, 320.0, 320.0, 256.0, 640, 512)
    fb = FrameBuffer(640, 512)
    sph = make_sphere()
    sph.compute_normals()
    mat = Material(albedo=(0.8, 0.2, 0.2), ks=0.3, shininess=50, d_ref=100)

    t0 = time.time()
    render_mesh(fb, sph, cam, np.eye(4), instance_id=1, material=mat)
    dt = time.time() - t0
    shade_background(fb)

    d = fb.depth
    valid = np.isfinite(d)
    print(f"渲染耗时 {dt*1000:.0f} ms, 覆盖率 {valid.mean()*100:.2f}% (理论≈9.7%)")
    assert 0.08 < valid.mean() < 0.105, "覆盖率异常"
    assert abs(d[256, 320] - 70.0) < 0.5, f"中心深度 {d[256,320]} 应≈70"

    # 解析抽查若干像素深度
    errs = []
    for (px, py) in [(320, 256), (360, 256), (400, 280), (300, 200), (340, 300), (280, 256)]:
        ref = ray_sphere_depth(cam, px, py)
        got = d[py, px]
        if ref is not None and np.isfinite(got):
            errs.append(abs(got - ref))
    max_err = max(errs)
    print(f"解析深度抽查 {len(errs)} 点, 最大误差 {max_err:.4f} mm")
    assert max_err < 0.5, "深度误差过大"

    # 性能: 高面数球
    big = make_sphere(n_lat=128, n_lon=192)
    big.compute_normals()
    fb2 = FrameBuffer(640, 512)
    t0 = time.time()
    render_mesh(fb2, big, cam, np.eye(4), instance_id=1, material=mat)
    print(f"高面数球({big.n_faces}面)渲染 {1000*(time.time()-t0):.0f} ms, 覆盖 {fb2.coverage*100:.2f}%")
    assert abs(fb2.depth[256, 320] - 70.0) < 0.5

    # ---- 纹理平面验证 ----
    fb3 = FrameBuffer(640, 512)
    tex = np.zeros((64, 64, 3), dtype=np.uint8)
    tex[:, :32] = [255, 0, 0]     # 左半红 (u<0.5)
    tex[:, 32:] = [0, 0, 255]     # 右半蓝
    # 平面 z=80, u沿+x
    v = np.array([[-40., -32., 80.], [40., -32., 80.], [40., 32., 80.], [-40., 32., 80.]])
    uv = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    m = SimMesh(v, f, uvs=uv)
    m.compute_normals()
    # 使法线朝向相机(绕序): 检查
    tri = m.verts[m.faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if fn[:, 2].mean() > 0:
        m.flip_winding()
    render_mesh(fb3, m, cam, np.eye(4), 2, Material(ks=0.0, d_ref=80), texture=tex)
    cov3 = fb3.coverage
    left_px = fb3.color[256, 300]   # u<0.5 → 红
    right_px = fb3.color[256, 340]  # u>0.5 → 蓝
    print(f"纹理平面: 覆盖 {cov3*100:.1f}%, 左半色 {np.round(left_px,2)}, 右半色 {np.round(right_px,2)}")
    assert left_px[0] > 0.5 and left_px[2] < 0.3, "左半应为红色"
    assert right_px[2] > 0.5 and right_px[0] < 0.3, "右半应为蓝色"
    assert abs(fb3.depth[256, 320] - 80.0) < 0.5, "平面深度应=80"

    print("\n✅ 全部光栅化器测试通过")

    import cv2
    img = (np.clip(fb.color, 0, 1)[..., ::-1] * 255).astype(np.uint8)
    cv2.imwrite(r'D:\ego_motiion_Camera\tests\sphere_test.png', img)
    dn = np.zeros(d.shape, dtype=np.uint8)
    dn[valid] = ((d[valid] - 60) / 40 * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(r'D:\ego_motiion_Camera\tests\sphere_depth.png', dn)


if __name__ == '__main__':
    main()
