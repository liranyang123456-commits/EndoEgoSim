"""GPU 光栅化器 vs CPU 光栅化器 一致性对比测试。

同一网格/相机/材质/纹理, 分别用 CPU (render_mesh) 与 GPU (render_mesh_gpu) 渲染,
比较: 深度/颜色/实例/三角形ID/重心坐标。

用法: python tests/test_gpu_raster.py [--no-texture]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.render.camera import PinholeCamera
from endosim.render.gpu_rasterizer import (MeshGPU, buffers_to_numpy,
                                           get_glctx, new_buffers,
                                           render_mesh_gpu)
from endosim.render.rasterizer import (FrameBuffer, Material, SimMesh,
                                       render_mesh, shade_background)
from endosim.scene.tissue import TissueTunnel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-texture", action="store_true")
    ap.add_argument("--triplanar", action="store_true", help="强制三平面纹理路径")
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    tunnel = TissueTunnel(rng, length=400, radius_base=32, n_rings=120,
                          n_sector=48)
    if args.triplanar:  # 强制走三平面纹理路径（无UV网格）
        tunnel.mesh = SimMesh(tunnel.mesh.verts, tunnel.mesh.faces,
                              triplanar=True, tex_scale=25.0)
        tunnel.mesh.compute_normals()
    cam = PinholeCamera.from_fov(90, 640, 512)
    # 相机放在管腔内 1/3 处沿轴向看
    T_wc = tunnel.axis_pose_at(tunnel.arc[-1] * 0.3)
    T_cw = np.linalg.inv(T_wc)
    mat = Material(albedo=(0.9, 0.7, 0.65), ambient=0.06, ks=0.25,
                   shininess=40.0, d_ref=45.0, atten_max=6.0)

    texture = None
    if not args.no_texture:
        if args.triplanar:
            # 程序化纹理
            from endosim.scene.texture import _procedural_texture
            texture = _procedural_texture(rng, 512)
        else:
            # 隧道有 UV: 用程序化纹理走 UV 路径
            from endosim.scene.texture import _procedural_texture
            texture = _procedural_texture(rng, 512)

    mesh = tunnel.mesh  # UV 网格(triplanar=False 时)

    # ---- CPU 渲染 ----
    fb = FrameBuffer(cam.width, cam.height, track_correspondence=True)
    render_mesh(fb, mesh, cam, T_cw, instance_id=1, material=mat,
                texture=texture, near=2.0)
    shade_background(fb)

    # ---- GPU 渲染 ----
    import torch
    glctx = get_glctx()
    mesh_gpu = MeshGPU(mesh)
    tex_gpu = (torch.as_tensor(texture, device="cuda").float() / 255.0
               if texture is not None else None)
    bufs = new_buffers(cam.height, cam.width, track=True)
    # 法线与 CPU 一致: CPU 用 mesh.normals (本地系)
    normals = (torch.as_tensor(np.asarray(mesh.normals, np.float32), device="cuda")
               if mesh.normals is not None else None)
    verts = mesh_gpu.verts_with_disp(None)
    render_mesh_gpu(glctx, bufs, mesh_gpu, verts, normals, cam, T_cw,
                    1, mat, tex_gpu, near=2.0)
    g = buffers_to_numpy(bufs)

    # ---- 对比 ----
    cpu_d, gpu_d = fb.depth, g["depth"]
    cpu_hit, gpu_hit = np.isfinite(cpu_d), np.isfinite(gpu_d)
    both = cpu_hit & gpu_hit
    print(f"覆盖: CPU {cpu_hit.mean()*100:.1f}% / GPU {gpu_hit.mean()*100:.1f}% "
          f"/ 共同 {both.mean()*100:.1f}%")
    derr = np.abs(cpu_d[both] - gpu_d[both])
    print(f"深度(共同命中): 中位 {np.median(derr):.5f} mm, p99 {np.percentile(derr,99):.4f} mm, "
          f"max {derr.max():.3f} mm")

    cerr = np.abs(fb.color[both] - g["color"][both])
    print(f"颜色(共同命中): 均值 {cerr.mean():.5f}, p99 {np.percentile(cerr,99):.4f}, max {cerr.max():.4f}")

    if both.any():
        tri_agree = (fb.tri[both] == g["tri"][both]).mean()
        print(f"三角形ID一致率: {tri_agree*100:.3f}%")
        same = both & (fb.tri == g["tri"])
        if same.any():
            berr = np.abs(fb.bary[same] - g["bary"][same]).max()
            print(f"重心坐标(同三角形像素): max |ΔB| = {berr:.6f}")
    inst_agree = (fb.instance[both] == g["instance"][both]).mean()
    print(f"实例ID一致率: {inst_agree*100:.1f}%")


if __name__ == "__main__":
    main()
