"""单序列生成编排: 场景采样 -> 轨迹 -> 逐帧渲染 -> 全部GT计算。

一条序列 = 一次完整仿真，含:
- 组织腔道（默认非刚性形变: 蠕动波/局灶搏动/谐波呼吸/接触压陷）
- 0~2 个动态物体（器械/BOP物体）+ 可选贴壁棋盘格标记（显式参照物）
- 相机轨迹
- 逐帧渲染: RGB / 深度 / 实例mask / 对应关系(三角形ID+重心)
- GT: T_wc(绝对) / ΔT_ij(相对) / T_wo / T_co / 深度 / bbox
  + 稠密前向光流 / 运动分解掩码(静态参照|形变组织|运动物体) / 参照物比例

坐标系: 世界系 = 首帧相机系（首帧 c2w = I）。
运动分解定义（世界系场景运动, 与"绝对运动锚定"严格一致）:
  label 1 静态参照: 材料点世界位移 < eps  → 可用于绝对 egomotion
  label 2 形变组织: 组织材料点位移 ≥ eps
  label 3 运动物体: 物体世界位姿变化 ≥ eps
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import GenConfig, sample_int
from ..geometry.se3 import ensure_se3, relative
from ..geometry.trajectories import generate_trajectory, motion_stats
from ..motion.deformation import TissueDeformation
from ..motion.object_motion import object_motion
from ..render.camera import PinholeCamera
from ..render.rasterizer import (FrameBuffer, Material, SimMesh, render_mesh,
                                 shade_background)
from ..render.sensor import apply_sensor
from ..scene.objects import (BOP_MATERIAL, INSTRUMENT_MATERIAL, TISSUE_MATERIAL,
                             MARKER_MATERIAL, available_bop_ids, load_bop_object,
                             make_instrument, make_marker_patch)
from ..scene.texture import TextureBank, make_texture
from ..scene.tissue import TissueTunnel


@dataclass
class SceneObject:
    """序列中的动态物体（或静态标记）。"""
    name: str
    mesh: SimMesh
    material: Material
    texture: np.ndarray | None
    poses_wo: np.ndarray                     # (N,4,4) 世界系位姿
    info: dict
    is_marker: bool = False


@dataclass
class SequenceData:
    """一条生成序列的全部数据（内存态, 由 writer 落盘）。"""
    seq_id: str
    n_frames: int
    camera: PinholeCamera
    poses_wc: np.ndarray                     # (N,4,4) 绝对位姿 GT
    objects: list = field(default_factory=list)
    frames: list = field(default_factory=list)   # 每帧 dict(color, depth, instance, bboxes)
    motion: list = field(default_factory=list)   # 每对帧 dict(flow, mask, ref_frac)
    meta: dict = field(default_factory=dict)

    def relative_poses(self, i: int, j: int) -> np.ndarray:
        return relative(self.poses_wc[i], self.poses_wc[j])


def _sample_uniform(rng, lo_hi):
    lo, hi = lo_hi
    return rng.uniform(lo, hi)


def generate_sequence(seq_id: str, seed: int, cfg: GenConfig,
                      bank: TextureBank | None = None,
                      split: str = "train",
                      texture_size: int = 1024,
                      meta_split: str | None = None,
                      use_gpu: bool = False) -> SequenceData:
    """生成一条完整序列（确定性: 同 seed 同结果）。

    split: 纹理池划分（防泄漏; train/val 共用 train 纹理池）。
    meta_split: 写入 meta.json 的真实数据划分; 缺省时回落到 split。
    use_gpu: True 时用 nvdiffrast GPU 光栅化（需单进程/少量进程共享 GPU）。
    """
    rng = np.random.default_rng(seed)
    t_start = time.time()

    # ---------------- 相机 ----------------
    if cfg.camera.intrinsics == "scared":
        cam = PinholeCamera.scared_like(cfg.camera.width, cfg.camera.height,
                                        cfg.camera.jitter, rng)
    else:
        cam = PinholeCamera.from_fov(cfg.camera.fov_deg, cfg.camera.width,
                                     cfg.camera.height)

    # ---------------- 组织场景: 程序化管道 或 真实器官几何(C3VD) ----------------
    from ..scene.organ import ORGAN_MESHES, OrganMeshScene, available_organs
    use_organ = False
    organ_name = None
    if cfg.tissue.kind == "organ":
        use_organ = True
    elif cfg.tissue.kind == "mixed":
        use_organ = rng.random() < cfg.tissue.organ_prob
    organs_avail = available_organs()
    if use_organ and not organs_avail:
        use_organ = False

    if use_organ:
        organ_name = organs_avail[int(rng.integers(0, len(organs_avail)))]
        mesh_path, pose_txt = ORGAN_MESHES[organ_name]
        scene = OrganMeshScene(
            mesh_path, rng,
            decimate_faces=cfg.tissue.organ_decimate,
            tex_scale=_sample_uniform(rng, cfg.tissue.organ_tex_scale_mm),
            pose_txt=pose_txt)
        tunnel = scene  # 统一接口别名
        scene_kind = "organ"
    else:
        tunnel = TissueTunnel(
            rng,
            length=_sample_uniform(rng, cfg.tissue.length_mm),
            radius_base=_sample_uniform(rng, cfg.tissue.radius_base_mm),
            fold_amplitude=_sample_uniform(rng, cfg.tissue.fold_amplitude),
            fold_count=int(rng.integers(*cfg.tissue.fold_count)),
            n_rings=cfg.tissue.n_rings,
            n_sector=cfg.tissue.n_sector,
            cap_end=cfg.tissue.cap_end,
        )
        scene_kind = "tunnel"

    # 纹理
    use_real = (cfg.appearance.texture_source == "real" and bank is not None
                and rng.random() < cfg.appearance.real_texture_prob)
    if use_real:
        texture = make_texture(rng, bank, split=split, size=texture_size,
                               hsv_jitter=cfg.appearance.hsv_jitter)
    else:
        from ..scene.texture import _procedural_texture
        texture = _procedural_texture(rng, texture_size)
    jit = cfg.appearance.tissue_material_jitter
    tissue_mat = Material(
        albedo=tuple(np.clip(np.array([0.92, 0.72, 0.66]) * rng.uniform(1 - jit, 1 + jit, 3), 0, 1)),
        ambient=TISSUE_MATERIAL.ambient * rng.uniform(1 - jit * 0.5, 1 + jit * 0.5),
        ks=TISSUE_MATERIAL.ks * rng.uniform(1 - jit, 1 + jit),
        shininess=TISSUE_MATERIAL.shininess * rng.uniform(0.7, 1.4),
        d_ref=tunnel.radius_base * rng.uniform(0.9, 1.5),
        atten_max=TISSUE_MATERIAL.atten_max,
    )

    # ---------------- 相机轨迹（场景坐标系, 渲染用原始位姿） ----------------
    n_frames = int(rng.integers(*cfg.trajectory.n_frames))
    n_frames = max(n_frames, 10)
    motion_type = cfg.trajectory.motion_types[rng.integers(0, len(cfg.trajectory.motion_types))]
    # 器官场景行程受限于器官尺度; 管道场景按步长推算
    if scene_kind == "organ":
        travel = rng.uniform(0.25, 0.6) * tunnel.arc[-1]
        start_arc = rng.uniform(0.15, max(0.6 - travel / max(tunnel.arc[-1], 1e-6), 0.2)) * tunnel.arc[-1]
    else:
        travel = None
        start_arc = None
    poses_wc = generate_trajectory(
        rng, tunnel, n_frames=n_frames, motion_type=motion_type,
        step_mm=_sample_uniform(rng, cfg.trajectory.step_mm),
        rot_deg=_sample_uniform(rng, cfg.trajectory.rot_deg),
        tremor_mm=_sample_uniform(rng, cfg.trajectory.tremor_mm) if rng.random() < cfg.trajectory.tremor_prob else 0.0,
        tremor_deg=_sample_uniform(rng, cfg.trajectory.tremor_deg) if rng.random() < cfg.trajectory.tremor_prob else 0.0,
        travel_mm=travel, start_arc=start_arc,
        normalize_first=False,
        keyframe_hop_mm=cfg.trajectory.keyframe_hop_mm,
    )
    # free轨迹的世界系路标默认在原点附近 —— 器官场景需平移进器官内
    if scene_kind == "organ" and motion_type == "free":
        anchor = tunnel.sample_axis_point(0.5 * tunnel.arc[-1])
        shift = anchor - poses_wc[n_frames // 2][:3, 3]
        poses_wc[:, :3, 3] += shift

    # ---------------- 动态物体（先于形变, 以便器械接触压陷接到尖端） ----------------
    n_lo, n_hi = int(cfg.objects.n_objects_range[0]), int(cfg.objects.n_objects_range[1])
    n_obj = 0 if n_hi <= 0 else sample_int(rng, (n_lo, n_hi))
    scene_objects: list[SceneObject] = []
    bop_ids = available_bop_ids()
    instrument_tip = None
    for oi in range(n_obj):
        if rng.random() < cfg.objects.use_bop_prob and bop_ids:
            obj_id = int(bop_ids[rng.integers(0, len(bop_ids))])
            scale = _sample_uniform(rng, cfg.objects.bop_scale)
            mesh, info = load_bop_object(obj_id, scale)
            mat, tex, name = BOP_MATERIAL, None, f"bop_{obj_id:02d}"
        else:
            mesh, info = make_instrument(rng)
            mat, tex, name = INSTRUMENT_MATERIAL, None, f"instrument_{oi}"
        target_wc = _object_target(tunnel, poses_wc, n_frames, rng)
        model = cfg.objects.motion_models[rng.integers(0, len(cfg.objects.motion_models))]
        approach = _approach_dir(poses_wc, n_frames, rng)
        poses_wo = object_motion(rng, model, n_frames, target_wc, approach)
        if name.startswith("instrument"):
            instrument_tip = poses_wo[n_frames // 2][:3, 3].copy()
        scene_objects.append(SceneObject(name, mesh, mat, tex, poses_wo, info))

    # ---------------- 贴壁棋盘格标记（显式参照物） ----------------
    has_marker = rng.random() < cfg.objects.marker_prob
    if has_marker:
        m_mesh, m_tex, m_info = make_marker_patch(
            size_mm=rng.uniform(10.0, 18.0), n_cells=int(rng.integers(4, 8)))
        T_mark = _marker_on_wall(tunnel, rng)
        scene_objects.append(SceneObject(
            "marker", m_mesh, MARKER_MATERIAL, m_tex,
            np.repeat(T_mark[None], n_frames, axis=0), m_info, is_marker=True))

    # ---------------- 形变（内窥镜默认非刚性; 器械尖端可触发接触压陷） ----------------
    deform = None
    if rng.random() < cfg.deformation.prob:
        if scene_kind == "organ":
            from ..motion.deformation import MeshHarmonicDeformation
            deform = MeshHarmonicDeformation(
                tunnel.mesh.verts, rng,
                strength=_sample_uniform(rng, cfg.deformation.strength),
                fps=cfg.fps, ref_radius=tunnel.radius_base)
            deform.set_reference(tunnel.mesh.verts, tunnel.mesh.normals)
        else:
            deform = TissueDeformation(
                tunnel, rng,
                strength=_sample_uniform(rng, cfg.deformation.strength),
                fps=cfg.fps,
                contact_point=instrument_tip,
            )

    # 序列级外观参照 (同一张真实图, 保证时序颜色一致; 路径访问不拷贝)
    color_ref = None
    ct_strength = 0.0
    if (cfg.appearance.color_transfer_prob > 0 and bank is not None
            and rng.random() < cfg.appearance.color_transfer_prob):
        from ..render.appearance import pick_ref_image
        color_ref = pick_ref_image(bank, split, rng)
        if color_ref is not None:
            ct_strength = _sample_uniform(rng, cfg.appearance.color_transfer_strength)
    vignette = (_sample_uniform(rng, cfg.appearance.vignette_strength)
                if rng.random() < cfg.appearance.vignette_prob else 0.0)
    barrel_k1 = (_sample_uniform(rng, cfg.appearance.barrel_k1)
                 if rng.random() < cfg.appearance.barrel_prob else 0.0)

    # ---------------- 逐帧渲染 + 对应关系 ----------------
    ae_target = rng.uniform(0.20, 0.32)
    tissue_mesh = tunnel.mesh
    base_verts = tissue_mesh.verts.copy()
    base_faces = tissue_mesh.faces.copy()
    frames = []
    disp_cache = {}          # frame -> (N_v,3) 组织顶点位移
    n_tissue_verts = len(base_verts)

    def tissue_disp(i: int) -> np.ndarray:
        if i not in disp_cache:
            disp_cache[i] = (deform.displacement(i) if deform is not None
                             else np.zeros((n_tissue_verts, 3)))
        return disp_cache[i]

    # ---------------- GPU 渲染资源（可选, nvdiffrast） ----------------
    want_normals = bool(cfg.output.save_normals)
    stereo_b = float(cfg.output.stereo_baseline_mm or 0.0)
    T_lr = np.eye(4)
    T_lr[0, 3] = stereo_b                     # 右目在左目相机系的位姿(沿+X基线)
    T_rl = np.linalg.inv(T_lr)
    gpu_ctx = None
    if use_gpu:
        try:
            import torch
            from ..render.gpu_rasterizer import (MeshGPU, buffers_to_numpy,
                                                 compute_normals_gpu,
                                                 get_glctx, new_buffers,
                                                 render_mesh_gpu,
                                                 shade_background_gpu)
            gpu_ctx = {
                "glctx": get_glctx(),
                # 此刻 tissue_mesh.verts 仍是 base_verts（形变在循环内逐帧叠加）
                "tissue": MeshGPU(tissue_mesh),
                "tissue_tex": (torch.as_tensor(texture, device="cuda").float() / 255.0
                               if texture is not None else None),
                "objs": [MeshGPU(o.mesh) for o in scene_objects],
                "obj_tex": [(torch.as_tensor(o.texture, device="cuda").float() / 255.0
                             if o.texture is not None else None)
                            for o in scene_objects],
            }
        except Exception as e:  # pragma: no cover - GPU 不可用时回退
            print(f"[warn] GPU 渲染初始化失败({type(e).__name__}: {e}), 回退 CPU")
            gpu_ctx = None

    def render_view(T_cw, frame_idx, disp, track_corr=True):
        """渲染单视图（左/右目通用）。返回 dict(color, depth, instance[, tri, bary, normal])。"""
        if gpu_ctx is not None:
            from ..render.gpu_rasterizer import (buffers_to_numpy,
                                                 compute_normals_gpu,
                                                 new_buffers, render_mesh_gpu,
                                                 shade_background_gpu)
            gbufs = new_buffers(cam.height, cam.width, track=track_corr,
                                track_normals=want_normals)
            gt = gpu_ctx["tissue"]
            v_t = gt.verts_with_disp(disp)
            render_mesh_gpu(gpu_ctx["glctx"], gbufs, gt, v_t,
                            compute_normals_gpu(v_t, gt.faces_long),
                            cam, T_cw, 1, tissue_mat,
                            gpu_ctx["tissue_tex"], near=cfg.near_mm)
            for oi, obj in enumerate(scene_objects):
                T_co = T_cw @ obj.poses_wo[frame_idx]
                gm = gpu_ctx["objs"][oi]
                render_mesh_gpu(gpu_ctx["glctx"], gbufs, gm,
                                gm.verts_with_disp(None), gm.static_normals,
                                cam, T_co, 2 + oi, obj.material,
                                gpu_ctx["obj_tex"][oi], near=cfg.near_mm)
            shade_background_gpu(gbufs)
            return buffers_to_numpy(gbufs)
        fbv = FrameBuffer(cam.width, cam.height,
                          track_correspondence=track_corr,
                          track_normals=want_normals)
        tissue_mesh.verts = base_verts + disp
        tissue_mesh.compute_normals()
        render_mesh(fbv, tissue_mesh, cam, T_cw, instance_id=1,
                    material=tissue_mat, texture=texture, near=cfg.near_mm)
        for oi, obj in enumerate(scene_objects):
            T_co = T_cw @ obj.poses_wo[frame_idx]
            render_mesh(fbv, obj.mesh, cam, T_co, instance_id=2 + oi,
                        material=obj.material, texture=obj.texture,
                        near=cfg.near_mm)
        shade_background(fbv)
        out = {"color": fbv.color, "depth": fbv.depth, "instance": fbv.instance}
        if track_corr:
            out["tri"], out["bary"] = fbv.tri, fbv.bary
        if want_normals:
            out["normal"] = fbv.normal
        return out

    for i in range(n_frames):
        T_cw = np.linalg.inv(poses_wc[i])
        d_i = tissue_disp(i)
        view = render_view(T_cw, i, d_i, track_corr=True)
        fb_color, fb_depth, fb_instance = view["color"], view["depth"], view["instance"]
        fb_tri, fb_bary = view["tri"], view["bary"]
        fb_normal = view.get("normal")
        bboxes = {}
        for oi, obj in enumerate(scene_objects):
            mask_i = (fb_instance == 2 + oi)
            if mask_i.any():
                ys, xs = np.nonzero(mask_i)
                bboxes[obj.name] = [int(xs.min()), int(ys.min()),
                                    int(xs.max()), int(ys.max())]
        # 自动曝光
        lum = fb_color.mean(axis=2)[np.isfinite(fb_depth)]
        if lum.size and np.median(lum) > 1e-3:
            gain = float(np.clip(ae_target / np.median(lum), 0.25, 4.0))
            fb_color = fb_color * gain
        if color_ref is not None:
            from ..render.appearance import reinhard_color_transfer
            fb_color = reinhard_color_transfer(fb_color, color_ref, ct_strength)
        if vignette > 0.0 or abs(barrel_k1) > 1e-8:
            from ..render.appearance import apply_optics
            fb_color = apply_optics(fb_color, cam, vignette=vignette, k1=barrel_k1)
        if i + 1 < n_frames:
            blur_dir = _center_flow(poses_wc[i], poses_wc[i + 1], cam)
        elif i > 0:
            blur_dir = _center_flow(poses_wc[i - 1], poses_wc[i], cam)
        else:
            blur_dir = np.zeros(2)
        color = apply_sensor(
            fb_color, rng,
            shot_noise=_sample_uniform(rng, cfg.sensor.shot_noise),
            read_noise=_sample_uniform(rng, cfg.sensor.read_noise),
            blur_px=_sample_uniform(rng, cfg.sensor.blur_px),
            exposure_jitter=_sample_uniform(rng, cfg.sensor.exposure_jitter),
            wb_jitter=cfg.sensor.wb_jitter,
            haze=(_sample_uniform(rng, cfg.sensor.haze_strength)
                  if rng.random() < cfg.sensor.haze_prob else 0.0),
            blur_dir=blur_dir,
        )
        frame_rec = {
            "color": color,
            "depth": fb_depth.copy(),
            "instance": fb_instance.copy(),
            "tri": fb_tri.copy(),
            "bary": fb_bary.copy(),
            "bboxes": bboxes,
        }
        if fb_normal is not None:
            frame_rec["normal"] = fb_normal.copy()
        # 右目（沿相机+X基线; 独立传感器噪声, 同一曝光策略）
        if stereo_b > 0.0:
            view_r = render_view(T_rl @ T_cw, i, d_i, track_corr=False)
            cr = view_r["color"]
            lum_r = cr.mean(axis=2)[np.isfinite(view_r["depth"])]
            if lum_r.size and np.median(lum_r) > 1e-3:
                cr = cr * float(np.clip(ae_target / np.median(lum_r), 0.25, 4.0))
            if color_ref is not None:
                from ..render.appearance import reinhard_color_transfer
                cr = reinhard_color_transfer(cr, color_ref, ct_strength)
            if vignette > 0.0 or abs(barrel_k1) > 1e-8:
                from ..render.appearance import apply_optics
                cr = apply_optics(cr, cam, vignette=vignette, k1=barrel_k1)
            frame_rec["color_right"] = apply_sensor(
                cr, rng,
                shot_noise=_sample_uniform(rng, cfg.sensor.shot_noise),
                read_noise=_sample_uniform(rng, cfg.sensor.read_noise),
                blur_px=_sample_uniform(rng, cfg.sensor.blur_px),
                exposure_jitter=_sample_uniform(rng, cfg.sensor.exposure_jitter),
                wb_jitter=cfg.sensor.wb_jitter,
                haze=(_sample_uniform(rng, cfg.sensor.haze_strength)
                      if rng.random() < cfg.sensor.haze_prob else 0.0),
                blur_dir=blur_dir,
            )
            frame_rec["depth_right"] = view_r["depth"].copy()
        frames.append(frame_rec)
    tissue_mesh.verts = base_verts

    # ---------------- 运动分解 + 稠密光流 GT（成对 t-1 -> t） ----------------
    motion_gt = _compute_motion_gt(
        frames, poses_wc, cam, base_verts, base_faces,
        tissue_disp, scene_objects, cfg)

    # ---------------- GT 汇总（导出前归一化到首帧相机系） ----------------
    poses_wc_raw = np.stack([ensure_se3(T) for T in poses_wc])  # 场景系(渲染帧)
    G = np.linalg.inv(poses_wc_raw[0])                          # 场景系 -> 首帧相机系
    poses_wc = np.einsum('ij,njk->nik', G, poses_wc_raw)        # 导出的相机绝对位姿
    obj_gt = []
    for obj in scene_objects:
        T_co_seq = np.stack([np.linalg.inv(p) @ q
                             for p, q in zip(poses_wc_raw, obj.poses_wo)])
        obj_gt.append({
            "name": obj.name, "info": obj.info, "is_marker": obj.is_marker,
            "poses_wo": np.einsum('ij,njk->nik', G, obj.poses_wo),  # 归一化系绝对位姿
            "poses_co": T_co_seq,          # 物体在相机系的位姿(相对运动GT, 全局变换不变)
        })

    ref_fracs = [m["ref_frac"] for m in motion_gt] + [None]
    meta = {
        "seq_id": seq_id, "seed": seed, "split": meta_split or split, "n_frames": n_frames,
        "motion_type": motion_type,
        "motion_stats": motion_stats(poses_wc),
        "scene_kind": scene_kind,
        "organ": organ_name if scene_kind == "organ" else None,
        "tunnel": tunnel.to_dict(),
        "deformation": deform.to_dict() if deform is not None else None,
        "texture_source": "real" if use_real else "procedural",
        "camera": cam.to_dict(),
        "fps": cfg.fps,
        "n_objects": len([o for o in scene_objects if not o.is_marker]),
        "has_marker": has_marker,
        "reference_fraction": ref_fracs,   # 帧 t-1->t 的参照物像素占比
        "appearance": {
            "color_transfer": color_ref is not None,
            "color_transfer_strength": ct_strength,
            "vignette": vignette, "barrel_k1": barrel_k1,
        },
        "motion_eps_mm": cfg.motion_eps_mm,
        "gen_time_sec": round(time.time() - t_start, 2),
        "config": cfg.to_dict(),
    }
    return SequenceData(
        seq_id=seq_id, n_frames=n_frames, camera=cam,
        poses_wc=poses_wc, objects=obj_gt, frames=frames,
        motion=motion_gt, meta=meta,
    )


# ---------------------------------------------------------------------------
# 辅助: 物体目标位/标记物贴壁
# ---------------------------------------------------------------------------

def _object_target(tunnel, poses_wc, n_frames, rng):
    mid = n_frames // 2
    T_mid = poses_wc[mid]
    target = T_mid[:3, 3] + T_mid[:3, 2] * rng.uniform(35, 75)
    return _project_into_tunnel(tunnel, target)


def _approach_dir(poses_wc, n_frames, rng):
    T_mid = poses_wc[n_frames // 2]
    d = -(T_mid[:3, 2]) + rng.normal(0, 0.15, 3)
    return d / np.linalg.norm(d)


def _marker_on_wall(scene, rng) -> np.ndarray:
    """在组织表面放一个标记（管道/器官通用）: 位置贴壁, 法线朝向管腔轴心。"""
    from ..geometry.se3 import look_at
    s = rng.uniform(0.15, 0.85) * scene.arc[-1]
    T = scene.axis_pose_at(s)
    c = T[:3, 3]
    verts = scene.mesh.verts
    fwd = T[:3, 2]
    proj = (verts - c) @ fwd
    d = np.linalg.norm(verts - c, axis=1)
    cand = np.flatnonzero((proj > 2.0) & (d < scene.radius_base * 2.0))
    if len(cand) < 5:
        cand = np.argsort(d)[:100]
    idx = int(cand[rng.integers(0, len(cand))])
    v = verts[idx]
    n = c - v
    n = n / max(np.linalg.norm(n), 1e-9)
    pos = v + n * 0.6  # 略微悬离壁面
    up = T[:3, 1]
    return look_at(pos, c, up)  # +z 朝向轴心


def _center_flow(T_wc_a: np.ndarray, T_wc_b: np.ndarray, cam,
                 z_mm: float = 50.0) -> np.ndarray:
    """光心处深度 z 的材料点从帧 a 投到帧 b 的像素位移, 作方向模糊核。"""
    from ..geometry.se3 import relative
    T_ba = relative(T_wc_b, T_wc_a)  # cam-a 点 -> cam-b
    p1 = T_ba[:3, :3] @ np.array([0.0, 0.0, z_mm]) + T_ba[:3, 3]
    if p1[2] <= 1e-3:
        return np.zeros(2)
    uv1 = np.array([cam.fx * p1[0] / p1[2] + cam.cx,
                    cam.fy * p1[1] / p1[2] + cam.cy])
    return uv1 - np.array([cam.cx, cam.cy])


def _project_into_tunnel(tunnel, p: np.ndarray) -> np.ndarray:
    """把目标点拉回管腔内（贴壁内侧留 8mm 余量）。"""
    p = np.asarray(p, float)
    cl = tunnel.centerline
    i = int(np.argmin(np.linalg.norm(cl - p, axis=1)))
    i0, i1 = max(i - 1, 0), min(i + 1, len(cl) - 1)
    tan = cl[i1] - cl[i0]
    tan /= np.linalg.norm(tan)
    v = p - cl[i]
    radial = v - (v @ tan) * tan
    r = np.linalg.norm(radial)
    # centerline 采样数可大于 radius_axial（Catmull-Rom 分段采样），轴向超出时取最末环半径
    i_r = min(i, len(tunnel.radius_axial) - 1)
    r_max = max(tunnel.radius_axial[i_r] - 8.0, 4.0)
    if r > r_max:
        radial = radial / r * r_max
    return cl[i] + radial + (v @ tan) * tan


# ---------------------------------------------------------------------------
# 运动分解 + 光流 GT
# ---------------------------------------------------------------------------

def _compute_motion_gt(frames, poses_wc, cam, base_verts, base_faces,
                       tissue_disp, scene_objects, cfg):
    """对每对相邻帧 (t-1, t) 计算: 光流 / 运动分解掩码 / 参照物比例。

    材料点演化:
    - 组织: X(τ) = v_base + Σ b_i d_i(τ)   (三角形重心插值)
    - 物体: X(τ) = T_wo(τ) · p_local
    """
    from ..geometry.se3 import transform_points
    H, W = cam.height, cam.width
    eps = cfg.motion_eps_mm
    n = len(frames)
    out = []

    for t in range(1, n):
        f0, f1 = frames[t - 1], frames[t]
        tri = f0["tri"]
        bary = f0["bary"]
        inst = f0["instance"]
        valid = tri >= 0
        ys, xs = np.nonzero(valid)
        if len(ys) == 0:
            out.append({"flow": None, "mask": np.zeros((H, W), np.uint8),
                        "ref_frac": 0.0})
            continue
        b0 = bary[ys, xs, 0]
        b1 = bary[ys, xs, 1]
        b2 = 1.0 - b0 - b1
        ids = tri[ys, xs]
        insts = inst[ys, xs]

        # ---- 材料点世界位置: t-1 与 t ----
        X_prev = np.zeros((len(ys), 3))
        X_cur = np.zeros((len(ys), 3))
        is_tissue = insts == 1
        if is_tissue.any():
            F_idx = base_faces[ids[is_tissue]]
            w0, w1, w2 = b0[is_tissue], b1[is_tissue], b2[is_tissue]
            v_base = (w0[:, None] * base_verts[F_idx[:, 0]]
                      + w1[:, None] * base_verts[F_idx[:, 1]]
                      + w2[:, None] * base_verts[F_idx[:, 2]])
            d_prev = tissue_disp(t - 1)
            d_cur = tissue_disp(t)
            disp_prev = (w0[:, None] * d_prev[F_idx[:, 0]]
                         + w1[:, None] * d_prev[F_idx[:, 1]]
                         + w2[:, None] * d_prev[F_idx[:, 2]])
            disp_cur = (w0[:, None] * d_cur[F_idx[:, 0]]
                        + w1[:, None] * d_cur[F_idx[:, 1]]
                        + w2[:, None] * d_cur[F_idx[:, 2]])
            X_prev[is_tissue] = v_base + disp_prev
            X_cur[is_tissue] = v_base + disp_cur
        # ---- 物体像素（逐物体, 刚体演化） ----
        for oi, obj in enumerate(scene_objects):
            m = insts == 2 + oi
            if not m.any():
                continue
            verts = obj.mesh.verts
            F_idx = obj.mesh.faces[ids[m]]
            w0, w1, w2 = b0[m], b1[m], b2[m]
            p_local = (w0[:, None] * verts[F_idx[:, 0]]
                       + w1[:, None] * verts[F_idx[:, 1]]
                       + w2[:, None] * verts[F_idx[:, 2]])
            X_prev[m] = transform_points(obj.poses_wo[t - 1], p_local)
            X_cur[m] = transform_points(obj.poses_wo[t], p_local)

        # ---- 场景运动幅度 ----
        scene_motion = np.linalg.norm(X_cur - X_prev, axis=1)

        # ---- 光流: X(t) 投影到相机 t ----
        T_cw = np.linalg.inv(poses_wc[t])
        Xc = X_cur @ T_cw[:3, :3].T + T_cw[:3, 3]
        z = Xc[:, 2]
        good_z = z > 1e-3
        uv1 = np.zeros((len(ys), 2))
        uv1[good_z, 0] = cam.fx * Xc[good_z, 0] / z[good_z] + cam.cx
        uv1[good_z, 1] = cam.fy * Xc[good_z, 1] / z[good_z] + cam.cy
        flow = uv1 - np.stack([xs + 0.5, ys + 0.5], axis=1)

        # 有效性: 在界内 + 未被遮挡(投影深度 ≈ 帧t渲染深度)
        depth1 = f1["depth"]
        xi = np.clip(np.round(uv1[:, 0] - 0.5).astype(int), 0, W - 1)
        yi = np.clip(np.round(uv1[:, 1] - 0.5).astype(int), 0, H - 1)
        inb = ((uv1[:, 0] >= 0.5) & (uv1[:, 0] < W - 0.5)
               & (uv1[:, 1] >= 0.5) & (uv1[:, 1] < H - 0.5) & good_z)
        d1 = depth1[yi, xi]
        unoccluded = inb & np.isfinite(d1) & (z <= d1 + 1.5)
        # 帧t中该像素处是否还是同一实例(实例变化≈遮挡边缘, 保守视为有效)
        valid_flow = unoccluded

        # ---- 运动分解掩码 ----
        # 光流无效像素仍保留分解标签(仅流为 NaN); 标签只由场景运动决定
        mask = np.zeros((H, W), np.uint8)
        labels_full = np.where(scene_motion < eps, 1, np.where(is_tissue, 2, 3))
        mask[ys, xs] = labels_full.astype(np.uint8)

        flow_full = np.full((H, W, 2), np.nan, dtype=np.float32)
        ok = valid_flow
        flow_full[ys[ok], xs[ok], 0] = flow[ok, 0]
        flow_full[ys[ok], xs[ok], 1] = flow[ok, 1]

        ref_frac = float((labels_full == 1).sum() / max(len(ys), 1))
        out.append({"flow": flow_full, "mask": mask, "ref_frac": ref_frac})
    return out
