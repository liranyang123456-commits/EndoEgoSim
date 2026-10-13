"""Real-to-sim sequences: real endoscopic surfaces rendered along new camera paths.

Each scene is the structured-light surface of a real SCARED keyframe (in vivo
porcine abdomen, da Vinci endoscope), textured with the real left image whose
baked specular highlights are inpainted. New camera paths (hovering, stop-and-go,
slow sweeps, approach/retreat) are rendered with respiratory bulk motion over part
of the surface, local deformation, moving and resting instruments and coaxial
lighting, giving photo-real frames with exact camera poses, depth, flow and
absolute (world-frame) motion of every pixel.

  python -m endosim.dataset.r2s_generator --out sim_data_r2s --n-per-scene 30
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import cv2
import numpy as np

from ..config import GenConfig
from ..geometry.se3 import rot_trans, so3_exp
from ..motion.deformation import MeshHarmonicDeformation, RespiratoryMotion, ScaledDeformation
from ..motion.object_motion import object_motion
from ..render.appearance import apply_optics
from ..render.camera import PinholeCamera
from ..render.rasterizer import Material, SimMesh
from ..render.sensor import apply_sensor
from ..scene.objects import INSTRUMENT_MATERIAL, make_instrument
from .generator import SceneObject, SequenceData, _compute_motion_gt
from .writer import write_sequence

SCARED_ROOT = os.environ.get("SCARED_ROOT", "datasets/SCARED")  # obtain SCARED from its provider
FPS = 10.0
W_OUT, H_OUT = 640, 512


def scared_keyframes(datasets=(5, 6, 7)) -> list[str]:
    out = []
    for ds in datasets:
        for pat in (f"dataset_{ds}/keyframe_*", f"dataset_{ds}/dataset_{ds}/keyframe_*"):
            out += sorted(glob.glob(os.path.join(SCARED_ROOT, pat)))
    return [k for k in out if os.path.exists(os.path.join(k, "left_depth_map.tiff"))]


def scene_name(kf: str) -> str:
    m = re.search(r"dataset_(\d+).*keyframe_(\d+)", kf.replace("\\", "/"))
    return f"scared_d{m.group(1)}_k{m.group(2)}"


def _left_intrinsics(kf: str) -> np.ndarray:
    fs = cv2.FileStorage(os.path.join(kf, "endoscope_calibration.yaml"), cv2.FILE_STORAGE_READ)
    K = fs.getNode("M1").mat()
    fs.release()
    return np.asarray(K, dtype=np.float64)


def _inpaint_highlights(rgb: np.ndarray) -> np.ndarray:
    """Remove baked specular highlights; rendering adds view-dependent ones."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 2] > 230) & (hsv[..., 1] < 70)).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return cv2.inpaint(rgb, mask * 255, 7, cv2.INPAINT_TELEA) if mask.any() else rgb


def load_surface(kf: str, grid: int = 4, min_valid: float = 0.15, pad: float = 0.15) -> dict | None:
    """Textured 2.5D surface of a keyframe in its left-camera frame (mm).

    The surface is extended by `pad` of the image size on every side (depth by edge
    extension, texture by reflection) so that moving cameras do not see its border.
    """
    xyz = cv2.imread(os.path.join(kf, "left_depth_map.tiff"), cv2.IMREAD_UNCHANGED)
    bgr = cv2.imread(os.path.join(kf, "Left_Image.png"))
    if xyz is None or bgr is None:
        return None
    Z = xyz[..., 0].astype(np.float32)                 # OpenCV channel order: Z, Y, X
    lab_p = os.path.join(ANATOMY_DIR, f"{scene_name(kf)}.png")
    lab = cv2.imread(lab_p, cv2.IMREAD_UNCHANGED) if os.path.exists(lab_p) else None
    return surface_from_depth(Z,bgr,_left_intrinsics(kf),scene_name(kf),kf,grid,min_valid,pad,lab)


def surface_from_depth(Z,bgr,K,name,src,grid=4,min_valid=0.15,pad=0.15,lab=None):
    valid = np.isfinite(Z) & (Z > 1.0)
    if valid.mean() < min_valid:
        return None
    H, W = Z.shape
    K = np.asarray(K, dtype=np.float64)
    h, w = H // grid, W // grid
    Zb = np.where(valid, Z, np.nan)[: h * grid, : w * grid].reshape(h, grid, w, grid)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            Zg = np.nanmedian(Zb, axis=(1, 3)).astype(np.float32)
    hole = ~np.isfinite(Zg)
    lo, hi = np.nanpercentile(Zg, 2), np.nanpercentile(Zg, 98)
    Zf = np.where(hole, 0, Zg).astype(np.float32)
    Zf = cv2.inpaint(Zf, hole.astype(np.uint8), 9, cv2.INPAINT_TELEA)
    Zs = cv2.GaussianBlur(Zf, (0, 0), 3.0)
    Zg = np.where(hole, Zs, cv2.GaussianBlur(Zf, (0, 0), 0.8))
    Zg = np.clip(Zg, 0.7 * lo, 1.3 * hi)
    ph, pw_ = int(round(pad * h)), int(round(pad * w))
    Zg = cv2.copyMakeBorder(Zg, ph, ph, pw_, pw_, cv2.BORDER_REPLICATE)
    ring = np.ones_like(Zg, dtype=bool)
    ring[ph:ph + h, pw_:pw_ + w] = False
    Zg = np.where(ring, cv2.GaussianBlur(Zg, (0, 0), 4.0), Zg)
    h, w = Zg.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    jj, ii = np.meshgrid(np.arange(w) - pw_, np.arange(h) - ph)
    u = (jj + 0.5) * grid - 0.5
    v = (ii + 0.5) * grid - 0.5
    verts = np.stack([(u - cx) / fx * Zg, (v - cy) / fy * Zg, Zg], -1).reshape(-1, 3)
    W2, H2 = W + 2 * pw_ * grid, H + 2 * ph * grid
    uvs = np.stack([(u + pw_ * grid + 0.5) / W2, (v + ph * grid + 0.5) / H2], -1).reshape(-1, 2)
    idx = np.arange(h * w).reshape(h, w)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3),    # facing the camera
                            np.stack([b, c, d], -1).reshape(-1, 3)])
    zf = verts[faces, 2]
    faces = faces[(zf.max(1) / zf.min(1)) < 1.25]
    rgb = _inpaint_highlights(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    rgb = cv2.copyMakeBorder(rgb, ph * grid, ph * grid, pw_ * grid, pw_ * grid, cv2.BORDER_REFLECT_101)
    vcls = None
    if lab is not None:
        vu = np.clip(np.round(u), 0, W - 1).astype(int).reshape(-1)
        vv = np.clip(np.round(v), 0, H - 1).astype(int).reshape(-1)
        vcls = lab[vv, vu].astype(np.int64)
    return {"verts": verts.astype(np.float64), "faces": faces.astype(np.int64),
            "uvs": uvs.astype(np.float64), "texture": rgb,
            "K": K, "size": (W, H), "valid_frac": float(valid.mean()),
            "med_depth": float(np.median(Zg)), "name": name, "src": src,
            "grid_hw": (h, w), "vcls": vcls}


def hamlyn_keyframes(per_seq: int = 3, min_valid: float = 0.5) -> list[tuple]:
    """Return well-spaced (name,image,stereo-depth,K) tuples from Hamlyn rectified videos."""
    from ..lite.data import RealClips
    out=[]
    for name, d, frames, K, _ in RealClips(length=1).seqs:
        n=len(frames); picked=[]
        for c in np.linspace(0.08,0.92,max(per_seq*4,per_seq)):
            i=int(c*(n-1))
            p=frames[i]
            dp=os.path.splitext(p.replace(os.sep+'image01'+os.sep,os.sep+'depth01'+os.sep))[0]+'.png'
            z=cv2.imread(dp,cv2.IMREAD_UNCHANGED) if os.path.isfile(dp) else None
            if z is None or (z>0).mean()<min_valid: continue
            if picked and min(abs(i-j) for j,_ in picked)<0.15*n: continue
            picked.append((i,(p,dp)))
            if len(picked)>=per_seq: break
        out.extend([(f"hamlyn_{name[9:]}_f{i}",pair[0],pair[1],K) for i,pair in picked])
    return out


def load_surface_hamlyn(name,img_path,depth_path,K,grid=4,pad=0.15):
    bgr=cv2.imread(img_path)
    raw=cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)
    if bgr is None or raw is None: return None
    Z=cv2.medianBlur(raw.astype(np.float32),5); Z[raw==0]=np.nan
    # Crop rectification margins; crop intrinsics identically to preserve ray geometry.
    dark=(bgr.max(2)<=12).astype(np.float32); r0,c0=0,0; r1,c1=dark.shape
    while r1-r0>32 and c1-c0>32:
        side=[dark[r0,c0:c1].mean(),dark[r1-1,c0:c1].mean(),dark[r0:r1,c0].mean(),dark[r0:r1,c1-1].mean()]
        k=int(np.argmax(side))
        if side[k]<0.01: break
        if k==0:r0+=1
        elif k==1:r1-=1
        elif k==2:c0+=1
        else:c1-=1
    bgr=bgr[r0:r1,c0:c1]; Z=Z[r0:r1,c0:c1]
    K=np.asarray(K,dtype=np.float64).copy(); K[0,2]-=c0; K[1,2]-=r0
    grid=max(2,int(round(grid*bgr.shape[1]/1280)))
    return surface_from_depth(Z,bgr,K,name,img_path,grid,0.3,pad,None)


# CholecSeg8k classes grouped by how freely the tissue can move (anatomy prior)
ANATOMY_DIR = "results/anatomy_seg/scenes"
ANCHOR_CLS = (1, 6, 12)            # abdominal wall, connective tissue, liver ligament
LARGE_CLS = (2, 10, 11)            # liver, gallbladder, hepatic vein (diaphragm-driven)
INSTR_CLS = (5, 9)                 # instruments baked into the real texture


def anatomy_weights(surf: dict, rng) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-vertex respiration weight and local-deformation scale from anatomy classes."""
    c = surf["vcls"]
    no_static = rng.random() < 0.1                 # in vivo: sometimes nothing stays still
    f = {"anchor": rng.uniform(0.2, 0.5) if no_static else rng.uniform(0.0, 0.03),
         "large": rng.uniform(0.6, 1.0), "medium": rng.uniform(0.3, 0.8),
         "instr": rng.uniform(0.5, 1.0)}
    w = np.full(len(c), f["medium"])
    w[np.isin(c, ANCHOR_CLS)] = f["anchor"]
    w[np.isin(c, LARGE_CLS)] = f["large"]
    w[np.isin(c, INSTR_CLS)] = f["instr"]
    local = np.where(np.isin(c, ANCHOR_CLS), 0.5 if no_static else 0.0, 1.0)
    h, wd = surf["grid_hw"]
    w = cv2.GaussianBlur(w.reshape(h, wd).astype(np.float32), (0, 0), 1.5).reshape(-1)
    local = cv2.GaussianBlur(local.reshape(h, wd).astype(np.float32), (0, 0), 1.5).reshape(-1)
    f["no_static"] = bool(no_static)
    f["anchor_frac"] = float(np.isin(c, ANCHOR_CLS).mean())
    return w.astype(np.float64), local.astype(np.float64), f


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def camera_path(rng, n: int, kind: str, med_depth: float) -> np.ndarray:
    """c2w poses in the keyframe camera frame (original view = identity, +Z into the scene)."""
    lim_xy, lim_z = 0.15 * med_depth, (-0.30 * med_depth, 0.30 * med_depth)
    p0 = np.array([rng.uniform(-0.4, 0.4) * lim_xy, rng.uniform(-0.4, 0.4) * lim_xy,
                   rng.uniform(lim_z[0] * 0.5, lim_z[1] * 0.4)])
    a0 = np.deg2rad(rng.uniform(-4, 4, 2))
    pos = np.repeat(p0[None], n, 0)
    ang = np.repeat(a0[None], n, 0)
    t = np.arange(n)
    if kind == "hover":                                  # stop-and-go, 0-2 moves
        for _ in range(int(rng.integers(0, 3))):
            s, L = int(rng.integers(0, max(n - 6, 1))), int(rng.integers(5, 16))
            r = _smooth((t - s) / L)[:, None]
            pos = pos + r * rng.uniform(-1, 1, 3) * np.array([0.10, 0.10, 0.15]) * med_depth
            ang = ang + r * np.deg2rad(rng.uniform(-5, 5, 2))
    elif kind == "sweep":                                # slow pan/tilt about the view point
        amp = np.deg2rad(rng.uniform(3, 9, 2)) * rng.choice([-1, 1], 2)
        ph = rng.uniform(0, 2 * np.pi, 2)
        f = rng.uniform(0.3, 1.0, 2)
        ang = ang + amp * np.sin(2 * np.pi * f * t[:, None] / n + ph) - amp * np.sin(ph)
        pos = pos + np.outer(t / n, rng.uniform(-1, 1, 3) * 0.05 * med_depth)
    elif kind == "approach":                             # push in / pull back
        dz = rng.uniform(0.15, 0.3) * med_depth * rng.choice([-1, 1])
        s = _smooth(t / max(n - 1, 1) * rng.uniform(1.0, 1.6))[:, None]
        pos = pos + s * np.array([rng.uniform(-0.05, 0.05) * med_depth,
                                  rng.uniform(-0.05, 0.05) * med_depth, dz])
    else:                                                # wander: band-limited drift
        for k in range(3):
            freq = rng.uniform(0.5, 1.5)
            pos[:, k] += rng.uniform(0.03, 0.1) * med_depth * np.sin(
                2 * np.pi * freq * t / n + rng.uniform(0, 2 * np.pi))
        ang = ang + np.deg2rad(rng.uniform(-4, 4, 2)) * np.sin(
            2 * np.pi * rng.uniform(0.3, 1.0) * t[:, None] / n)
    trem = rng.uniform(0.0, 0.12)
    pos = pos + rng.normal(0, trem, (n, 3)) * 0.5 + 0.5 * np.cumsum(rng.normal(0, trem * 0.1, (n, 3)), 0)
    pos[:, :2] = np.clip(pos[:, :2], -lim_xy, lim_xy)
    pos[:, 2] = np.clip(pos[:, 2], *lim_z)
    return np.stack([rot_trans(so3_exp(np.array([ang[i, 0], ang[i, 1], 0.0])), pos[i])
                     for i in range(n)])


def _vertex_normals(verts, faces):
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)


def generate_r2s_sequence(surf: dict, seq_id: str, seed: int, split: str,
                          anatomy: bool = False) -> SequenceData:
    import torch
    from ..render.gpu_rasterizer import (MeshGPU, buffers_to_numpy, compute_normals_gpu,
                                         get_glctx, new_buffers, render_mesh_gpu,
                                         shade_background_gpu)
    t_start = time.time()
    rng = np.random.default_rng(seed)
    K = surf["K"]
    sx, sy = W_OUT / surf["size"][0], H_OUT / surf["size"][1]
    zoom = rng.uniform(1.1, 1.4)                          # narrower view keeps the border out
    cam = PinholeCamera(K[0, 0] * sx * zoom, K[1, 1] * sy * zoom, K[0, 2] * sx, K[1, 2] * sy,
                        W_OUT, H_OUT)
    n = int(rng.integers(50, 91))
    kind = rng.choice(["hover", "hover", "sweep", "approach", "wander"])
    md = surf["med_depth"]
    poses = camera_path(rng, n, str(kind), md)

    verts, faces = surf["verts"], surf["faces"]
    normals = _vertex_normals(verts, faces)
    tissue = SimMesh(verts, faces, normals=normals, uvs=surf["uvs"])

    deform = None
    anat = None
    if anatomy and surf.get("vcls") is not None:
        # tissue motion follows anatomy: anchors (abdominal wall, ligaments) stay still
        w_resp, local_scale, anat = anatomy_weights(surf, rng)
        local = MeshHarmonicDeformation(verts, rng, strength=rng.uniform(0.1, 0.6), fps=FPS,
                                        ref_radius=0.4 * md)
        local.set_reference(verts, normals)
        deform = RespiratoryMotion(verts, rng, fps=FPS, amp_mm=rng.uniform(2.0, 8.0),
                                   hz=rng.uniform(0.2, 0.4), static_frac=0.0,
                                   base=ScaledDeformation(local, local_scale), weights=w_resp)
    else:
        if rng.random() < 0.6:
            deform = MeshHarmonicDeformation(verts, rng, strength=rng.uniform(0.1, 0.6), fps=FPS,
                                             ref_radius=0.4 * md)
            deform.set_reference(verts, normals)
        if rng.random() < 0.75:
            deform = RespiratoryMotion(verts, rng, fps=FPS, amp_mm=rng.uniform(1.0, 6.0),
                                       hz=rng.uniform(0.2, 0.4), static_frac=rng.uniform(0.0, 0.7),
                                       base=deform)
    disp_cache = {}

    def tissue_disp(i):
        if i not in disp_cache:
            disp_cache[i] = deform.displacement(i) if deform is not None else np.zeros_like(verts)
        return disp_cache[i]

    # instruments: moving ones, plus resting ones that act as rigid background objects
    objects = []
    n_obj = int(rng.choice([0, 0, 1, 1, 2]))
    center_ids = np.argsort(np.linalg.norm(verts[:, :2] / verts[:, 2:3], axis=1))[:2000]
    for oi in range(n_obj):
        mesh, info = make_instrument(rng)
        target = verts[rng.choice(center_ids)] - np.array([0, 0, rng.uniform(3.0, 10.0)])
        model = rng.choice(["insertion", "manipulation", "free", "static", "static"])
        approach = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), -rng.uniform(0.3, 1.0)])
        approach /= np.linalg.norm(approach)
        if model == "static":
            ins = object_motion(rng, "insertion", n, target, approach, work_radius=0.15 * md)
            pw = np.repeat(ins[n // 2][None], n, 0)      # resting at the working position
        else:
            pw = object_motion(rng, str(model), n, target, approach, work_radius=0.15 * md)
        info = dict(info, motion_model=str(model))
        objects.append(SceneObject(f"instrument_{oi}", mesh, INSTRUMENT_MATERIAL, None, pw, info))

    mat = Material(albedo=(1, 1, 1), ambient=0.7, ks=rng.uniform(0.3, 0.9),
                   shininess=rng.uniform(25, 70), d_ref=0.5 * md, atten_max=3.0)
    face_cls = None
    if anat is not None:
        vc = surf["vcls"]
        face_cls = np.where(vc[faces[:, 1]] == vc[faces[:, 2]], vc[faces[:, 1]], vc[faces[:, 0]])
    glctx = get_glctx()
    tmesh = MeshGPU(tissue)
    tex = torch.as_tensor(surf["texture"], device="cuda").float() / 255.0
    omeshes = [MeshGPU(o.mesh) for o in objects]
    vignette = rng.uniform(0.0, 0.35)
    # sequence-level colour variation (SCARED is paler than e.g. StereoMIS)
    sat, gamma = rng.uniform(0.9, 1.5), rng.uniform(0.8, 1.2)
    gains = np.array([rng.uniform(0.95, 1.15), rng.uniform(0.88, 1.0), rng.uniform(0.85, 1.0)])
    frames = []
    for i in range(n):
        T_cw = np.linalg.inv(poses[i])
        bufs = new_buffers(H_OUT, W_OUT, track=True)
        v_t = tmesh.verts_with_disp(tissue_disp(i))
        render_mesh_gpu(glctx, bufs, tmesh, v_t, compute_normals_gpu(v_t, tmesh.faces_long),
                        cam, T_cw, 1, mat, tex, near=2.0)
        for oi, o in enumerate(objects):
            gm = omeshes[oi]
            render_mesh_gpu(glctx, bufs, gm, gm.verts_with_disp(None), gm.static_normals,
                            cam, T_cw @ o.poses_wo[i], 2 + oi, o.material, None, near=2.0)
        shade_background_gpu(bufs)
        fb = buffers_to_numpy(bufs)
        color = fb["color"]
        lum = color.mean(2)[np.isfinite(fb["depth"])]
        if lum.size and np.median(lum) > 1e-3:
            color = color * float(np.clip(0.42 / np.median(lum), 0.4, 2.5))
        gray = color.mean(2, keepdims=True)
        color = np.clip(gray + sat * (color - gray), 0, 1) ** gamma * gains
        if vignette > 0:
            color = apply_optics(color, cam, vignette=vignette, k1=0.0)
        color = apply_sensor(color, rng, shot_noise=rng.uniform(0.1, 0.5),
                             read_noise=rng.uniform(0.001, 0.004), blur_px=rng.uniform(0.0, 0.8),
                             exposure_jitter=0.05, wb_jitter=0.02)
        bboxes = {}
        for oi, o in enumerate(objects):
            m = fb["instance"] == 2 + oi
            if m.any():
                ys, xs = np.nonzero(m)
                bboxes[o.name] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        rec = {"color": color, "depth": fb["depth"], "instance": fb["instance"],
               "tri": fb["tri"], "bary": fb["bary"], "bboxes": bboxes}
        if face_cls is not None:
            # per-pixel anatomy class (CholecSeg8k ids); 13 = rendered instrument, 255 = none
            tri = fb["tri"]
            amap = np.full(tri.shape, 255, np.uint8)
            tis = (fb["instance"] == 1) & (tri >= 0)
            amap[tis] = face_cls[tri[tis]].astype(np.uint8)
            amap[fb["instance"] >= 2] = 13
            rec["anatomy"] = amap
        frames.append(rec)

    cfg = GenConfig()
    cfg.fps = FPS
    cfg.output.save_abs_motion = True
    cfg.output.save_flow = rng.random() < 0.25
    motion = _compute_motion_gt(frames, poses, cam, verts, faces, tissue_disp, objects, cfg)
    G = np.linalg.inv(poses[0])
    poses_out = np.einsum("ij,njk->nik", G, poses)
    obj_gt = [{"name": o.name, "info": o.info, "is_marker": False,
               "poses_wo": np.einsum("ij,njk->nik", G, o.poses_wo),
               "poses_co": np.stack([np.linalg.inv(p) @ q for p, q in zip(poses, o.poses_wo)])}
              for o in objects]
    meta = {
        "seq_id": seq_id, "seed": seed, "split": split, "n_frames": n,
        "motion_type": str(kind), "scene_kind": "real_" + surf["name"].split("_")[0], "scene": surf["name"],
        "source": surf["src"], "deformation": deform.to_dict() if deform is not None else None,
        "anatomy_motion": anat,
        "camera": cam.to_dict(), "fps": FPS, "n_objects": len(objects),
        "objects": [o.info for o in objects], "has_marker": False,
        "reference_fraction": [m["ref_frac"] for m in motion] + [None],
        "abs_reference_fraction": [m.get("abs_ref_frac") for m in motion] + [None],
        "appearance": {"real_texture": True, "vignette": vignette, "barrel_k1": 0.0,
                       "surface_valid_frac": surf["valid_frac"], "zoom": zoom,
                       "saturation": sat, "gamma": gamma, "gains": gains.tolist()},
        "motion_eps_mm": cfg.motion_eps_mm, "gen_time_sec": round(time.time() - t_start, 2),
        "config": cfg.to_dict(),
    }
    return SequenceData(seq_id=seq_id, n_frames=n, camera=cam, poses_wc=poses_out,
                        objects=obj_gt, frames=frames, motion=motion, meta=meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sim_data_r2s")
    ap.add_argument("--datasets", default="5,6,7")
    ap.add_argument("--n-per-scene", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=2000000)
    ap.add_argument("--val-scenes", default="scared_d7_k4", help="comma-separated scene names")
    ap.add_argument("--test-scenes", default="scared_d7_k5")
    ap.add_argument("--shard", default="", help="i/n scene shard; deterministic seed offsets")
    ap.add_argument("--source", default="scared", choices=["scared", "hamlyn"])
    ap.add_argument("--per-seq", type=int, default=4)
    ap.add_argument("--anatomy", action="store_true", help="apply anatomy-group motion labels where available")
    args = ap.parse_args()
    if args.source == "hamlyn":
        kfs = hamlyn_keyframes(args.per_seq, min_valid=0.45)
        print(f"{len(kfs)} Hamlyn surface scenes", flush=True)
    else:
        kfs = scared_keyframes(tuple(int(x) for x in args.datasets.split(",")))
    shard_i, shard_n = (int(x) for x in args.shard.split("/")) if args.shard else (0,1)
    val, test = set(args.val_scenes.split(",")), set(args.test_scenes.split(","))
    index = []
    seed = args.seed_start
    for si,kf in enumerate(kfs):
        if si % shard_n != shard_i: continue
        if args.shard: seed=args.seed_start+si*args.n_per_scene
        surf = load_surface_hamlyn(*kf) if args.source == "hamlyn" else load_surface(kf)
        if surf is None:
            print(f"skip {kf} (surface coverage too low)", flush=True)
            continue
        split = "val" if surf["name"] in val else "test" if surf["name"] in test else "train"
        for _ in range(args.n_per_scene):
            seq_id = f"seq_{seed:08d}"
            out_dir = os.path.join(args.out, split, seq_id)
            if os.path.exists(os.path.join(out_dir, "meta.json")):
                seed += 1
                continue
            t0 = time.time()
            seq = generate_r2s_sequence(surf, seq_id, seed, split, anatomy=args.anatomy)
            write_sequence(seq, out_dir)
            index.append({"seq_id": seq_id, "split": split, "scene": surf["name"]})
            print(f"{surf['name']} {seq_id} ({split}) {seq.meta['motion_type']} n={seq.n_frames} "
                  f"abs_ref={np.nanmean([x for x in seq.meta['abs_reference_fraction'] if x is not None]):.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            seed += 1
    os.makedirs(args.out, exist_ok=True)
    json.dump(index, open(os.path.join(args.out, f"index_r2s_{shard_i}of{shard_n}.json" if args.shard else "index_r2s.json"), "a", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
