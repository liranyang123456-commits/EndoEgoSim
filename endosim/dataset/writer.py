"""序列落盘: TUM 兼容格式 + 自定义 GT 扩展。

目录结构:
seq_XXXXXX/
  color/000000.png            # RGB uint8
  depth/000000.png            # uint16 mm (或 .tiff)
  object_mask/000000.png      # uint8 实例ID (0背景 1组织 2..物体)
  flow/000000.npz             # 稠密前向光流 t->t+1 (float16 HxWx2, NaN=无效)
  motion_mask/000000.png      # 运动分解 t->t+1 (0无效 1静态参照 2形变组织 3运动物体)
  pose_c2w.txt                # 每行16数=行主序4x4 c2w (SCARED d1_key1 同款格式)
  groundtruth.txt             # TUM格式: t tx ty tz qx qy qz qw
  intrinsics.json
  object_poses.json           # 物体绝对/相机系位姿 + bbox + 标记物
  motion_gt.json              # 相对运动真值(索引化)
  meta.json                   # 全部仿真参数(含逐帧参照物比例)
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from ..geometry.se3 import quat_from_R
from .generator import SequenceData


def write_sequence(seq: SequenceData, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cfg_out = seq.meta["config"]["output"]
    color_dir = os.path.join(out_dir, "color")
    depth_dir = os.path.join(out_dir, "depth")
    mask_dir = os.path.join(out_dir, "object_mask")
    flow_dir = os.path.join(out_dir, "flow")
    mmask_dir = os.path.join(out_dir, "motion_mask")
    for d in (color_dir, depth_dir, mask_dir, flow_dir, mmask_dir):
        os.makedirs(d, exist_ok=True)

    n = seq.n_frames
    has_normals = bool(cfg_out.get("save_normals")) and seq.frames and "normal" in seq.frames[0]
    has_stereo = seq.frames and "color_right" in seq.frames[0]
    normal_dir = os.path.join(out_dir, "normal")
    color_r_dir = os.path.join(out_dir, "color_right")
    depth_r_dir = os.path.join(out_dir, "depth_right")
    if has_normals:
        os.makedirs(normal_dir, exist_ok=True)
    if has_stereo:
        os.makedirs(color_r_dir, exist_ok=True)
        os.makedirs(depth_r_dir, exist_ok=True)

    # ---- 帧数据 ----
    for i, fr in enumerate(seq.frames):
        if cfg_out["save_color"]:
            img = (np.clip(fr["color"], 0, 1)[..., ::-1] * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(color_dir, f"{i:06d}.png"), img)
        if cfg_out["save_depth"]:
            d = fr["depth"].copy()
            d[~np.isfinite(d)] = 0
            d16 = np.clip(np.round(d), 0, 65535).astype(np.uint16)
            if cfg_out["depth_format"] == "tiff":
                cv2.imwrite(os.path.join(depth_dir, f"{i:06d}.tiff"), d16)
            else:
                cv2.imwrite(os.path.join(depth_dir, f"{i:06d}.png"), d16)
        if cfg_out["save_mask"]:
            cv2.imwrite(os.path.join(mask_dir, f"{i:06d}.png"),
                        fr["instance"].astype(np.uint8))
        if has_normals:
            # 法线(相机系, [-1,1]) -> uint16 三通道: 0=无效, 其余=round((n+1)/2*65534)+1
            nn = fr["normal"]
            valid = np.linalg.norm(nn, axis=2) > 0.5
            enc = np.zeros(nn.shape, dtype=np.uint16)
            enc[valid] = (np.clip((nn[valid] + 1.0) * 0.5, 0, 1) * 65534 + 1).astype(np.uint16)
            cv2.imwrite(os.path.join(normal_dir, f"{i:06d}.png"), enc)
        if has_stereo:
            img_r = (np.clip(fr["color_right"], 0, 1)[..., ::-1] * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(color_r_dir, f"{i:06d}.png"), img_r)
            dr = fr["depth_right"].copy()
            dr[~np.isfinite(dr)] = 0
            cv2.imwrite(os.path.join(depth_r_dir, f"{i:06d}.png"),
                        np.clip(np.round(dr), 0, 65535).astype(np.uint16))

    # ---- 运动分解 + 光流 (帧对 t-1 -> t, 存于t) ----
    for t, m in enumerate(seq.motion):
        if cfg_out["save_flow"] and m["flow"] is not None:
            np.savez_compressed(os.path.join(flow_dir, f"{t:06d}.npz"),
                                flow=m["flow"].astype(np.float16))
        if cfg_out["save_motion_mask"]:
            cv2.imwrite(os.path.join(mmask_dir, f"{t:06d}.png"), m["mask"])

    # ---- 位姿 ----
    with open(os.path.join(out_dir, "pose_c2w.txt"), "w") as f:
        for T in seq.poses_wc:
            f.write(" ".join(f"{v:.9f}" for v in T.reshape(-1)) + "\n")
    with open(os.path.join(out_dir, "groundtruth.txt"), "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i, T in enumerate(seq.poses_wc):
            q = quat_from_R(T[:3, :3])
            t = i / seq.meta["fps"]
            f.write(f"{t:.6f} {T[0,3]:.9f} {T[1,3]:.9f} {T[2,3]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")

    # ---- 内参 ----
    intr = seq.camera.to_dict()
    if has_stereo:
        intr["stereo_baseline_mm"] = float(cfg_out.get("stereo_baseline_mm", 0.0))
    with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
        json.dump(intr, f, indent=2)

    # ---- 物体 GT ----
    if cfg_out["save_object_poses"]:
        obj_out = []
        for oi, obj in enumerate(seq.objects):
            obj_out.append({
                "name": obj["name"],
                "info": obj["info"],
                "is_marker": obj.get("is_marker", False),
                "poses_wo": [T.reshape(-1).tolist() for T in obj["poses_wo"]],
                "poses_co": [T.reshape(-1).tolist() for T in obj["poses_co"]],
                "bboxes": {str(i): seq.frames[i]["bboxes"].get(obj["name"])
                           for i in range(n)},
            })
        with open(os.path.join(out_dir, "object_poses.json"), "w") as f:
            json.dump(obj_out, f, indent=1)

    # ---- 相对运动 GT ----
    if cfg_out["save_motion_gt"]:
        rel = {}
        P = seq.poses_wc
        for gap in (1, 2, 5):
            g = []
            for i in range(n - gap):
                Tij = np.linalg.inv(P[i]) @ P[i + gap]
                g.append({"i": i, "j": i + gap, "T_ij": Tij.reshape(-1).tolist()})
            rel[f"gap_{gap}"] = g
        with open(os.path.join(out_dir, "motion_gt.json"), "w") as f:
            json.dump(rel, f)

    # ---- meta ----
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(seq.meta, f, indent=2, ensure_ascii=False, default=str)

    files = sum(len(fs) for _, _, fs in os.walk(out_dir))
    size_mb = sum(os.path.getsize(os.path.join(r, fn))
                  for r, _, fs in os.walk(out_dir) for fn in fs) / 1e6
    return {"seq_id": seq.seq_id, "n_frames": n, "files": files,
            "size_mb": round(size_mb, 1)}
