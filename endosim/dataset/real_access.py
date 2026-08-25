"""真实数据集路径索引: 只引用原始磁盘路径, 绝不拷贝像素。

产出轻量序列目录:
  sim_data/real_refs/<seq_id>/
    meta.json
    pose_c2w.txt          # 归一化到首帧, 单位 mm
    intrinsics.json
    color_index.json      # 有序列表, 每项是原始帧绝对路径
"""
from __future__ import annotations

import glob
import json
import os
import tarfile

import numpy as np

from .c3vd_converter import C3VD_INTRINSICS, C3VD_ROOT, load_c3vd_poses
from .scared_full_converter import SCARED_ROOT, scared_left_view


def _find_c3vd_color(seq_dir: str, i: int) -> str | None:
    for cand in (os.path.join(seq_dir, "rgb", f"{i:06d}.png"),
                 os.path.join(seq_dir, "rgb", f"{i:04d}.png"),
                 os.path.join(seq_dir, f"{i}_color.png"),
                 os.path.join(seq_dir, f"{i:04d}_color.png")):
        if os.path.exists(cand):
            return cand
    return None


def discover_c3vd(root: str = C3VD_ROOT) -> list[dict]:
    """扫描 C3VD, 返回可索引序列描述 (含全部原始帧路径)。"""
    seq_dirs = []
    for d in sorted(glob.glob(os.path.join(root, "*", "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "pose.txt")):
            name = os.path.basename(os.path.dirname(d)) + "_" + os.path.basename(d)
            if "mold" in name:
                continue
            seq_dirs.append((name, d))
    for d in sorted(glob.glob(os.path.join(root, "c[12]_*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "pose.txt")):
            seq_dirs.append((os.path.basename(d), d))
    out = []
    for name, d in seq_dirs:
        poses = load_c3vd_poses(os.path.join(d, "pose.txt"), normalize=True)
        paths, keep = [], []
        for i in range(len(poses)):
            p = _find_c3vd_color(d, i)
            if p is None:
                continue
            paths.append(p)
            keep.append(i)
        if len(keep) < 8:
            continue
        out.append({
            "seq_id": name, "source": "C3VD", "src_path": d,
            "frame_paths": paths, "poses": poses[keep],
            "intrinsics": dict(C3VD_INTRINSICS),
        })
    return out


def _load_scared_poses(kf_dir: str, official: bool = True) -> dict | None:
    """official=True: 左目 1280×1024 + 原生 KL + rgb.mp4 全帧（SurgCUT3R 口径）。"""
    data_dir = os.path.join(kf_dir, "data")
    rgb_dir = os.path.join(data_dir, "rgb_frames")
    fd_path = os.path.join(data_dir, "frame_data.tar.gz")
    video = os.path.join(data_dir, "rgb.mp4")
    if not os.path.exists(fd_path):
        return None
    if not (os.path.exists(video) or os.path.isdir(rgb_dir)):
        return None
    poses, K = [], None
    with tarfile.open(fd_path) as t:
        for name in sorted(t.getnames()):
            if not name.endswith(".json"):
                continue
            d = json.load(t.extractfile(name))
            T = np.asarray(d["camera-pose"], dtype=np.float64)
            T[:3, 3] *= 1000.0
            poses.append(T)
            if K is None:
                K = np.asarray(d["camera-calibration"]["KL"], dtype=np.float64)
    if not poses:
        return None
    poses = np.stack(poses)
    G = np.linalg.inv(poses[0])
    poses = np.einsum("ij,njk->nik", G, poses)
    n = len(poses)
    if official:
        w, h = 1280, 1024
        sx, sy = 1.0, 1.0
    else:
        w, h = 640, 512
        sx, sy = w / 1280.0, h / 1024.0
    intr = {"fx": float(K[0, 0] * sx), "fy": float(K[1, 1] * sy),
            "cx": float(K[0, 2] * sx), "cy": float(K[1, 2] * sy),
            "width": w, "height": h}
    rec = {"poses": poses[:n], "intrinsics": intr, "src_path": kf_dir,
           "frame_paths": []}
    if os.path.exists(video):
        rec["video"] = {
            "type": "video", "path": video, "n_frames": n,
            "width": w, "height": h, "layout": "scared_tb_left",
            "decode_max_side": 640,
        }
    else:
        frames = sorted(f for f in os.listdir(rgb_dir) if f.lower().endswith(".jpg"))
        n = min(len(frames), n)
        rec["frame_paths"] = [os.path.join(rgb_dir, frames[i]) for i in range(n)]
        rec["poses"] = poses[:n]
    return rec


def discover_scared(root: str = SCARED_ROOT) -> list[dict]:
    out = []
    for ds in sorted(glob.glob(os.path.join(root, "dataset_*"))):
        if not os.path.isdir(ds):
            continue
        for kf in sorted(glob.glob(os.path.join(ds, "keyframe_*"))):
            rec = _load_scared_poses(kf, official=True)
            if rec is None:
                continue
            rec["seq_id"] = f"scared_{os.path.basename(ds)}_{os.path.basename(kf)}"
            rec["source"] = "SCARED"
            out.append(rec)
    return out


def write_ref_sequence(rec: dict, out_dir: str) -> dict:
    """写轻量引用目录 (无像素拷贝)。"""
    os.makedirs(out_dir, exist_ok=True)
    poses = rec["poses"]
    paths = rec["frame_paths"]
    with open(os.path.join(out_dir, "pose_c2w.txt"), "w") as f:
        for T in poses:
            f.write(" ".join(f"{v:.9f}" for v in np.asarray(T).reshape(-1)) + "\n")
    with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
        json.dump(rec["intrinsics"], f, indent=2)
    color_index = rec.get("video") or paths
    n = len(poses) if rec.get("video") else len(paths)
    with open(os.path.join(out_dir, "color_index.json"), "w", encoding="utf-8") as f:
        json.dump(color_index, f, indent=1, ensure_ascii=False)
    meta = {
        "seq_id": rec["seq_id"], "source": rec["source"] + " (path-ref)",
        "src_path": rec["src_path"], "n_frames": n,
        "copied_pixels": False,
        "camera": rec["intrinsics"],
        "video": bool(rec.get("video")),
        "note": "帧像素保留在原始数据集路径, 本目录仅索引; SCARED 视频取左目上半",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {"seq": rec["seq_id"], "n_frames": n, "dir": out_dir,
            "video": bool(rec.get("video"))}


def materialize_video_frames(seq_dir: str, spec: dict) -> list[str]:
    """从原始 mp4 按需解码到系统临时目录 (不写入本仓库)。SCARED 取左目。"""
    import tempfile
    import cv2
    seq_id = os.path.basename(seq_dir.rstrip("/\\"))
    w, h = int(spec.get("width", 640)), int(spec.get("height", 512))
    max_side = int(spec.get("decode_max_side", 640))
    if max(w, h) > max_side:
        s = max_side / float(max(w, h))
        w, h = int(round(w * s)), int(round(h * s))
    cache = os.path.join(tempfile.gettempdir(), "endoego_frames", f"{seq_id}_{w}x{h}")
    n = int(spec["n_frames"])
    last = os.path.join(cache, f"{n - 1:06d}.png")
    if os.path.exists(last):
        return [os.path.join(cache, f"{i:06d}.png") for i in range(n)]
    os.makedirs(cache, exist_ok=True)
    cap = cv2.VideoCapture(spec["path"])
    layout = spec.get("layout", "scared_tb_left")
    i = 0
    while i < n:
        ok, fr = cap.read()
        if not ok:
            break
        if layout == "scared_tb_left":
            fr = scared_left_view(fr)
        fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(cache, f"{i:06d}.png"), fr)
        i += 1
    cap.release()
    return [os.path.join(cache, f"{k:06d}.png") for k in range(i)]


def build_real_refs(out_root: str, include_c3vd: bool = True,
                    include_scared: bool = True) -> list[dict]:
    os.makedirs(out_root, exist_ok=True)
    recs = []
    if include_c3vd:
        recs.extend(discover_c3vd())
    if include_scared:
        recs.extend(discover_scared())
    results = []
    for rec in recs:
        results.append(write_ref_sequence(rec, os.path.join(out_root, rec["seq_id"])))
    index = {"n": len(results), "copied_pixels": False, "sequences": results}
    with open(os.path.join(out_root, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1, ensure_ascii=False)
    return results
