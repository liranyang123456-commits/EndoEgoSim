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

STEREOMIS_ROOT = r"E:\World_Agent_Enoscopy\datasets\StereoMIS"
ENDOMAPPER_ROOT = r"E:\World_Agent_Enoscopy\datasets\EndoMapper"


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


def _to_local_path(path: str) -> str:
    """Map Windows drive paths to WSL /mnt/<drive>/ when the original is missing."""
    if path and os.path.exists(path):
        return path
    if path and len(path) >= 3 and path[1] == ":" and path[2] in "\\/":
        wsl = f"/mnt/{path[0].lower()}{path[2:].replace(chr(92), '/')}"
        if os.path.exists(wsl):
            return wsl
    return path


def _video_cache_dirs() -> list[str]:
    import tempfile
    out = [os.path.join(tempfile.gettempdir(), "endoego_frames")]
    win = os.environ.get("WIN_ENDOEGO_FRAMES")
    if win:
        out.append(win)
    for cand in (
            "/mnt/c/Users/lry/AppData/Local/Temp/endoego_frames",
            r"C:\Users\lry\AppData\Local\Temp\endoego_frames"):
        if os.path.isdir(cand) and cand not in out:
            out.append(cand)
    return out


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
    offset = int(spec.get("frame_offset", 0))
    n = int(spec["n_frames"])
    key = f"{seq_id}_{w}x{h}_off{offset}"
    for root in _video_cache_dirs():
        cache = os.path.join(root, key)
        last = os.path.join(cache, f"{n - 1:06d}.png")
        if os.path.exists(last):
            return [os.path.join(cache, f"{i:06d}.png") for i in range(n)]
    cache = os.path.join(_video_cache_dirs()[0], key)
    os.makedirs(cache, exist_ok=True)
    cap = cv2.VideoCapture(_to_local_path(spec["path"]))
    if offset > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
    layout = spec.get("layout", "scared_tb_left")
    i = 0
    while i < n:
        ok, fr = cap.read()
        if not ok:
            break
        if layout in ("scared_tb_left", "stereo_tb_left"):
            fr = scared_left_view(fr)
        fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(cache, f"{i:06d}.png"), fr)
        i += 1
    cap.release()
    return [os.path.join(cache, f"{k:06d}.png") for k in range(i)]


def materialize_video_indices(seq_dir: str, spec: dict, indices) -> list[str]:
    """Decode only the requested frame indices (accurate sequential read)."""
    import tempfile
    import cv2
    idxs = [int(i) for i in indices]
    if not idxs:
        return []
    seq_id = os.path.basename(seq_dir.rstrip("/\\"))
    w, h = int(spec.get("width", 640)), int(spec.get("height", 512))
    max_side = int(spec.get("decode_max_side", 640))
    if max(w, h) > max_side:
        s = max_side / float(max(w, h))
        w, h = int(round(w * s)), int(round(h * s))
    offset = int(spec.get("frame_offset", 0))
    key = f"{seq_id}_{w}x{h}_off{offset}_n{len(idxs)}_{idxs[0]}_{idxs[-1]}"
    for root in _video_cache_dirs():
        cache = os.path.join(root, key)
        out = [os.path.join(cache, f"{i:06d}.png") for i in idxs]
        if all(os.path.isfile(p) for p in out):
            return out
    # 全帧缓存已含所有帧时直接索引，避免为每种抽帧数重复解码整段视频
    full_key = f"{seq_id}_{w}x{h}_off{offset}"
    for root in _video_cache_dirs():
        cache = os.path.join(root, full_key)
        out = [os.path.join(cache, f"{i:06d}.png") for i in idxs]
        if all(os.path.isfile(p) for p in out):
            return out
    cache = os.path.join(_video_cache_dirs()[0], key)
    out = [os.path.join(cache, f"{i:06d}.png") for i in idxs]
    os.makedirs(cache, exist_ok=True)
    cap = cv2.VideoCapture(_to_local_path(spec["path"]))
    if offset > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
    layout = spec.get("layout", "scared_tb_left")
    need = set(idxs)
    hi = max(idxs)
    i = 0
    while i <= hi:
        if i in need:
            ok, fr = cap.read()
            if not ok:
                break
            if layout in ("scared_tb_left", "stereo_tb_left"):
                fr = scared_left_view(fr)
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(cache, f"{i:06d}.png"), fr)
        elif not cap.grab():
            break
        i += 1
    cap.release()
    return [p for p in out if os.path.isfile(p)]


def _load_freiburg_c2w(path: str) -> np.ndarray | None:
    """TUM/Freiburg: [ts] tx ty tz qx qy qz qw -> (N,4,4) c2w."""
    from scipy.spatial.transform import Rotation
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.replace(",", " ").split()]
            if len(vals) >= 8:
                tx, ty, tz, qx, qy, qz, qw = vals[1:8]
            elif len(vals) >= 7:
                tx, ty, tz, qx, qy, qz, qw = vals[:7]
            else:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = (tx, ty, tz)
            rows.append(T)
    return np.stack(rows) if rows else None


def _stereomis_start_stop(seq_dir: str, n: int) -> tuple[int, int]:
    lo, hi = 0, n
    for name in sorted(os.listdir(seq_dir)):
        if not name.lower().endswith(".csv"):
            continue
        p = os.path.join(seq_dir, name)
        try:
            raw = open(p, encoding="utf-8", errors="replace").read().strip().splitlines()
        except OSError:
            continue
        if not raw:
            continue
        for line in raw[1:] if any(c.isalpha() for c in raw[0]) else raw:
            nums = [int(float(x)) for x in line.replace(";", ",").split(",")
                    if x.strip() and _is_number(x)]
            if len(nums) >= 2:
                a, b = nums[0], nums[1]
                if 0 <= a < b:
                    return a, min(b, n)
            if len(nums) == 1 and 0 < nums[0] < n:
                hi = min(hi, nums[0])
    return lo, hi


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _stereomis_left_intrinsics(seq_dir: str) -> dict | None:
    """Left pinhole from official StereoMIS calib names (Hayoz et al.)."""
    cands = (
        "camcal.json", "camera_calibration.json",
        "StereoCalibration.ini", "endoscope_calibration.yaml",
    )
    path = None
    for name in cands:
        p = os.path.join(seq_dir, name)
        if os.path.isfile(p):
            path = p
            break
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        d = json.load(open(path, encoding="utf-8"))
        data = d.get("data", d)
        intr = data.get("intrinsics", data)
        if isinstance(intr, list) and intr:
            left = intr[0]
            fx, fy = left["f"][0], left["f"][1]
            cx, cy = left["c"][0], left["c"][1]
            w, h = int(data.get("width", 0)), int(data.get("height", 0))
        else:
            return None
    elif ext == ".ini":
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(path)
        sec = cfg["StereoLeft"]
        fx, fy = float(sec["fc_x"]), float(sec["fc_y"])
        cx, cy = float(sec["cc_x"]), float(sec["cc_y"])
        w, h = int(float(sec["res_x"])), int(float(sec["res_y"]))
    elif ext in (".yaml", ".yml"):
        import cv2
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        K = fs.getNode("M1").mat()
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        w = int(fs.getNode("Camera.width").real())
        h = int(fs.getNode("Camera.height").real())
        fs.release()
    else:
        return None
    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "width": int(w), "height": int(h)}


def discover_stereomis(root: str = STEREOMIS_ROOT) -> list[dict]:
    """Hayoz StereoMIS: stacked stereo mp4 + Freiburg GT + left calib.

    Pose is da Vinci FK (relative motion trusted; absolute has FK drift).
    Units auto-detected: span < 20 => metres, converted to mm.
    """
    if not os.path.isdir(root):
        return []
    out = []
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root):
        names = {n.lower() for n in filenames}
        if "groundtruth.txt" not in names:
            continue
        videos = [n for n in filenames if n.lower().endswith(".mp4")]
        if not videos:
            continue
        pose_p = os.path.join(dirpath, "groundtruth.txt")
        poses = _load_freiburg_c2w(pose_p)
        if poses is None or len(poses) < 8:
            continue
        extent = float(np.max(np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1)))
        if extent < 20.0:
            poses = poses.copy()
            poses[:, :3, 3] *= 1000.0
        G = np.linalg.inv(poses[0])
        poses = np.einsum("ij,njk->nik", G, poses)
        n = len(poses)
        lo, hi = _stereomis_start_stop(dirpath, n)
        poses = poses[lo:hi]
        n = len(poses)
        if n < 8:
            continue
        intr = _stereomis_left_intrinsics(dirpath)
        if intr is None:
            continue
        video = os.path.join(dirpath, videos[0])
        rel = os.path.relpath(dirpath, root).replace("\\", "/")
        sid = "stereomis_" + rel.replace("/", "_").replace("\\", "_")
        if sid in seen:
            continue
        seen.add(sid)
        out.append({
            "seq_id": sid, "source": "StereoMIS", "src_path": dirpath,
            "frame_paths": [], "poses": poses, "intrinsics": intr,
            "video": {
                "type": "video", "path": video, "n_frames": n,
                "width": int(intr["width"]), "height": int(intr["height"]),
                "layout": "stereo_tb_left", "decode_max_side": 640,
                "frame_offset": int(lo),
            },
        })
    out.sort(key=lambda r: r["seq_id"])
    return out


def discover_endomapper(root: str = ENDOMAPPER_ROOT) -> list[dict]:
    """Photorealistic EndoMapper sim sequences with trajectory.csv, if present.

    Real EndoMapper videos have no metric pose GT. Synapse syn26707219 is
    access-controlled; this returns [] until the user places the sim subset here.
    The local ENDOMAPPER_ROOT is missing on this machine (2026-08-27 probe).
    """
    if not os.path.isdir(root):
        return []
    out = []
    for dirpath, _, filenames in os.walk(root):
        if "trajectory.csv" not in {n.lower() for n in filenames}:
            continue
        pose_p = os.path.join(dirpath, "trajectory.csv")
        try:
            arr = np.loadtxt(pose_p, delimiter=",", skiprows=1)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[0] < 8:
            continue
        # leave unused until a licensed copy is on disk and the column layout is verified
        _ = arr
        _ = dirpath
    return out


def build_real_refs(out_root: str, include_c3vd: bool = True,
                    include_scared: bool = True,
                    include_stereomis: bool = False,
                    include_endomapper: bool = False) -> list[dict]:
    os.makedirs(out_root, exist_ok=True)
    recs = []
    if include_c3vd:
        recs.extend(discover_c3vd())
    if include_scared:
        recs.extend(discover_scared())
    if include_stereomis:
        recs.extend(discover_stereomis())
    if include_endomapper:
        recs.extend(discover_endomapper())
    results = []
    for rec in recs:
        results.append(write_ref_sequence(rec, os.path.join(out_root, rec["seq_id"])))
    index = {"n": len(results), "copied_pixels": False, "sequences": results}
    with open(os.path.join(out_root, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1, ensure_ascii=False)
    return results
