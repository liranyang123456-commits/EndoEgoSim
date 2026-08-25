"""完整 SCARED 转换器: E:\\World_Agent_Enoscopy\\datasets\\SCARED 的 dataset_1/2/3
各5个keyframe序列 -> EndoEgoSim 统一格式（真实锚定测试集, 15条序列）。

每 keyframe_N:
  data/rgb_frames/frame_XXXXXX.jpg   左目视频帧
  data/frame_data.tar.gz             每帧 camera-pose(4x4 c2w) + KL 内参
位姿单位: 米(m) -> 统一转 mm; 归一化到首帧相机系。

用法: python -m endosim.dataset.scared_full_converter --out .../real_test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import tarfile

import cv2
import numpy as np

SCARED_ROOT = r"E:\World_Agent_Enoscopy\datasets\SCARED"


def scared_left_view(img):
    """SCARED 上下立体堆叠 (约 1280x2048) -> 左目上半 (1280x1024)。

    旧转换把整幅 2048 高直接缩到 512, 左右目被压进同一张图, 几何完全破坏。
    """
    if img is None:
        return img
    h, w = img.shape[:2]
    if h >= int(w * 1.4):
        return img[: h // 2]
    return img


def convert_keyframe(kf_dir: str, out_dir: str, width: int = 640,
                     height: int = 512) -> dict | None:
    data_dir = os.path.join(kf_dir, "data")
    rgb_dir = os.path.join(data_dir, "rgb_frames")
    fd_path = os.path.join(data_dir, "frame_data.tar.gz")
    if not (os.path.isdir(rgb_dir) and os.path.exists(fd_path)):
        return None
    os.makedirs(os.path.join(out_dir, "color"), exist_ok=True)

    # 位姿+内参
    poses, K = [], None
    with tarfile.open(fd_path) as t:
        for name in sorted(t.getnames()):
            if not name.endswith(".json"):
                continue
            d = json.load(t.extractfile(name))
            T = np.asarray(d["camera-pose"], dtype=np.float64)
            T[:3, 3] *= 1000.0  # m -> mm
            poses.append(T)
            if K is None:
                K = np.asarray(d["camera-calibration"]["KL"], dtype=np.float64)
    poses = np.stack(poses)
    G = np.linalg.inv(poses[0])
    poses = np.einsum('ij,njk->nik', G, poses)

    frames = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".jpg"))
    n = min(len(frames), len(poses))
    sx, sy = width / 1280.0, height / 1024.0
    for i in range(n):
        img = cv2.imread(os.path.join(rgb_dir, frames[i]))
        if img is None:
            continue
        img = scared_left_view(img)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir, "color", f"{i:06d}.png"), img)

    with open(os.path.join(out_dir, "pose_c2w.txt"), "w") as f:
        for T in poses[:n]:
            f.write(" ".join(f"{v:.9f}" for v in T.reshape(-1)) + "\n")
    intr = {"fx": K[0, 0] * sx, "fy": K[1, 1] * sy, "cx": K[0, 2] * sx,
            "cy": K[1, 2] * sy, "width": width, "height": height}
    with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
        json.dump(intr, f, indent=2)
    meta = {"seq_id": os.path.basename(out_dir), "source": "SCARED (real)",
            "src_path": kf_dir, "n_frames": n,
            "camera": intr, "note": "da Vinci Xi 离体猪组织, 位姿GT来自机器人运动学"}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {"seq": os.path.basename(out_dir), "n_frames": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"D:\ego_motiion_Camera\sim_data\real_test")
    args = ap.parse_args()
    results = []
    for ds in sorted(glob.glob(os.path.join(SCARED_ROOT, "dataset_*"))):
        if not os.path.isdir(ds):
            continue
        for kf in sorted(glob.glob(os.path.join(ds, "keyframe_*"))):
            name = f"scared_{os.path.basename(ds)}_{os.path.basename(kf)}"
            r = convert_keyframe(kf, os.path.join(args.out, name))
            if r:
                results.append(r)
    print(f"转换完成: {len(results)} 条 SCARED 序列")
    for r in results:
        print(f"  {r['seq']}: {r['n_frames']}帧")


if __name__ == "__main__":
    main()
