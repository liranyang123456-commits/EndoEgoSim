"""C3VD 真实数据转换器: 把本地 C3VD 序列转成 EndoEgoSim 统一格式（真实验证集）。

数据源（只读路径引用）:
- 完整GT序列(深度/光流/法线/遮挡): E:\\World_Agent_Enoscopy\\datasets\\C3VD\\cecum_t1_a
- RGB+位姿序列(12+条): E:\\World_Agent_Enoscopy\\datasets\\C3VD\\c1_* / c2_*

C3VD pose.txt 格式: 每行16个逗号分隔数, 行主序4x4、平移在末行
=> c2w = reshape(4,4).transpose()；单位 mm；归一化到首帧相机系。
C3VD 内参(官方): 675x540, fx≈401.16, fy≈400.94, cx≈334.14, cy≈273.87。

用法:
  python -m endosim.dataset.c3vd_converter --out D:/ego_motiion_Camera/sim_data/real_test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

C3VD_ROOT = r"E:\World_Agent_Enoscopy\datasets\C3VD"
C3VD_INTRINSICS = dict(fx=401.1595, fy=400.9425, cx=334.143, cy=273.8665,
                       width=675, height=540)


def load_c3vd_poses(path: str, normalize: bool = True) -> np.ndarray:
    """C3VD pose.txt -> (N,4,4) c2w（可选归一化到首帧）。"""
    P = np.loadtxt(path, delimiter=",")
    T = P.reshape(-1, 4, 4).transpose(0, 2, 1)  # 平移从末行移到末列
    if normalize:
        G = np.linalg.inv(T[0])
        T = np.einsum('ij,njk->nik', G, T)
    return T


def convert_sequence(seq_dir: str, out_dir: str, stride: int = 1,
                     with_modalities: bool = True) -> dict:
    """转换单条 C3VD 序列。stride: 帧采样步长（C3VD 30fps 步长0.2mm, 可加大步长）。"""
    os.makedirs(out_dir, exist_ok=True)
    color_dir = os.path.join(out_dir, "color")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    poses = load_c3vd_poses(os.path.join(seq_dir, "pose.txt"))
    n = len(poses)
    idx = list(range(0, n, stride))
    T = poses[idx]

    for k, i in enumerate(idx):
        # 颜色: c1_* 系列在 rgb/, 官方系列直接 *_color.png
        src = None
        for cand in (os.path.join(seq_dir, "rgb", f"{i:06d}.png"),
                     os.path.join(seq_dir, "rgb", f"{i:04d}.png"),
                     os.path.join(seq_dir, f"{i}_color.png"),
                     os.path.join(seq_dir, f"{i:04d}_color.png")):
            if os.path.exists(cand):
                src = cand
                break
        if src is None:
            continue
        img = cv2.imread(src)
        if img is not None:
            cv2.imwrite(os.path.join(color_dir, f"{k:06d}.png"), img)
        # 深度(如有): *_depth.tiff, 单位mm
        if with_modalities:
            for cand in (os.path.join(seq_dir, f"{i:04d}_depth.tiff"),
                         os.path.join(seq_dir, f"{i}_depth.tiff")):
                if os.path.exists(cand):
                    d = cv2.imread(cand, cv2.IMREAD_UNCHANGED)
                    if d is not None:
                        d16 = np.clip(np.round(d), 0, 65535).astype(np.uint16)
                        cv2.imwrite(os.path.join(depth_dir, f"{k:06d}.png"), d16)
                    break

    with open(os.path.join(out_dir, "pose_c2w.txt"), "w") as f:
        for Tk in T:
            f.write(" ".join(f"{v:.9f}" for v in Tk.reshape(-1)) + "\n")
    with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
        json.dump(C3VD_INTRINSICS, f, indent=2)
    # C3VD帧间步长极小(0.2mm), 采样后仍远小于仿真; meta记录原始fps
    meta = {"seq_id": os.path.basename(out_dir), "source": "C3VD (real)",
            "src_path": seq_dir, "n_frames": len(T), "stride": stride,
            "camera": C3VD_INTRINSICS, "note": "真实结肠镜+物理phantom, 2D-3D配准GT"}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {"seq": os.path.basename(out_dir), "n_frames": len(T)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"D:\ego_motiion_Camera\sim_data\real_test")
    ap.add_argument("--stride", type=int, default=5,
                    help="帧采样步长(C3VD原生步长0.2mm太小, 默认5=1mm/帧)")
    args = ap.parse_args()

    results = []
    # 官方完整序列(嵌套目录)
    for d in sorted(glob.glob(os.path.join(C3VD_ROOT, "*", "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "pose.txt")):
            name = os.path.basename(os.path.dirname(d)) + "_" + os.path.basename(d)
            # 跳过模具
            if "mold" in name:
                continue
            results.append(convert_sequence(d, os.path.join(args.out, name),
                                            stride=args.stride))
    # c1_/c2_ 系列(平铺)
    for d in sorted(glob.glob(os.path.join(C3VD_ROOT, "c[12]_*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "pose.txt")):
            results.append(convert_sequence(d, os.path.join(args.out, os.path.basename(d)),
                                            stride=args.stride))
    print(f"转换完成: {len(results)} 条真实C3VD序列")
    for r in results:
        print(f"  {r['seq']}: {r['n_frames']}帧")


if __name__ == "__main__":
    main()
