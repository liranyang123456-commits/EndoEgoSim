"""SCARED 真实数据转换器: 把本地 SCARED d1_key1 转成 EndoEgoSim 统一格式。

用途: 真实域锚定测试集（zero-shot 验证）。
注意: 只通过路径引用原始数据, 输出仅包含轻量索引文件与位姿/内参,
图像不拷贝（用符号引用+按需读取）。

SCARED d1_key1 结构（本地勘察确认）:
  color/0000.png     1280x2048 (上下立体堆叠, 上半=左目)
  depth/0000.tiff    uint16 LZW-TIFF (本机 PIL 读会崩溃, 必须 cv2)
  pose.txt           每行16数, 行主序 4x4 c2w
内参: fx=1035.3081, fy=1035.0876, cx=596.9550, cy=520.4100 @1280x1024
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

SCARED_ROOT = r"D:\LumenGSLAM\dataset\SCARED\d1_key1"
SCARED_K = dict(fx=1035.3081, fy=1035.0876, cx=596.9550, cy=520.4100,
                width=1280, height=1024)


def load_scared_poses(path: str) -> np.ndarray:
    """pose.txt: 每行16数=行主序4x4 c2w。归一化到首帧, m -> mm。"""
    P = np.loadtxt(path).reshape(-1, 4, 4)
    P[:, :3, 3] *= 1000.0  # SCARED 位姿单位是米, 统一转 mm
    T0_inv = np.linalg.inv(P[0])
    return np.einsum('ij,njk->nik', T0_inv, P)


def convert(out_dir: str, width: int = 640, height: int = 512,
            copy_frames: bool = True) -> dict:
    """转换: 取左目(上半) -> 缩放 -> 写成统一格式。

    copy_frames=False 时只写位姿/内参/索引(图像按需从原路径读)。
    """
    os.makedirs(out_dir, exist_ok=True)
    color_out = os.path.join(out_dir, "color")
    depth_out = os.path.join(out_dir, "depth")
    if copy_frames:
        os.makedirs(color_out, exist_ok=True)
        os.makedirs(depth_out, exist_ok=True)

    poses = load_scared_poses(os.path.join(SCARED_ROOT, "pose.txt"))
    n = len(poses)
    sx, sy = width / SCARED_K["width"], height / SCARED_K["height"]
    intr = {"fx": SCARED_K["fx"] * sx, "fy": SCARED_K["fy"] * sy,
            "cx": SCARED_K["cx"] * sx, "cy": SCARED_K["cy"] * sy,
            "width": width, "height": height}

    files_color = sorted(os.listdir(os.path.join(SCARED_ROOT, "color")))
    files_depth = sorted(os.listdir(os.path.join(SCARED_ROOT, "depth")))
    assert len(files_color) == n and len(files_depth) == n, \
        f"帧数不一致: color={len(files_color)} depth={len(files_depth)} pose={n}"

    if copy_frames:
        for i, (fc, fd) in enumerate(zip(files_color, files_depth)):
            img = cv2.imread(os.path.join(SCARED_ROOT, "color", fc))
            H0 = img.shape[0] // 2
            left = img[:H0]
            left = cv2.resize(left, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(color_out, f"{i:06d}.png"), left)
            # 深度: LZW TIFF 只能用 cv2 读
            d = cv2.imread(os.path.join(SCARED_ROOT, "depth", fd),
                           cv2.IMREAD_UNCHANGED)
            d = d[:H0]
            d = cv2.resize(d, (width, height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(depth_dir := depth_out, f"{i:06d}.png"),
                        d.astype(np.uint16))

    with open(os.path.join(out_dir, "pose_c2w.txt"), "w") as f:
        for T in poses:
            f.write(" ".join(f"{v:.9f}" for v in T.reshape(-1)) + "\n")
    with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
        json.dump(intr, f, indent=2)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seq_id": "scared_d1_key1", "source": "SCARED (real)",
            "src_path": SCARED_ROOT, "n_frames": n,
            "note": "左目=上下堆叠上半; 深度原始编码未知(仅作参考), 位姿为c2w已归一化首帧",
            "camera": intr,
        }, f, indent=2, ensure_ascii=False)
    return {"n_frames": n, "out_dir": out_dir}


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else r"D:\ego_motiion_Camera\sim_data\real_test\scared_d1_key1"
    print(convert(out))
