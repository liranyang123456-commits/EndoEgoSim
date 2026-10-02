"""把 EGO_Mo 真实采集（pose_gt_fused）转成 MD-VGGT 评测格式。

每条 traj_*：cam0/images（左目帧）+ npz(R,p 逐帧位姿，标定板系，米)
  -> 输出 color/ + pose_c2w.txt（mm）+ intrinsics.json（去畸变后等效内参）。

用法：python scripts/convert_egomo_to_eval.py [--only NAME] [--max-frames N]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import cv2
import numpy as np

SRC = r"E:\EGO_Mo\datasets"
POSE_DIR = os.path.join(SRC, "pose_gt_fused")
CALIB = os.path.join(SRC, "calib_intrinsics_20260922_123120", "camera_calibration_cam0.json")
OUT = os.path.join(r"D:\ego_motiion_Camera", "sim_data", "egomo_real")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    calib = json.load(open(CALIB, encoding="utf-8"))
    K = np.array(calib["camera_matrix"], dtype=np.float64)
    dist = np.array(calib["distortion_coefficients"], dtype=np.float64)
    w0, h0 = calib["image_size"]
    # 去畸变后的等效内参
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w0, h0), 0)

    names = [f[:-4] for f in sorted(os.listdir(POSE_DIR)) if f.endswith(".npz")]
    if args.only:
        names = [n for n in names if n == args.only]
    os.makedirs(args.out, exist_ok=True)
    index = []
    for name in names:
        npz = np.load(os.path.join(POSE_DIR, name + ".npz"))
        R, p, usable = npz["R"], npz["p"], npz["usable"]
        img_dir = os.path.join(SRC, name, "cam0", "images")
        frames = sorted(os.listdir(img_dir))
        n = min(len(frames), len(R))
        if args.max_frames > 0:
            n = min(n, args.max_frames)
        seq_out = os.path.join(args.out, name)
        color_out = os.path.join(seq_out, "color")
        os.makedirs(color_out, exist_ok=True)
        poses = []
        for i in range(n):
            img = cv2.imread(os.path.join(img_dir, frames[i]))
            if img is None:
                continue
            img = cv2.undistort(img, K, dist, None, K_new)
            cv2.imwrite(os.path.join(color_out, f"{i:06d}.png"), img)
            Ti = np.eye(4)
            Ti[:3, :3] = R[i]
            Ti[:3, 3] = p[i] * 1000.0  # 米 -> mm
            poses.append(Ti)
        poses = np.stack(poses)
        np.savetxt(os.path.join(seq_out, "pose_c2w.txt"),
                   poses.reshape(len(poses), -1), fmt="%.6f")
        json.dump({
            "fx": float(K_new[0, 0]), "fy": float(K_new[1, 1]),
            "cx": float(K_new[0, 2]), "cy": float(K_new[1, 2]),
            "width": w0, "height": h0,
        }, open(os.path.join(seq_out, "intrinsics.json"), "w"), indent=1)
        json.dump({"n_frames": int(n), "usable_frac": float(usable[:n].mean()),
                   "source": "EGO_Mo pose_gt_fused", "seq": name},
                  open(os.path.join(seq_out, "meta.json"), "w"), indent=1)
        index.append({"seq": name, "n_frames": int(n),
                      "usable_frac": round(float(usable[:n].mean()), 3)})
        print(f"[convert] {name}: {n} frames, usable {usable[:n].mean():.2f}", flush=True)
    json.dump(index, open(os.path.join(args.out, "index.json"), "w"), indent=1)
    print(f"Wrote {len(index)} sequences to {args.out}")


if __name__ == "__main__":
    main()
