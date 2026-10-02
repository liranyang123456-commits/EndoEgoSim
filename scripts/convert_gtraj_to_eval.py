"""把 EGO_Mo gtraj（全局 4K + 双目 + 双棋盘格）采集转成 MD-VGGT 评测格式。

fused_gt/segment_*_fused.csv：每帧各刚体全局位姿（四元数 + 平移，米，cam2_fixed 世界系）。
用左相机 C0 轨迹 + 左目图像。输出 color/ + pose_c2w.txt（mm）+ intrinsics.json。

用法：python scripts/convert_gtraj_to_eval.py [--session NAME] [--cam cam0] [--max-frames N]
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import cv2
import numpy as np

SRC = r"E:\EGO_Mo\datasets"
CALIB0 = os.path.join(SRC, "calib_intrinsics_20260922_123120", "camera_calibration_cam0.json")
OUT = os.path.join(r"D:\ego_motiion_Camera", "sim_data", "egomo_gtraj")


def quat_to_R(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="gtraj_20261002_002134")
    ap.add_argument("--cam", default="cam0", help="左目图像目录")
    ap.add_argument("--body", default="C0", help="轨迹刚体（C0=左相机）")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    calib = json.load(open(CALIB0, encoding="utf-8"))
    K = np.array(calib["camera_matrix"], dtype=np.float64)
    dist = np.array(calib["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    w0, h0 = calib["image_size"]
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w0, h0), 0)

    sess = os.path.join(SRC, args.session)
    fused_dir = os.path.join(sess, "fused_gt")
    img_dir = os.path.join(sess, args.cam, "images")
    times = [float(x) for x in open(os.path.join(sess, args.cam, "times.txt"), encoding="utf-8").read().split()]
    frames = sorted(os.listdir(img_dir))

    out_root = os.path.join(args.out, args.session)
    index = []
    for seg_csv in sorted(os.listdir(fused_dir)):
        if not seg_csv.endswith("_fused.csv"):
            continue
        seg = seg_csv.replace("_fused.csv", "")
        # 帧级时间戳来自 segments/segment_*.csv（30fps）；fused_gt 是 200Hz 位姿
        seg_frames_csv = os.path.join(sess, "segments", seg + ".csv")
        if not os.path.isfile(seg_frames_csv):
            continue
        with open(os.path.join(fused_dir, seg_csv), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        f_stamps = np.array([float(r["stamp"]) for r in rows])
        qw = np.array([float(r[f"{args.body}_qw"]) for r in rows])
        qx = np.array([float(r[f"{args.body}_qx"]) for r in rows])
        qy = np.array([float(r[f"{args.body}_qy"]) for r in rows])
        qz = np.array([float(r[f"{args.body}_qz"]) for r in rows])
        tx = np.array([float(r[f"{args.body}_tx"]) for r in rows])
        ty = np.array([float(r[f"{args.body}_ty"]) for r in rows])
        tz = np.array([float(r[f"{args.body}_tz"]) for r in rows])
        # 该段的帧（30fps）
        with open(seg_frames_csv, encoding="utf-8") as f:
            seg_rows = list(csv.DictReader(f))
        frame_stamps = np.array([float(r["stamp"]) for r in seg_rows])
        seq_out = os.path.join(out_root, seg)
        color_out = os.path.join(seq_out, "color")
        os.makedirs(color_out, exist_ok=True)
        poses = []
        n_written = 0
        for st in frame_stamps:
            # 位姿：fused_gt 最近时间戳
            i = int(np.argmin(np.abs(f_stamps - st)))
            # 图像：cam times 最近时间戳
            j = int(np.argmin(np.abs(np.array(times) - st)))
            if abs(times[j] - st) > 0.05:
                continue
            img = cv2.imread(os.path.join(img_dir, frames[j]))
            if img is None:
                continue
            img = cv2.undistort(img, K, dist, None, K_new)
            cv2.imwrite(os.path.join(color_out, f"{n_written:06d}.png"), img)
            Ti = np.eye(4)
            Ti[:3, :3] = quat_to_R(qw[i], qx[i], qy[i], qz[i])
            Ti[:3, 3] = np.array([tx[i], ty[i], tz[i]]) * 1000.0  # 米->mm
            poses.append(Ti)
            n_written += 1
            if args.max_frames > 0 and n_written >= args.max_frames:
                break
        if not poses:
            continue
        poses = np.stack(poses)
        np.savetxt(os.path.join(seq_out, "pose_c2w.txt"), poses.reshape(len(poses), -1), fmt="%.6f")
        json.dump({"fx": float(K_new[0, 0]), "fy": float(K_new[1, 1]),
                   "cx": float(K_new[0, 2]), "cy": float(K_new[1, 2]),
                   "width": w0, "height": h0},
                  open(os.path.join(seq_out, "intrinsics.json"), "w"), indent=1)
        c = poses[:, :3, 3]
        path_mm = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
        json.dump({"n_frames": int(n_written), "path_mm": round(path_mm, 1),
                   "source": f"{args.session}/{seg}", "body": args.body, "cam": args.cam},
                  open(os.path.join(seq_out, "meta.json"), "w"), indent=1)
        index.append({"seg": seg, "n_frames": int(n_written), "path_mm": round(path_mm, 1)})
        print(f"[convert] {seg}: {n_written} frames, path {path_mm:.0f} mm", flush=True)
    os.makedirs(out_root, exist_ok=True)
    json.dump(index, open(os.path.join(out_root, "index.json"), "w"), indent=1)
    print(f"Wrote {len(index)} segments to {out_root}")


if __name__ == "__main__":
    main()
