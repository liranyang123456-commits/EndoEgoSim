"""双刚体评估：相机轨迹 + 场景棋盘格位姿轨迹。

各方法估计左相机轨迹 T_W_C(t)；左目每帧检测场景棋盘格 GP050 并 solvePnP
得 T_C_B(t)；棋盘格绝对轨迹 T_W_B = T_W_C · T_C_B，与 fused_gt 的 B 轨迹对比。

用法：python scripts/eval_board_pose.py --session gtraj_20261002_002134 \
    --methods ours_v7_egomo_gtraj,reloc3r_egomo_gtraj
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from endosim.eval.metrics import align_trajectories  # noqa: E402

SRC = r"E:\EGO_Mo\datasets"
CALIB0 = os.path.join(SRC, "calib_intrinsics_20260922_123120", "camera_calibration_cam0.json")
# GP050 棋盘格：11x8 内角点，3mm 方格
INNER = (11, 8)
SQ = 3.0
OBJ = np.array([(i * SQ, j * SQ, 0.0) for j in range(INNER[1]) for i in range(INNER[0])],
               dtype=np.float32)


def quat_to_R(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def load_gt_board(session, seg):
    """fused_gt 的 B（场景棋盘格）轨迹，全局系，返回 (N,4,4) mm。"""
    p = os.path.join(SRC, session, "fused_gt", seg + "_fused.csv")
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        T = np.eye(4)
        T[:3, :3] = quat_to_R(float(r["B_qw"]), float(r["B_qx"]), float(r["B_qy"]), float(r["B_qz"]))
        T[:3, 3] = np.array([float(r["B_tx"]), float(r["B_ty"]), float(r["B_tz"])]) * 1000.0
        out.append(T)
    return np.stack(out), np.array([float(r["stamp"]) for r in rows])


def detect_board_poses(img_dir, K, frame_idx):
    """每帧检测棋盘格并 solvePnP，返回 {帧序: T_C_B}。"""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    out = {}
    for fi in frame_idx:
        img = cv2.imread(os.path.join(img_dir, f"{fi:06d}.png"))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, INNER, flags)
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        okp, rvec, tvec = cv2.solvePnP(OBJ, corners, K, None,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        if not okp:
            continue
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()  # mm（OBJ 是 mm）
        out[fi] = T
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="gtraj_20261002_002134")
    ap.add_argument("--methods", required=True)
    ap.add_argument("--cam", default="cam0")
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "sota", "board_pose_eval.json"))
    args = ap.parse_args()

    calib = json.load(open(CALIB0, encoding="utf-8"))
    K = np.array(calib["camera_matrix"], dtype=np.float64)
    dist = np.array(calib["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    w0, h0 = calib["image_size"]
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w0, h0), 0)

    sess = os.path.join(SRC, args.session)
    img_dir = os.path.join(sess, args.cam, "images")
    times = [float(x) for x in open(os.path.join(sess, args.cam, "times.txt"), encoding="utf-8").read().split()]

    conv_root = os.path.join(ROOT, "sim_data", "egomo_gtraj", args.session)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results = {}
    for seg in sorted(os.listdir(conv_root)):
        if not seg.startswith("segment_"):
            continue
        seg_dir = os.path.join(conv_root, seg)
        if not os.path.isdir(seg_dir):
            continue
        gt_board, gt_stamps = load_gt_board(args.session, seg)
        # 该段转换后的帧时间戳（与 GT 对齐）：转换时按 segments csv 的 stamp 写帧
        seg_csv = os.path.join(sess, "segments", seg + ".csv")
        with open(seg_csv, encoding="utf-8") as f:
            seg_rows = list(csv.DictReader(f))
        frame_stamps = np.array([float(r["stamp"]) for r in seg_rows])
        # 转换后第 i 帧对应原 cam 帧序
        frame_to_cam = []
        for st in frame_stamps:
            j = int(np.argmin(np.abs(np.array(times) - st)))
            frame_to_cam.append(j)
        for tag in methods:
            est_p = os.path.join(ROOT, "results", "sota", tag, f"{seg}_est_c2w.txt")
            if not os.path.isfile(est_p):
                continue
            est = np.loadtxt(est_p).reshape(-1, 4, 4)
            n = min(len(est), len(frame_stamps))
            # 棋盘格检测（用转换前的原图，去畸变已在转换时做——这里用转换后 color 帧 + K_new）
            conv_color = os.path.join(seg_dir, "color")
            board_rel = detect_board_poses(conv_color, K_new, list(range(n)))
            # 棋盘格绝对轨迹：T_W_B = T_W_C · T_C_B
            est_board = []
            gt_sel = []
            for i in range(n):
                if i not in board_rel:
                    continue
                T_W_B = est[i] @ board_rel[i]
                est_board.append(T_W_B)
                # GT：该帧 stamp 对应 fused_gt 最近
                k = int(np.argmin(np.abs(gt_stamps - frame_stamps[i])))
                gt_sel.append(gt_board[k])
            if len(est_board) < 5:
                continue
            est_board = np.stack(est_board)
            gt_sel = np.stack(gt_sel)
            _, (s, R, t) = align_trajectories(est_board, gt_sel, with_scale=True)
            ec = est_board[:, :3, 3]
            ec = (s * (ec @ R.T)) + t
            err = np.linalg.norm(ec - gt_sel[:, :3, 3], axis=1)
            results.setdefault(tag, {})[seg] = {
                "board_ate_mm": float(np.sqrt((err ** 2).mean())),
                "n": int(len(est_board)),
            }
            print(f"[{tag}] {seg}: board ATE {np.sqrt((err ** 2).mean()):.2f} mm (n={len(est_board)})", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w", encoding="utf-8"), indent=1)
    # 汇总
    for tag, segs in results.items():
        vals = [v["board_ate_mm"] for v in segs.values()]
        if vals:
            print(f"{tag}: board ATE mean {np.mean(vals):.2f} mm over {len(vals)} segs")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
