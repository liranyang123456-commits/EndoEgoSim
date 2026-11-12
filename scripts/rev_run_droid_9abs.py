"""Run official DROID-SLAM (monocular) on the 9 EndoEgoSim-Abs AR test sequences in WSL2 with small buffer=128 (~0.9 GB VRAM)."""
import argparse, glob, json, os, sys
import cv2, numpy as np, torch
from scipy.spatial.transform import Rotation

DROID_ROOT = "/root/DROID-SLAM"
DROID_CKPT = "/mnt/d/DROID-SLAM/droid.pth"
if not os.path.isfile(DROID_CKPT):
    DROID_CKPT = "/root/DROID-SLAM/droid.pth"

sys.path.insert(0, os.path.join(DROID_ROOT, "droid_slam"))
lt_dir = os.path.join(DROID_ROOT, "thirdparty", "lietorch")
if os.path.isdir(lt_dir):
    sys.path.insert(0, lt_dir)

from droid import Droid  # noqa: E402

SEQS = [
    "seq_00900344", "seq_00900672", "seq_00900107", "seq_00900185",
    "seq_00900395", "seq_00900147", "seq_00900654", "seq_00900467", "seq_00900248",
]
BASE = "/mnt/d/ego_motiion_Camera/sim_data_abs/test"
OUT = "/mnt/d/ego_motiion_Camera/results/revision_vrih/droid_abs9"
os.makedirs(OUT, exist_ok=True)


def run_one(seq_dir: str):
    intr = json.load(open(os.path.join(seq_dir, "intrinsics.json")))
    K = np.array(intr["K"], float)
    paths = sorted(glob.glob(os.path.join(seq_dir, "color", "*.png")))
    stream = []
    for t, p in enumerate(paths):
        im = cv2.imread(p)
        h0, w0 = im.shape[:2]
        sc = np.sqrt((384 * 512) / (h0 * w0))
        h1, w1 = int(h0 * sc) // 8 * 8, int(w0 * sc) // 8 * 8
        im_r = cv2.resize(im, (w1, h1))
        intr_t = torch.as_tensor([K[0, 0] * w1 / w0, K[1, 1] * h1 / h0, K[0, 2] * w1 / w0, K[1, 2] * h1 / h0], dtype=torch.float32)
        stream.append((t, torch.as_tensor(im_r).permute(2, 0, 1)[None], intr_t))
    args = argparse.Namespace(
        weights=DROID_CKPT, image_size=[h1, w1], buffer=128,
        stereo=False, disable_vis=True, upsample=False,
        beta=0.3, filter_thresh=2.4, warmup=8, keyframe_thresh=4.0,
        frontend_thresh=16.0, frontend_window=25, frontend_radius=2,
        frontend_nms=1, backend_thresh=22.0, backend_radius=2, backend_nms=3,
    )
    droid = Droid(args)
    for t, image, intrinsics in stream:
        droid.track(t, image, intrinsics=intrinsics)
    traj = np.asarray(droid.terminate(stream))
    poses = np.repeat(np.eye(4)[None], len(paths), 0)
    for i in range(min(len(traj), len(poses))):
        poses[i, :3, 3] = traj[i, :3]
        # Exact lietorch [tx, ty, tz, qx, qy, qz, qw] convention!
        poses[i, :3, :3] = Rotation.from_quat(traj[i, 3:7]).as_matrix()
    del droid
    torch.cuda.empty_cache()
    return poses


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    for s in SEQS:
        out_p = os.path.join(OUT, f"{s}_c2w.txt")
        if os.path.isfile(out_p):
            continue
        ps = run_one(os.path.join(BASE, s))
        np.savetxt(out_p, ps.reshape(len(ps), 16), fmt="%.8f")
        print("DROID done:", s, len(ps), flush=True)
