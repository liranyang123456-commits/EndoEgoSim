"""Run official DROID-SLAM (monocular) on the 20 STIR 2024 clips in WSL2."""
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

LEDGER = "/mnt/d/ego_motiion_Camera/results/revision_vrih/r2_1_stir_real_ar.json"
STIR_ROOT = "/mnt/e/World_Agent_Enoscopy/datasets/STIR/STIRChallenge_2024"
OUT = "/mnt/d/ego_motiion_Camera/results/revision_vrih/droid_stir20"
os.makedirs(OUT, exist_ok=True)


def decode_mp4(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, im = cap.read()
        if not ok or im is None:
            break
        out.append(im)
    cap.release()
    return out


def run_clip(clip_id: str):
    coll, seq = clip_id.split("/")
    calib = json.loads(open(os.path.join(STIR_ROOT, coll, "calib.json")).read())
    K = np.asarray(calib["leftcameramat"], np.float64)
    mp4s = glob.glob(os.path.join(STIR_ROOT, coll, "left", seq, "frames", "*.mp4"))
    frames = decode_mp4(mp4s[0])
    # Feed up to 48 uniformly spaced frames (or all frames if <= 48) so DROID's warmup=8 & BA work smoothly
    sel = np.unique(np.linspace(0, len(frames) - 1, min(len(frames), 48)).round().astype(int))
    f_sub = [frames[i] for i in sel]
    stream = []
    for t, im in enumerate(f_sub):
        h0, w0 = im.shape[:2]
        sc = np.sqrt((384 * 512) / (h0 * w0))
        h1, w1 = int(h0 * sc) // 8 * 8, int(w0 * sc) // 8 * 8
        im_r = cv2.resize(im, (w1, h1), interpolation=cv2.INTER_AREA)
        intr_t = torch.as_tensor([K[0, 0] * w1 / w0, K[1, 1] * h1 / h0, K[0, 2] * w1 / w0, K[1, 2] * h1 / h0], dtype=torch.float32)
        stream.append((t, torch.as_tensor(im_r).permute(2, 0, 1)[None], intr_t))

    warmup = min(8, max(2, len(stream) // 3))
    args = argparse.Namespace(
        weights=DROID_CKPT, image_size=[h1, w1], buffer=128,
        stereo=False, disable_vis=True, upsample=False,
        beta=0.3, filter_thresh=1.5, warmup=warmup, keyframe_thresh=2.5,
        frontend_thresh=16.0, frontend_window=25, frontend_radius=2,
        frontend_nms=1, backend_thresh=22.0, backend_radius=2, backend_nms=3,
    )
    droid = Droid(args)
    for t, image, intrinsics in stream:
        droid.track(t, image, intrinsics=intrinsics)
    traj = np.asarray(droid.terminate(stream))
    poses = np.repeat(np.eye(4)[None], len(stream), 0)
    for i in range(min(len(traj), len(poses))):
        poses[i, :3, 3] = traj[i, :3]
        poses[i, :3, :3] = Rotation.from_quat(traj[i, 3:7]).as_matrix()
    del droid
    torch.cuda.empty_cache()
    return poses


def main():
    d = json.load(open(LEDGER))
    for row in d["clips"]:
        cid = row["clip"]
        tag = cid.replace("/", "_")
        out_p = os.path.join(OUT, f"{tag}_c2w.txt")
        if os.path.isfile(out_p):
            continue
        poses = run_clip(cid)
        np.savetxt(out_p, poses.reshape(len(poses), 16), fmt="%.8f")
        print(f"STIR DROID done: {cid} ({len(poses)} frames)", flush=True)


if __name__ == "__main__":
    main()
