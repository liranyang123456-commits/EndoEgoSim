"""Evaluate official Endo-FASt3r (MICCAI 2025) pose model under EgoLite protocols.

Runs the released Reloc3rX pose network (Reloc3r-512 base + their pose.pth)
pairwise over our registered sequences, chains pair poses into a trajectory,
and reports the same Sim(3)/SE(3) ATE + RPE metrics as EgoLite evals.

Direction convention: Endo-FASt3r's own dump chains cam_to_world @ T_pair,
i.e. T_pair maps source frame k to target frame k+1 (SfMLearner convention).
Verified on a SCARED smoke run before full evaluation.

Example:
  python scripts/eval_endofast3r.py --list lists/stereomis.txt --tag endofast3r_stereomis_full --max-frames 0
  python scripts/eval_endofast3r.py --list lists/scared_official.txt --tag endofast3r_scared_full --max-frames 0
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "Endo_FASt3r"))

from endosim.eval.metrics import evaluate_trajectory, load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames, select_frame_indices  # noqa: E402

FAST3R = os.path.join(ROOT, "third_party", "Endo_FASt3r")
WEIGHTS = os.path.join(FAST3R, "weights", "best_weights", "pose.pth")
RELOC3R = os.path.join(FAST3R, "Reloc3r-512-fixed.pth")

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def resize_pil_image(img, long_edge_size):
    S = max(img.size)
    interp = Image.LANCZOS if S > long_edge_size else Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def prepare(path, device=None):
    img = Image.open(path).convert("RGB")
    img = resize_pil_image(img, 512)
    W, H = img.size
    cx, cy = W // 2, H // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if W == H:
        halfh = 3 * halfw / 4
    img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    x = ImgNorm(img)[None]
    return x.to(device) if device is not None else x


def load_model(device):
    import networks
    pose_model = networks.Reloc3rX(RELOC3R)
    sd = torch.load(WEIGHTS, map_location="cpu")
    msd = pose_model.state_dict()
    matched = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    missing = [k for k in msd if k not in matched]
    n_lora = sum(1 for k in matched if "lora" in k.lower())
    print(f"pose.pth matched {len(matched)}/{len(sd)} keys ({n_lora} lora/mora); model params missing from ckpt: {len(missing)}")
    assert n_lora > 0, "fine-tuned DoRA/MoRA weights not loaded - would evaluate plain Reloc3r!"
    pose_model.load_state_dict(matched, strict=False)
    return pose_model.to(device).eval()


@torch.no_grad()
def run_seq(model, d, device, batch, max_frames, invert):
    gt = load_pose_txt(os.path.join(d, "pose_c2w.txt"))
    idx = select_frame_indices(gt, "uniform", max_frames=max_frames) if max_frames else np.arange(len(gt))
    paths = list_color_frames(d, indices=idx)
    n = len(paths)
    Ts = []
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        stream = ex.map(prepare, paths)
        prev = next(stream).to(device)
        for st in range(0, n - 1, batch):
            n_chunk = min(batch, n - 1 - st)
            nxt = [next(stream).to(device, non_blocking=True) for _ in range(n_chunk)]
            a = torch.cat([prev] + nxt[:-1], 0)
            b = torch.cat(nxt, 0)
            pose2, _ = model({"img": a}, {"img": b})
            T = pose2["pose"].cpu().numpy().astype(np.float64)
            Ts.append(np.linalg.inv(T) if invert else T)
            prev = nxt[-1]
    Ts = np.concatenate(Ts, 0)
    dt = (time.time() - t0) / max(1, n - 1) * 1000
    c2w = [np.eye(4)]
    for T in Ts:
        c2w.append(c2w[-1] @ T)
    est = np.stack(c2w, 0)
    return est, gt[idx], dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/sota")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--invert", action="store_true", help="chain inv(T) instead (convention check)")
    args = ap.parse_args()

    device = torch.device("cuda")
    model = load_model(device)
    seqs = [s.strip() for s in open(args.list, encoding="utf-8") if s.strip()]
    if args.limit:
        seqs = seqs[: args.limit]
    outdir = os.path.join(args.out, args.tag)
    os.makedirs(outdir, exist_ok=True)

    records, ates, times = [], [], []
    for s in seqs:
        d = s if os.path.isabs(s) else os.path.join(ROOT, s)
        name = os.path.basename(d)
        rec_path = os.path.join(outdir, f"{name}_record.json")
        if os.path.exists(rec_path):
            rec = json.load(open(rec_path, encoding="utf-8"))
            records.append(rec)
            ates.append(rec["ate_sim3"]["rmse"])
            times.append(rec["ms_per_pair"])
            print(f"{name}: cached sim3={rec['ate_sim3']['rmse']:.2f} mm", flush=True)
            continue
        est, gt, dt = run_seq(model, d, device, args.batch, args.max_frames, args.invert)
        m = evaluate_trajectory(est, gt)
        np.savetxt(os.path.join(outdir, f"{name}_est_c2w.txt"), est.reshape(len(est), 16))
        rec = {"seq_id": name, "ate_sim3": m["ate_sim3"], "ate_se3": m["ate_se3"],
               "rpe_1": m.get("rpe_1"), "n_frames": len(gt), "ms_per_pair": dt}
        json.dump(rec, open(rec_path, "w"), indent=1)
        records.append(rec)
        ates.append(m["ate_sim3"]["rmse"])
        times.append(dt)
        print(f"{name}: sim3={m['ate_sim3']['rmse']:.2f} mm  ({dt:.1f} ms/pair)", flush=True)
    summ = {"method": "endofast3r-official", "n_seq": len(records),
            "ate_sim3_rmse_mean": float(np.mean(ates)),
            "ate_sim3_rmse_median": float(np.median(ates)),
            "ms_per_pair": float(np.mean(times)), "invert": args.invert,
            "records": records}
    json.dump({"summary": summ, "records": records}, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    print("MEAN Sim(3) ATE:", f"{np.mean(ates):.2f}", "median:", f"{np.median(ates):.2f}")


if __name__ == "__main__":
    main()
