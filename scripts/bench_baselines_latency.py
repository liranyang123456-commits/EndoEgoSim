"""GPU latency of the published baselines on one machine (one method per process).

Model forward time only: frames are decoded and resized beforehand, CUDA is synchronised around every call.
Pairwise methods (AF-SfMLearner, EndoDAC, Endo-FASt3r, Reloc3r-512, EgoLite) are timed per frame pair at
batch size 1.  Window and sequence methods (pi3, CUT3R, DROID-SLAM) are timed on a 64-frame clip and divided
by 64 (protocol-A frame budget).  Run with the interpreter that has the baseline stacks (conda env `vggt`):

  python scripts/bench_baselines_latency.py --method pi3 --out results/latency/x18_pi3.json
  python scripts/bench_baselines_latency.py --all --out-dir results/latency --tag x18
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
METHODS = ["egolite", "afsfm", "endodac", "endofast3r", "reloc3r", "pi3", "cut3r", "droid"]
N_CLIP = 64


def sample_frames():
    seq = [ln.strip() for ln in open(os.path.join(ROOT, "lists", "simtest92.txt"), encoding="utf-8") if ln.strip()][0]
    if os.name != "nt" and len(seq) > 2 and seq[1] == ":":      # WSL: D:\x -> /mnt/d/x
        seq = "/mnt/" + seq[0].lower() + seq[2:].replace("\\", "/")
    seq = seq if os.path.isabs(seq) else os.path.join(ROOT, seq)
    paths = sorted(glob.glob(os.path.join(seq, "color", "*.png")))
    idx = np.linspace(0, len(paths) - 1, N_CLIP).round().astype(int)
    return seq, [paths[i] for i in idx]


def cuda_median(fn, warm, reps):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.asarray(ts)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "reps": reps}


def nparams(*mods):
    return sum(p.numel() for m in mods for p in m.parameters()) / 1e6


def bench(method):
    import torch
    seq, paths = sample_frames()
    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()
    out = {"method": method, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "sequence": seq}

    if method == "egolite":
        from endosim.lite.egolite import build_from_args
        ck = torch.load(os.path.join(ROOT, "results/egolite/r_v2/last.pth"), map_location="cpu", weights_only=False)
        model = build_from_args(ck["args"]); model.load_state_dict(ck["model"]); model = model.cuda().eval()
        img = torch.rand(2, 3, 256, 320, device=dev); ph = torch.rand(2, 1, 256, 320, device=dev)
        K = torch.tensor([[250.0, 0, 160], [0, 250.0, 128], [0, 0, 1]], device=dev)[None]
        def fn():
            with torch.no_grad():
                model(img[:1], img[1:], K, ph[:1], ph[1:])
        out.update(unit="pair", params_M=nparams(model), **cuda_median(fn, 20, 100), frames_per_call=1)

    elif method in ("afsfm", "endodac"):
        import egomotion_baselines as eb
        enc, dec, from_params = eb._load(method, dev)
        imgs = torch.from_numpy(np.stack([eb._read(p) for p in paths[:8]])).permute(0, 3, 1, 2)
        xs = [torch.cat([imgs[i + 1:i + 2], imgs[i:i + 1]], 1).to(dev) for i in range(7)]
        k = [0]
        def fn():
            with torch.no_grad():
                o = dec([enc(xs[k[0] % 7])]); from_params(o[0][:, 0], o[1][:, 0])
            k[0] += 1
        out.update(unit="pair", params_M=nparams(enc, dec), **cuda_median(fn, 10, 100), frames_per_call=1)

    elif method == "endofast3r":
        import eval_endofast3r as ef
        model = ef.load_model(dev)
        xs = [ef.prepare(p, dev) for p in paths[:8]]
        k = [0]
        def fn():
            i = k[0] % 7
            with torch.no_grad():
                model({"img": xs[i]}, {"img": xs[i + 1]})
            k[0] += 1
        out.update(unit="pair", params_M=nparams(model), input=list(xs[0].shape[-2:]),
                   **cuda_median(fn, 5, 60), frames_per_call=1)

    elif method == "reloc3r":
        import eval_slam as es
        sys.path.insert(0, es.RELOC3R_ROOT)
        from reloc3r.utils.image import load_images, check_images_shape_format
        from reloc3r.reloc3r_relpose import inference_relpose
        model = es._load_reloc3r(dev)
        pairs = []
        for i in range(7):
            ims = check_images_shape_format(load_images([paths[i], paths[i + 1]], size=512), dev)
            pairs.append([ims[0], ims[1]])
        k = [0]
        def fn():
            inference_relpose(pairs[k[0] % 7], model, dev)
            k[0] += 1
        out.update(unit="pair", params_M=nparams(model), **cuda_median(fn, 5, 60), frames_per_call=1)

    elif method == "pi3":
        import eval_slam as es
        from PIL import Image
        from torchvision import transforms
        model = es._load_pi3(dev)
        imgs = [Image.open(p).convert("RGB") for p in paths]
        w0, h0 = imgs[0].size
        limit = 255000
        scale = math.sqrt(limit / float(w0 * h0)); wt, ht = w0 * scale, h0 * scale
        kk, mm = round(wt / 14), round(ht / 14)
        while (kk * 14) * (mm * 14) > limit:
            if kk / mm > wt / ht: kk -= 1
            else: mm -= 1
        tw, th = max(1, kk) * 14, max(1, mm) * 14
        to_t = transforms.ToTensor()
        x = torch.stack([to_t(im.resize((tw, th), Image.Resampling.LANCZOS)) for im in imgs]).to(dev)[None].float()
        from torch.nn.attention import SDPBackend, sdpa_kernel
        default_attn = os.environ.get("PI3_ATTN") == "default"   # PI3_ATTN=default: let PyTorch pick (flash) attention
        def fn():
            with torch.no_grad():
                if default_attn:
                    model(x)
                else:                             # as in the accuracy evaluation (scripts/eval_slam.py)
                    with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                        model(x)
        r = cuda_median(fn, 1, 5)
        out.update(unit="clip64", params_M=nparams(model), input=[th, tw], frames_per_call=N_CLIP, **r)

    elif method == "cut3r":
        import eval_cut3r as ec
        model, _ = ec._load_cut3r(dev, ec.CUT3R_CKPT, 512)
        from dust3r.inference import inference
        views = ec._views_from_paths(paths, 512)
        def fn():
            with torch.no_grad():
                inference(views, model, dev, verbose=False)
        r = cuda_median(fn, 1, 3)
        out.update(unit="clip64", params_M=nparams(model), frames_per_call=N_CLIP, **r)

    elif method == "droid":
        import argparse as ap_
        import eval_slam as es
        slam_dir = os.path.join(es.DROID_ROOT, "droid_slam"); lt = os.path.join(es.DROID_ROOT, "thirdparty", "lietorch")
        for p in (slam_dir, lt):
            if p not in sys.path: sys.path.insert(0, p)
        from droid import Droid
        import cv2
        K = es.load_K(seq, cv2.imread(paths[0]))
        stream, image_size = es._droid_stream(paths, K)
        torch.multiprocessing.set_start_method("spawn", force=True)
        times, t_track, t_term = [], [], []
        for rep in range(4):                      # rep 0 is a warm-up (CUDA/cuDNN initialisation)
            args = ap_.Namespace(weights=es.DROID_CKPT, image_size=image_size, buffer=512, stereo=False, disable_vis=True,
                                 upsample=False, beta=0.3, filter_thresh=2.4, warmup=8, keyframe_thresh=4.0,
                                 frontend_thresh=16.0, frontend_window=25, frontend_radius=2, frontend_nms=1,
                                 backend_thresh=22.0, backend_radius=2, backend_nms=3)
            droid = Droid(args)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for t, image, intr in stream:
                droid.track(t, image, intrinsics=intr)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            droid.terminate(stream)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            if rep > 0:
                times.append((t2 - t0) * 1000); t_track.append((t1 - t0) * 1000); t_term.append((t2 - t1) * 1000)
            del droid
        ts = np.asarray(times)
        out.update(unit="clip64", params_M=None, median_ms=float(np.median(ts)), p95_ms=float(ts.max()), reps=3,
                   frames_per_call=N_CLIP, input=list(image_size),
                   track_ms_per_frame=float(np.median(t_track)) / N_CLIP, terminate_ms=float(np.median(t_term)))
    else:
        raise ValueError(method)

    out["ms_per_frame"] = out["median_ms"] / out["frames_per_call"]
    out["peak_mem_MB"] = float(torch.cuda.max_memory_allocated() / 2 ** 20)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--out")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "latency"))
    ap.add_argument("--tag", default="x18")
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    if a.all:
        os.makedirs(a.out_dir, exist_ok=True)
        for m in a.methods.split(","):
            o = os.path.join(a.out_dir, f"{a.tag}_{m}.json")
            log = os.path.join(a.out_dir, f"{a.tag}_{m}.log")
            with open(log, "w", encoding="utf-8") as f:
                subprocess.run([sys.executable, os.path.abspath(__file__), "--method", m, "--out", o],
                               stdout=f, stderr=subprocess.STDOUT)
            print(m, "->", "ok" if os.path.exists(o) else "FAILED (see " + log + ")", flush=True)
        return
    try:
        res = bench(a.method)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        res = {"method": a.method, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
