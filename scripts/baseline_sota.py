"""SOTA 基线评测 (M5): VGGT / MASt3R / DUSt3R 的内窥镜 egomotion 评测。

本地代码/权重（路径引用, 不拷贝）:
- VGGT:   代码 D:/vggt-main/vggt-main, 权重 HF cache facebook/VGGT-1B
- MASt3R: 代码 D:/mast3r (含 dust3r 子仓库), 权重 D:/mast3r_checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
- DUSt3R: 代码 D:/mast3r/dust3r, 权重 D:/dust3r_checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

评测协议（对齐 endosim.eval, SCARED 惯例）:
- ATE(SE3): 6-DoF Umeyama 对齐（度量尺度口径）
- ATE(Sim3): 7-DoF 对齐（单目 up-to-scale 方法主口径）
- RPE(1/5/10): 平移按 Sim3 尺度校正后计算 + 旋转角
- 帧子采样: --max-frames 帧内均匀采样（GT 同步取对应位姿）
- 参照物分层: 仿真序列按 meta.reference_fraction 分桶

用法:
  python scripts/baseline_sota.py --method vggt   --seq "sim_data/test/seq_*" --out results/sota
  python scripts/baseline_sota.py --method mast3r --seq "sim_data/real_test/*" --out results/sota --max-frames 32
  python scripts/baseline_sota.py --method dust3r --seq "sim_data/test/seq_*" --out results/sota --max-frames 32 --limit 60
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.metrics import evaluate_trajectory, load_pose_txt, rpe
from endosim.eval.protocol import (chain_window_poses, list_color_frames,
                                   motion_stats_of_indices, select_frame_indices,
                                   sliding_windows)

VGGT_ROOT = r"D:\vggt-main\vggt-main"
MAST3R_ROOT = r"D:\mast3r"
MAST3R_CKPT = r"D:\mast3r_checkpoints\MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
DUST3R_CKPT = r"D:\dust3r_checkpoints\DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
ENDO3R_ROOT = r"E:\SOTA_Methods\Endo3R"
ENDO3R_CKPT = r"E:\SOTA_Methods\Endo3R\checkpoints\endo3r.pth"
ENDO3R_BASE = r"E:\SOTA_Methods\Endo3R\checkpoints\DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
VGGT_CKPT = os.path.expanduser(
    r"~\.cache\huggingface\hub\models--facebook--VGGT-1B\snapshots"
    r"\860abec7937da0a4c03c41d3c269c366e82abdf9\model.safetensors")


def find_vggt_ckpt() -> str:
    if os.path.isfile(VGGT_CKPT):
        return VGGT_CKPT
    import glob as g
    for p in g.glob(os.path.expanduser(
            r"~\.cache\huggingface\hub\models--facebook--VGGT-1B\snapshots\*\model.safetensors")):
        return p
    raise FileNotFoundError("VGGT 权重未找到 (HF cache facebook/VGGT-1B)")


# ---------------------------------------------------------------------------
# 模型加载与推理
# ---------------------------------------------------------------------------

def _torch_load_any(path: str):
    """加载 safetensors 或 torch.save 的 {'model': ...} / 裸 state_dict。"""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd.get("model", sd) if isinstance(sd, dict) else sd


def load_model(method: str, device: str, ckpt: str | None = None):
    if method == "vggt":
        sys.path.insert(0, VGGT_ROOT)
        import torch
        from safetensors.torch import load_file
        from vggt.models.vggt import VGGT
        model = VGGT()
        if ckpt:
            sd = _torch_load_any(ckpt)
            model.load_state_dict(sd)
        else:
            model.load_state_dict(load_file(find_vggt_ckpt()))
        return model.to(device).eval()

    import argparse as _argparse  # noqa: F401
    import torch as _torch
    # torch>=2.6 weights_only 默认值变化; 检查点含 argparse.Namespace, 白名单放行
    _torch.serialization.add_safe_globals([_argparse.Namespace])

    if method == "endo3r":
        # Endo3R: 内窥镜 DUSt3R 变体(流一致性正则); 评测用其微调后的 dust3r 配对预测器 + dense GA
        sys.path.insert(0, ENDO3R_ROOT)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "_shim"))
        from dust3r.model import Endo3R
        full = Endo3R(dus3r_name=ENDO3R_BASE, use_feat=False)
        sd = _torch.load(ENDO3R_CKPT, map_location="cpu", weights_only=False)["model"]
        full.load_state_dict(sd, strict=False)
        full.eval()
        return full.dust3r.to(device).eval()

    # dust3r / mast3r 共用 MASt3R 仓库(内嵌 dust3r 子仓库, 需 path_to_dust3r 建立路径)
    sys.path.insert(0, MAST3R_ROOT)
    import mast3r.utils.path_to_dust3r  # noqa: F401
    if method == "mast3r":
        from mast3r.model import AsymmetricMASt3R
        model = AsymmetricMASt3R.from_pretrained(MAST3R_CKPT)
    else:
        from dust3r.model import AsymmetricCroCo3DStereo
        model = AsymmetricCroCo3DStereo.from_pretrained(DUST3R_CKPT)
    if ckpt:
        sd = _torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        model.load_state_dict(sd)
    return model.to(device).eval()


def run_vggt(model, frame_paths, device, args):
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    images = load_and_preprocess_images(frame_paths).to(device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            pred = model(images)
    ext, _ = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    ext = ext.reshape(-1, 3, 4).float().cpu().numpy()   # (S,3,4) world->cam
    est = np.zeros((len(ext), 4, 4))
    for i, e in enumerate(ext):
        T = np.eye(4)
        T[:3, :] = e
        est[i] = np.linalg.inv(T)                        # -> c2w
    return est


def run_dust3r_like(model, frame_paths, device, args):
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    imgs = load_images(frame_paths, size=args.image_size, verbose=False)
    pairs = make_pairs(imgs, scene_graph=args.scene_graph,
                       prefilter=None, symmetrize=True)
    t_inf = time.time()
    output = inference(pairs, model, device, batch_size=args.batch_size,
                       verbose=False)
    t_inf = time.time() - t_inf
    if getattr(args, "gt_mask", False):
        _apply_gt_motion_mask(output, frame_paths,
                              mode=getattr(args, "gt_mask_mode", "hard"))
    # 该版本 dust3r (MASt3R 仓库内嵌) 的 GA 不接受 network 参数, init 用 'mst'
    scene = global_aligner(output, device=device,
                           mode=GlobalAlignerMode.PointCloudOptimizer,
                           verbose=False)
    t_ga = time.time()
    scene.compute_global_alignment(init="mst", niter=args.niter,
                                   schedule="cosine", lr=0.01)
    est = scene.get_im_poses().detach().float().cpu().numpy()   # (S,4,4) c2w
    print(f"    [timing] pairs={len(pairs)} inference={t_inf:.0f}s "
          f"GA={time.time()-t_ga:.0f}s")
    return est


def _apply_gt_motion_mask(output, frame_paths, mode: str = "hard"):
    """运动分割条件评测。

    hard: 非静态像素置信度置零 (朴素上界, M7c 已证对稠密 GA 有害)
    soft: 形变组织降权、独立运动物体更强降权 —— 保留相对位姿观测, 抑制动态干扰
          (对应鲁棒核/动态感知 BA 的可微近似)

    output: inference 的批处理 dict —— view1/view2={img,instance,idx,...},
    pred1/pred2={pts3d,conf,...}, conf (B,H,W), instance 列表对应 frame_paths 下标。
    """
    import cv2
    import torch
    n_zeroed = 0
    for vk, pk in (("view1", "pred1"), ("view2", "pred2")):
        views, preds = output[vk], output[pk]
        conf = preds.get("conf")
        if conf is None:
            continue
        insts = views.get("instance", [])
        for b, inst in enumerate(insts):
            if b >= conf.shape[0]:
                break
            try:
                t = int(str(inst).split(".")[0])
            except ValueError:
                t = int(views["idx"][b])
            if not (0 <= t < len(frame_paths)):
                continue
            mp = os.path.join(os.path.dirname(os.path.dirname(frame_paths[t])),
                              "motion_mask", f"{t:06d}.png")
            if not os.path.exists(mp):
                continue
            m = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            h, w = conf.shape[-2:]
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            if mode == "soft":
                # 形变组织仍携带相对位姿; 独立运动/无效像素更强降权
                w = np.ones(m.shape, np.float32)
                w[m == 0] = 0.05
                w[m == 2] = 0.25
                w[m == 3] = 0.08
                conf[b] = conf[b] * torch.as_tensor(
                    w, device=conf[b].device, dtype=conf[b].dtype)
                n_zeroed += 1
            else:
                dyn = (m != 1)                      # 非静态参照(2形变 3运动物体 0无效)
                if dyn.any():
                    conf[b][dyn] = 1e-8
                    n_zeroed += 1
    if n_zeroed:
        print(f"    [gt-mask] 已置零 {n_zeroed} 个视图的非静态像素置信度")


def run_mast3r_sparse(model, frame_paths, device, args):
    """MASt3R sparse GA（官方 SfM 路径, 远快于 dense GA）。"""
    import tempfile
    import torch
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    imgs = load_images(frame_paths, size=args.image_size, verbose=False)
    pairs = make_pairs(imgs, scene_graph=args.scene_graph,
                       prefilter=None, symmetrize=True)
    t0 = time.time()
    cache_dir = tempfile.mkdtemp(prefix="mast3r_ga_")
    scene = sparse_global_alignment(
        frame_paths, pairs, cache_dir, model,
        lr1=0.07, niter1=args.niter, lr2=0.014, niter2=max(args.niter // 2, 0),
        device=device, opt_depth=True, shared_intrinsics=False,
        matching_conf_thr=5.0)
    est = scene.get_im_poses().detach().float().cpu().numpy()   # (S,4,4) c2w
    print(f"    [timing] sparse_ga pairs={len(pairs)} total={time.time()-t0:.0f}s")
    torch.cuda.empty_cache()
    return est


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------

def eval_with_scale_correction(est: np.ndarray, gt: np.ndarray) -> dict:
    res = evaluate_trajectory(est, gt)
    s = res["ate_sim3"]["scale"]
    est_sc = est.copy()
    est_sc[:, :3, 3] *= s          # Sim3 尺度校正, RPE 平移对 up-to-scale 方法才公平
    for g in (1, 5, 10):
        if f"rpe_{g}" in res:
            res[f"rpe_{g}"] = rpe(est_sc, gt, g)
    return res


def stratified_summary(records: list) -> list:
    """按参照物比例分桶汇总（仅仿真序列有 reference_fraction）。"""
    buckets = {"低参照(<0.3)": [], "中参照(0.3-0.7)": [], "高参照(>0.7)": []}
    for r in records:
        rf = r.get("reference_fraction")
        if rf is None:
            continue
        key = ("低参照(<0.3)" if rf < 0.3 else
               "中参照(0.3-0.7)" if rf < 0.7 else "高参照(>0.7)")
        buckets[key].append(r)
    out = []
    for key, items in buckets.items():
        if not items:
            continue
        out.append({
            "bucket": key, "n": len(items),
            "ate_se3_mean": float(np.mean([r["ate_se3"]["rmse"] for r in items])),
            "ate_sim3_mean": float(np.mean([r["ate_sim3"]["rmse"] for r in items])),
            "rpe1_t_mean": float(np.mean([r["rpe_1"]["trans_mm_mean"] for r in items])),
            "rpe1_r_mean": float(np.mean([r["rpe_1"]["rot_deg_mean"] for r in items])),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["vggt", "mast3r", "dust3r", "endo3r"])
    ap.add_argument("--seq", default=None, help="序列目录 glob（如 sim_data/test/seq_*）")
    ap.add_argument("--list", default=None,
                    help="序列目录清单文件（每行一个目录; 与 --seq 二选一）")
    ap.add_argument("--protocol", default="uniform",
                    choices=["uniform", "consecutive", "stride", "adaptive", "sliding"],
                    help="选帧协议: uniform=旧均匀抽帧(稀疏序列会放大基线); "
                         "adaptive=限制相邻平移; sliding=重叠窗估计再拼接")
    ap.add_argument("--stride", type=int, default=1, help="protocol=stride 的帧步长")
    ap.add_argument("--max-step-mm", type=float, default=12.0,
                    help="protocol=adaptive 时相邻选中帧的最大平移(mm)")
    ap.add_argument("--window", type=int, default=16, help="sliding 窗口长度")
    ap.add_argument("--window-stride", type=int, default=8, help="sliding 窗口步长")
    ap.add_argument("--gt-mask-mode", default="hard", choices=["hard", "soft"],
                    help="--gt-mask 时 hard=置零 / soft=鲁棒降权")
    ap.add_argument("--out", default="results/sota", help="结果输出目录")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=0,
                    help=">0 时帧内均匀子采样到此帧数（GT 同步取对应位姿）")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只评前 N 条序列")
    ap.add_argument("--stride-seqs", type=int, default=1, help="序列级步进采样")
    ap.add_argument("--image-size", type=int, default=512, help="dust3r/mast3r 输入尺寸")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--scene-graph", default="complete",
                    choices=["complete", "swin", "swin2", "swin3", "logwin"])
    ap.add_argument("--niter", type=int, default=300, help="全局对齐优化迭代数")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="VGGT 推理精度")
    ap.add_argument("--tag", default=None, help="结果子目录名（默认 = 方法名）")
    ap.add_argument("--ckpt", default=None, help="微调权重(VGGT/dust3r 通用)")
    ap.add_argument("--gt-mask", action="store_true",
                    help="dust3r/endo3r: 用GT运动掩码置零非静态像素置信度(运动分割条件评测上界)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="已有 {seq}_est_c2w.txt 则跳过该序列")
    args = ap.parse_args()

    if args.list:
        seqs = [ln.strip() for ln in open(args.list, encoding="utf-8")
                if ln.strip() and not ln.startswith("#")]
    else:
        seqs = sorted(glob.glob(args.seq or ""))
    def _has_frames(s):
        return (os.path.isdir(os.path.join(s, "color"))
                or os.path.isfile(os.path.join(s, "color_index.json")))
    seqs = [s for s in seqs if os.path.isdir(s)
            and os.path.exists(os.path.join(s, "pose_c2w.txt"))
            and _has_frames(s)]
    if args.stride_seqs > 1:
        seqs = seqs[::args.stride_seqs]
    if args.limit > 0:
        seqs = seqs[:args.limit]
    print(f"[{args.method}] 待评测 {len(seqs)} 条序列 "
          f"(max_frames={args.max_frames or '全部'})")

    tag = args.tag or args.method
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    model = load_model(args.method, args.device, ckpt=args.ckpt)
    print(f"模型加载完成 ({time.time()-t0:.0f}s)")

    runner = run_vggt if args.method == "vggt" else (
        run_mast3r_sparse if args.method == "mast3r" else run_dust3r_like)
    records = []
    for i, seq_dir in enumerate(seqs):
        seq_id = os.path.basename(seq_dir.rstrip("/\\"))
        est_path = os.path.join(out_dir, f"{seq_id}_est_c2w.txt")
        if args.skip_existing and os.path.isfile(est_path):
            print(f"[{i+1}/{len(seqs)}] {seq_id}: skip existing", flush=True)
            continue
        t_seq = time.time()
        try:
            frames = list_color_frames(seq_dir)
            gt_all = load_pose_txt(os.path.join(seq_dir, "pose_c2w.txt"))
            n_use = min(len(frames), len(gt_all))
            frames, gt_all = frames[:n_use], gt_all[:n_use]
            if args.protocol == "sliding":
                wins_idx = sliding_windows(n_use, args.window, args.window_stride)
                packed = []
                for w in wins_idx:
                    wp = [frames[k] for k in w]
                    est_w = runner(model, wp, args.device, args)
                    packed.append((w, est_w))
                est, idx = chain_window_poses(packed, with_scale=True)
            else:
                idx = select_frame_indices(
                    gt_all, protocol=args.protocol, max_frames=args.max_frames,
                    stride=args.stride, max_step_mm=args.max_step_mm)
                frame_paths = [frames[k] for k in idx]
                est = runner(model, frame_paths, args.device, args)
            gt = gt_all[idx]
            hop = motion_stats_of_indices(gt_all, idx)

            res = eval_with_scale_correction(est, gt)
            res["protocol_hop"] = hop

            meta_p = os.path.join(seq_dir, "meta.json")
            if os.path.exists(meta_p):
                meta = json.load(open(meta_p, encoding="utf-8"))
                refs = [r for r in meta.get("reference_fraction", []) if r is not None]
                res["reference_fraction"] = float(np.mean(refs)) if refs else None
                res["motion_type"] = meta.get("motion_type")
                res["scene_kind"] = meta.get("scene_kind")
            res["seq_id"] = seq_id
            res["n_frames_used"] = int(len(est))
            res["time_sec"] = round(time.time() - t_seq, 1)
            records.append(res)

            np.savetxt(os.path.join(out_dir, f"{seq_id}_est_c2w.txt"),
                       est.reshape(len(est), 16), fmt="%.6f")
            print(f"[{i+1}/{len(seqs)}] {seq_id}: "
                  f"ATE(Sim3)={res['ate_sim3']['rmse']:.3f}mm "
                  f"scale={res['ate_sim3']['scale']:.3f} "
                  f"RPE1_t={res['rpe_1']['trans_mm_mean']:.3f}mm "
                  f"hop={hop['step_mm_mean']:.1f}/{hop['step_mm_max']:.1f}mm "
                  f"({res['time_sec']}s)")
        except Exception:
            print(f"[{i+1}/{len(seqs)}] {seq_id}: FAILED")
            traceback.print_exc()
            records.append({"seq_id": seq_id, "error": traceback.format_exc()[-300:]})

    # ---- 汇总 ----
    ok = [r for r in records if "error" not in r]
    summary = {
        "method": args.method, "n_seq": len(ok), "n_failed": len(records) - len(ok),
        "protocol": {"name": args.protocol,
                     "max_frames": args.max_frames or "all",
                     "stride": args.stride, "max_step_mm": args.max_step_mm,
                     "window": args.window, "window_stride": args.window_stride,
                     "gt_mask": bool(args.gt_mask), "gt_mask_mode": args.gt_mask_mode,
                     "image_size": args.image_size if args.method != "vggt" else 518,
                     "scene_graph": args.scene_graph, "niter": args.niter},
        "protocol_hop_mm_mean": (float(np.mean([r.get("protocol_hop", {}).get("step_mm_mean", 0)
                                                for r in ok])) if ok else None),
        "ate_se3_rmse_mean": float(np.mean([r["ate_se3"]["rmse"] for r in ok])) if ok else None,
        "ate_se3_rmse_median": float(np.median([r["ate_se3"]["rmse"] for r in ok])) if ok else None,
        "ate_sim3_rmse_mean": float(np.mean([r["ate_sim3"]["rmse"] for r in ok])) if ok else None,
        "ate_sim3_rmse_median": float(np.median([r["ate_sim3"]["rmse"] for r in ok])) if ok else None,
        "rpe1_trans_mean": float(np.mean([r["rpe_1"]["trans_mm_mean"] for r in ok])) if ok else None,
        "rpe1_rot_mean": float(np.mean([r["rpe_1"]["rot_deg_mean"] for r in ok])) if ok else None,
        "total_time_sec": round(time.time() - t0, 1),
    }
    if ok:
        summary["stratified"] = stratified_summary(ok)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=1)

    print(f"\n=== {args.method} 汇总 ({len(ok)} 成功 / {len(records)-len(ok)} 失败) ===")
    for k in ("ate_se3_rmse_mean", "ate_sim3_rmse_mean", "ate_sim3_rmse_median",
              "rpe1_trans_mean", "rpe1_rot_mean"):
        v = summary[k]
        print(f"  {k}: {v:.3f}" if v is not None else f"  {k}: n/a")
    for b in summary.get("stratified", []):
        print(f"  [{b['bucket']}] n={b['n']}  ATE(Sim3)={b['ate_sim3_mean']:.3f}mm  "
              f"RPE1_t={b['rpe1_t_mean']:.3f}mm")
    print(f"结果: {out_dir}/summary.json  总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
