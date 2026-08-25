"""海量数据集生成入口（多进程）。

用法:
  python scripts/generate_dataset.py --n-seq 100 --out D:/ego_motiion_Camera/sim_data
  python scripts/generate_dataset.py --config configs/default.json --n-seq 1000 --workers 16

每序列: seed 决定 split(train/val/test), 目录 sim_data/{split}/seq_{seed:08d}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.config import GenConfig
from endosim.dataset.splits import split_of_seed, write_index

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "sim_data")

_BANK = None
_CFG = None
_OUT = None
_FLOW_FRAC = None
_USE_GPU = False


def _init_worker(cfg: GenConfig, out_root: str, use_bank: bool, flow_frac=None,
                 use_gpu=False):
    global _BANK, _CFG, _OUT, _FLOW_FRAC, _USE_GPU
    _CFG = cfg
    _OUT = out_root
    _FLOW_FRAC = flow_frac
    _USE_GPU = use_gpu
    if use_bank:
        from endosim.scene.texture import build_bank
        _BANK = build_bank(train_frac=0.7, seed=0)
    else:
        _BANK = None


def _gen_one(seed: int) -> dict:
    global _BANK, _CFG, _OUT
    from endosim.dataset.generator import generate_sequence
    from endosim.dataset.writer import write_sequence
    import copy
    split = split_of_seed(seed)
    # 纹理 split 与序列 split 一致（防泄漏: test 序列用 test 纹理）
    tex_split = "train" if split == "train" else ("test" if split == "test" else "train")
    seq_id = f"seq_{seed:08d}"
    out_dir = os.path.join(_OUT, split, seq_id)
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        return {"seq_id": seq_id, "split": split, "skip": True}
    try:
        cfg_seq = copy.deepcopy(_CFG)
        # 光流子集（确定性: 种子哈希决定, 控制磁盘占用）
        if _FLOW_FRAC is not None:
            cfg_seq.output.save_flow = ((seed * 2654435761) % 100) < _FLOW_FRAC * 100
        seq = generate_sequence(seq_id, seed, cfg_seq, bank=_BANK, split=tex_split,
                                meta_split=split, use_gpu=_USE_GPU)
        info = write_sequence(seq, out_dir)
        info.update({"split": split, "skip": False, "flow": cfg_seq.output.save_flow})
        return info
    except Exception as e:
        import traceback
        return {"seq_id": seq_id, "split": split,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="GenConfig JSON（默认内置配置）")
    ap.add_argument("--n-seq", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 4, 1))
    ap.add_argument("--no-bank", action="store_true", help="禁用真实纹理库（纯程序化）")
    ap.add_argument("--gpu", action="store_true",
                    help="用 nvdiffrast GPU 光栅化（建议 workers<=4 共享单卡）")
    ap.add_argument("--flow-frac", type=float, default=None,
                    help="带稠密光流GT的序列比例(0-1, 默认按配置文件全量)")
    args = ap.parse_args()

    if args.config:
        cfg = GenConfig.load(args.config)
    else:
        cfg = GenConfig()
    os.makedirs(args.out, exist_ok=True)
    cfg.save(os.path.join(args.out, "gen_config.json"))

    seeds = list(range(args.seed_start, args.seed_start + args.n_seq))
    use_bank = not args.no_bank
    t0 = time.time()
    print(f"生成 {len(seeds)} 条序列 -> {args.out} (workers={args.workers}, "
          f"纹理库={'启用' if use_bank else '禁用'})")

    results = []
    with Pool(args.workers, initializer=_init_worker,
              initargs=(cfg, args.out, use_bank, args.flow_frac, args.gpu)) as pool:
        for i, r in enumerate(pool.imap_unordered(_gen_one, seeds)):
            results.append(r)
            status = "skip" if r.get("skip") else ("ERR: " + r.get("error", "?")
                                                   if "error" in r else "ok")
            print(f"[{i+1}/{len(seeds)}] {r['seq_id']} ({r['split']}) {status} "
                  f"({time.time()-t0:.0f}s)")

    # 汇总索引
    idx = write_index(args.out)
    ok = [r for r in results if not r.get("skip") and "error" not in r]
    err = [r for r in results if "error" in r]
    total_frames = sum(r.get("n_frames", 0) for r in ok)
    total_mb = sum(r.get("size_mb", 0) for r in ok)
    print(f"\n完成: {len(ok)} 成功 / {len(err)} 失败 / "
          f"{sum(1 for r in results if r.get('skip'))} 跳过")
    print(f"总帧数 {total_frames}, 总大小 {total_mb:.0f} MB, 耗时 {time.time()-t0:.0f}s")
    print(f"索引: {args.out}/index.json "
          f"(train={len(idx['train'])}, val={len(idx['val'])}, test={len(idx['test'])})")
    if err:
        print("失败列表:")
        for r in err[:10]:
            print(" ", r["seq_id"], r["error"])


if __name__ == "__main__":
    main()
