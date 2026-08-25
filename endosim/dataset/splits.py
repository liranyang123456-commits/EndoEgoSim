"""数据集划分（防泄漏）。

- 序列级: 按种子哈希确定性地分 train/val/test（70/15/15）
- 纹理源: 在 TextureBank 构建时按源视频 ID 互斥划分
- 索引文件: {split}.json 列出序列目录与统计
"""
from __future__ import annotations

import hashlib
import json
import os


def split_of_seed(seed: int, train_frac: float = 0.7, val_frac: float = 0.15) -> str:
    """确定性划分: 同一 seed 永远进同一 split。"""
    h = int(hashlib.md5(f"endoegosim-{seed}".encode()).hexdigest()[:8], 16)
    u = h / 0xFFFFFFFF
    if u < train_frac:
        return "train"
    if u < train_frac + val_frac:
        return "val"
    return "test"


def assign_splits(seeds: list[int]) -> dict[str, list[int]]:
    out = {"train": [], "val": [], "test": []}
    for s in seeds:
        out[split_of_seed(s)].append(s)
    return out


def build_index(root: str) -> dict:
    """扫描生成目录, 汇总各 split 的序列索引与统计。"""
    index = {"train": [], "val": [], "test": []}
    if not os.path.isdir(root):
        return index
    for split in index:
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            meta_p = os.path.join(d, name, "meta.json")
            if not os.path.exists(meta_p):
                continue
            try:
                with open(meta_p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                refs = [r for r in meta.get("reference_fraction", []) if r is not None]
                index[split].append({
                    "dir": os.path.join(split, name),
                    "seq_id": meta.get("seq_id", name),
                    "seed": meta.get("seed"),
                    "n_frames": meta.get("n_frames"),
                    "motion_type": meta.get("motion_type"),
                    "scene_kind": meta.get("scene_kind"),
                    "step_mm_mean": meta.get("motion_stats", {}).get("step_mm_mean"),
                    "deformation": (meta.get("deformation") or {}).get("strength", 0.0),
                    "n_objects": meta.get("n_objects"),
                    "reference_fraction_mean": (float(sum(refs) / len(refs))
                                                if refs else None),
                })
            except Exception:
                continue
    return index


def write_index(root: str, index: dict | None = None) -> None:
    idx = index if index is not None else build_index(root)
    with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=1, ensure_ascii=False)
    # 每split单独文件
    for split, items in idx.items():
        with open(os.path.join(root, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, indent=1, ensure_ascii=False)
    return idx
