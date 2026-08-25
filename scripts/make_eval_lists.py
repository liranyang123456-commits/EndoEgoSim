"""从已有评测记录抽出官方 92 条分层子集, 保证后续对比同一序列集。"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "sota", "vggt_ft2_simtest", "summary.json")
OUT_DIR = os.path.join(ROOT, "lists")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    recs = json.load(open(SRC, encoding="utf-8"))["records"]
    ok = [r for r in recs if "error" not in r]
    lines = []
    buckets = {"low": [], "mid": [], "high": []}
    for r in ok:
        sid = r["seq_id"]
        # 序列可能在 train/val/test 任一 split (分层子集跨 split)
        found = None
        for split in ("test", "val", "train"):
            p = os.path.join(ROOT, "sim_data", split, sid)
            if os.path.isdir(p):
                found = p
                break
        if found is None:
            print("MISSING", sid)
            continue
        lines.append(found)
        rf = r.get("reference_fraction")
        if rf is None:
            continue
        key = "low" if rf < 0.3 else ("mid" if rf < 0.7 else "high")
        buckets[key].append(found)
    out = os.path.join(OUT_DIR, "simtest92.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for k, vs in buckets.items():
        with open(os.path.join(OUT_DIR, f"simtest92_{k}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(vs) + "\n")
    print(f"wrote {len(lines)} seqs -> {out}")
    print({k: len(v) for k, v in buckets.items()})


if __name__ == "__main__":
    main()
