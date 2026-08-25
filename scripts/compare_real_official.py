"""官方真实域表: SCARED 左目全视频 + C3VD 全帧。"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOTA = os.path.join(ROOT, "results", "sota")

RUNS = [
    ("eight_scared_official", "8-point", "SCARED official left video"),
    ("ours_v3_scared_official", "MD-VGGT-v3 sliding", "SCARED official left video"),
    ("eight_c3vd_full", "8-point", "C3VD full consecutive"),
    ("ours_v3_c3vd_full", "MD-VGGT-v3 sliding", "C3VD full sliding"),
    ("ours_v3_c3vd", "MD-VGGT-v3", "C3VD 32-frame (旧协议复测)"),
    ("ours_v2_c3vd", "MD-VGGT-v2", "C3VD 32-frame (旧协议)"),
]


def main():
    print("| 协议 | 方法 | n | ATE Sim3 mean | median | RPE1 t | hop mm |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for tag, name, proto in RUNS:
        p = os.path.join(SOTA, tag, "summary.json")
        if not os.path.isfile(p):
            print(f"| {proto} | {name} | — | 待跑 | — | — | — |")
            continue
        s = json.load(open(p, encoding="utf-8"))["summary"]
        hop = s.get("protocol_hop_mm_mean")
        hop_s = "—" if hop is None else f"{hop:.1f}"
        ate = s.get("ate_sim3_rmse_mean")
        med = s.get("ate_sim3_rmse_median")
        rpe = s.get("rpe1_trans_mean")
        print(f"| {proto} | {name} | {s.get('n_seq')} | "
              f"{ate:.3f} | {med:.3f} | {rpe:.3f} | {hop_s} |")


if __name__ == "__main__":
    main()
