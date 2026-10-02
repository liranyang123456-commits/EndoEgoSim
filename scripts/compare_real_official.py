"""官方真实域表: SCARED 左目全视频 + C3VD 全帧。"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOTA = os.path.join(ROOT, "results", "sota")

RUNS = [
    ("eight_scared_official", "8-point", "SCARED official left video"),
    ("ours_v3_scared_official", "MD-VGGT-v3 sliding", "SCARED official left video"),
    ("ours_v5_scared_official", "MD-VGGT-v5 sliding", "SCARED official left video"),
    ("ours_v5kf_scared_official", "MD-VGGT-v5kf sliding", "SCARED official left video"),
    ("eight_c3vd_full", "8-point", "C3VD full consecutive"),
    ("ours_v3_c3vd_full", "MD-VGGT-v3 sliding", "C3VD full sliding"),
    ("ours_v3_c3vd", "MD-VGGT-v3", "C3VD 32-frame (旧协议复测)"),
    ("ours_v2_c3vd", "MD-VGGT-v2", "C3VD 32-frame (旧协议)"),
    ("eight_stereomis", "8-point", "StereoMIS 64-frm uniform"),
    ("ours_v3_stereomis", "MD-VGGT-v3", "StereoMIS 64-frm uniform"),
    ("droid_stereomis", "DROID-SLAM", "StereoMIS 64-frm uniform"),
    ("eight_stereomis_full", "8-point", "StereoMIS Hayoz full-rate"),
    ("ours_v5_stereomis_full", "MD-VGGT-v5 sliding", "StereoMIS Hayoz full-rate"),
    ("ours_v6_infer_stereomis_full", "MD-VGGT-v6-infer gated", "StereoMIS Hayoz full-rate"),
    ("droid_stereomis_full", "DROID-SLAM", "StereoMIS Hayoz full-rate"),
]


def _hop_norm(blob: dict) -> float | None:
    """Macro-mean of per-sequence ATE / hop. Not a ranking metric."""
    recs = [r for r in blob.get("records", []) if "error" not in r]
    xs = []
    for r in recs:
        ate = r.get("ate_sim3", {}).get("rmse")
        hop = r.get("protocol_hop", {}).get("step_mm_mean")
        if ate is None or hop is None or hop <= 1e-6:
            continue
        xs.append(float(ate) / float(hop))
    if not xs:
        hop = blob.get("summary", {}).get("protocol_hop_mm_mean")
        ate = blob.get("summary", {}).get("ate_sim3_rmse_mean")
        if hop and ate and hop > 1e-6:
            return float(ate) / float(hop)
        return None
    return float(sum(xs) / len(xs))


def main():
    print("| 协议 | 方法 | n | ATE Sim3 mean | median | RPE1 t | hop mm | ATE/hop |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for tag, name, proto in RUNS:
        p = os.path.join(SOTA, tag, "summary.json")
        if not os.path.isfile(p):
            print(f"| {proto} | {name} | — | 待跑 | — | — | — | — |")
            continue
        blob = json.load(open(p, encoding="utf-8"))
        s = blob["summary"]
        hop = s.get("protocol_hop_mm_mean")
        hop_s = "—" if hop is None else f"{hop:.1f}"
        recs = [r for r in blob.get("records", []) if "error" not in r]
        finite = []
        for r in recs:
            a = r.get("ate_sim3", {}).get("rmse")
            if a is None or not isinstance(a, (int, float)):
                continue
            if a < 1e6:
                finite.append(r)
        exploded = len(recs) - len(finite)
        if exploded and tag.endswith("stereomis_full"):
            if not finite:
                print(f"| {proto} | {name} | 0/{len(recs)} | 非有限 (全部爆炸) | — | — | {hop_s} | — |")
                continue
            ate = sum(r["ate_sim3"]["rmse"] for r in finite) / len(finite)
            meds = sorted(r["ate_sim3"]["rmse"] for r in finite)
            med = meds[len(meds) // 2]
            rpe = sum(r.get("rpe_1", {}).get("trans_mm_mean", 0.0) for r in finite) / len(finite)
            hn = _hop_norm({"records": finite, "summary": s})
            hn_s = "—" if hn is None else f"{hn:.2f}"
            print(f"| {proto} | {name} | {len(finite)}/{len(recs)} 有限 | "
                  f"{ate:.3f} | {med:.3f} | {rpe:.3f} | {hop_s} | {hn_s} |")
            continue
        ate = s.get("ate_sim3_rmse_mean")
        med = s.get("ate_sim3_rmse_median")
        rpe = s.get("rpe1_trans_mean")
        hn = _hop_norm(blob)
        hn_s = "—" if hn is None else f"{hn:.2f}"
        if ate is None:
            print(f"| {proto} | {name} | {s.get('n_seq')} | 待跑 | — | — | {hop_s} | — |")
            continue
        print(f"| {proto} | {name} | {s.get('n_seq')} | "
              f"{ate:.3f} | {med:.3f} | {rpe:.3f} | {hop_s} | {hn_s} |")


if __name__ == "__main__":
    main()
