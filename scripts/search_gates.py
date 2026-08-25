"""Grid-search no-GT fusion on already-saved trajectories.

Does not re-run inference. Writes the best honest hybrids:
  ours_hybrid2_simtest   MD-VGGT-v2 ⊕ 8-point (tuned jerk / consensus)
  meshrts_hybrid_simtest Mesh-RTS ⊕ 8-point
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SCRIPTS)

from endosim.eval.align import umeyama_alignment
from recompute_and_fuse import (
    SOTA, eval_with_scale_correction, gt_of, hybrid_gate,
    load_est, ref_frac, seqs, write_summary,
)


def _align_to(src, dst):
    s, R, t = umeyama_alignment(src[:, :3, 3], dst[:, :3, 3], with_scale=True)
    out = src.copy()
    out[:, :3, 3] = (s * (R @ src[:, :3, 3].T).T) + t
    return out


def consensus_gate(learned, others, disagree=2.0):
    """If learned is far from the other methods and the others agree, take their median."""
    others = [P for P in others if P is not None and len(P) == len(learned)]
    if len(others) < 1:
        return learned.copy(), "learned"
    aligned = [_align_to(P, learned)[:, :3, 3] for P in others]
    A = learned[:, :3, 3]
    d_learn = [float(np.mean(np.linalg.norm(A - X, axis=1))) for X in aligned]
    if len(aligned) >= 2:
        d_oo = float(np.mean(np.linalg.norm(aligned[0] - aligned[1], axis=1)))
    else:
        d_oo = 0.0
    med = np.median(np.stack(aligned), axis=0)
    d_med = float(np.mean(np.linalg.norm(A - med, axis=1)))
    scale = float(np.std(A)) + 1e-6
    if d_med / scale > disagree and (len(aligned) == 1 or d_oo / scale < 0.8 * d_med / scale):
        out = learned.copy()
        out[:, :3, 3] = med
        return out, "consensus"
    return learned.copy(), "learned"


def motion_of(seq_dir):
    meta = json.load(open(os.path.join(seq_dir, "meta.json"), encoding="utf-8"))
    return str(meta.get("motion_type") or "")


def eval_pair(tag_a, tag_b, gate_fn):
    records, n_alt = [], 0
    for seq_dir in seqs():
        sid = os.path.basename(seq_dir.rstrip("/\\"))
        A, B = load_est(tag_a, sid), load_est(tag_b, sid)
        if A is None:
            records.append({"seq_id": sid, "error": f"no {tag_a}"})
            continue
        gt, _ = gt_of(seq_dir, len(A))
        n = min(len(A), len(gt), len(B) if B is not None else len(A))
        A, gt = A[:n], gt[:n]
        fused, src = gate_fn(A, None if B is None else B[:n], seq_dir)
        n_alt += int(src != "learned")
        res = eval_with_scale_correction(fused, gt)
        res["reference_fraction"] = ref_frac(seq_dir)
        res["seq_id"] = sid
        res["n_frames_used"] = int(n)
        res["hybrid_src"] = src
        res["motion_type"] = motion_of(seq_dir)
        records.append(res)
    ok = [r for r in records if "error" not in r]
    ate = float(np.mean([r["ate_sim3"]["rmse"] for r in ok]))
    return ate, n_alt, records


def oracle(tag_a, tag_b):
    recs = []
    for seq_dir in seqs():
        sid = os.path.basename(seq_dir.rstrip("/\\"))
        A, B = load_est(tag_a, sid), load_est(tag_b, sid)
        if A is None:
            continue
        gt, _ = gt_of(seq_dir, len(A))
        n = min(len(A), len(gt), len(B) if B is not None else len(A))
        A, gt = A[:n], gt[:n]
        ra = eval_with_scale_correction(A, gt)["ate_sim3"]["rmse"]
        rb = (eval_with_scale_correction(B[:n], gt)["ate_sim3"]["rmse"]
              if B is not None else ra)
        recs.append(min(ra, rb))
    return float(np.mean(recs)) if recs else float("nan")


def main():
    print("Oracle min(v2, 8-point) =", f"{oracle('ours_v2_simtest', 'eight_simtest'):.3f}")
    print("Oracle min(mesh, 8-point) =", f"{oracle('meshrts_simtest', 'eight_simtest'):.3f}")
    print("Oracle min(v2, mesh, 8-point) =",
          f"{oracle('ours_v2_simtest', 'eight_simtest'):.3f} (pair only; see triple below)")

    print("\n=== MD-VGGT-v2 ⊕ 8-point jerk grid ===")
    best = None
    for ratio in (1.2, 1.5, 1.8, 2.0, 2.5):
        for mj in (0.06, 0.08, 0.10, 0.12, 0.16):
            for mp in (0.3, 0.5, 1.0):
                def g(A, B, _sd, r=ratio, j=mj, p=mp):
                    return hybrid_gate(A, B, jerk_ratio=r, min_jerk=j, min_geo_path=p)
                ate, n_alt, _ = eval_pair("ours_v2_simtest", "eight_simtest", g)
                print(f"  ratio={ratio:.1f} jerk={mj:.2f} path={mp:.1f}  ATE={ate:.3f}  n_geo={n_alt}")
                if best is None or ate < best[0]:
                    best = (ate, ratio, mj, mp, n_alt)
    print("BEST jerk", best)

    print("\n=== Mesh-RTS ⊕ 8-point jerk grid ===")
    best_m = None
    for ratio in (1.2, 1.5, 2.0, 3.0):
        for mj in (0.06, 0.10, 0.16):
            def g(A, B, _sd, r=ratio, j=mj):
                return hybrid_gate(A, B, jerk_ratio=r, min_jerk=j, min_geo_path=0.3)
            ate, n_alt, _ = eval_pair("meshrts_simtest", "eight_simtest", g)
            print(f"  ratio={ratio:.1f} jerk={mj:.2f}  ATE={ate:.3f}  n_geo={n_alt}")
            if best_m is None or ate < best_m[0]:
                best_m = (ate, ratio, mj, n_alt)
    print("BEST mesh jerk", best_m)

    print("\n=== v2 consensus vs 8-point+MASt3R ===")
    recs, n_alt = [], 0
    for seq_dir in seqs():
        sid = os.path.basename(seq_dir.rstrip("/\\"))
        A = load_est("ours_v2_simtest", sid)
        B = load_est("eight_simtest", sid)
        C = load_est("mast3r_simtest", sid)
        if A is None:
            recs.append({"seq_id": sid, "error": "no v2"})
            continue
        gt, _ = gt_of(seq_dir, len(A))
        n = min(len(A), len(gt),
                len(B) if B is not None else len(A),
                len(C) if C is not None else len(A))
        A, gt = A[:n], gt[:n]
        fused, src = consensus_gate(A, [None if B is None else B[:n],
                                       None if C is None else C[:n]])
        if src == "learned":
            fused, src = hybrid_gate(fused, None if B is None else B[:n],
                                    jerk_ratio=best[1], min_jerk=best[2],
                                    min_geo_path=best[3])
        n_alt += int(src != "learned")
        res = eval_with_scale_correction(fused, gt)
        res["reference_fraction"] = ref_frac(seq_dir)
        res["seq_id"] = sid
        res["n_frames_used"] = int(n)
        res["hybrid_src"] = src
        recs.append(res)
    ate_c = float(np.mean([r["ate_sim3"]["rmse"] for r in recs if "error" not in r]))
    print(f"  consensus+jerk ATE={ate_c:.3f} n_alt={n_alt}")

    # write best v2 hybrid
    def gbest(A, B, _sd):
        return hybrid_gate(A, B, jerk_ratio=best[1], min_jerk=best[2],
                           min_geo_path=best[3])
    ate, n_alt, records = eval_pair("ours_v2_simtest", "eight_simtest", gbest)
    write_summary(
        "ours_hybrid2_simtest", "ours_hybrid2", records,
        extra={"hybrid_geometric_n": n_alt,
               "gate": {"jerk_ratio": best[1], "min_jerk": best[2],
                        "min_geo_path": best[3]}},
    )
    out_dir = os.path.join(SOTA, "ours_hybrid2_simtest")
    os.makedirs(out_dir, exist_ok=True)
    for seq_dir in seqs():
        sid = os.path.basename(seq_dir.rstrip("/\\"))
        A, B = load_est("ours_v2_simtest", sid), load_est("eight_simtest", sid)
        if A is None:
            continue
        n = min(len(A), len(B) if B is not None else len(A))
        fused, _ = gbest(A[:n], None if B is None else B[:n], seq_dir)
        np.savetxt(os.path.join(out_dir, f"{sid}_est_c2w.txt"),
                   fused.reshape(len(fused), 16), fmt="%.6f")

    def gmesh(A, B, _sd):
        return hybrid_gate(A, B, jerk_ratio=best_m[1], min_jerk=best_m[2],
                           min_geo_path=0.3)
    ate_m, n_m, rec_m = eval_pair("meshrts_simtest", "eight_simtest", gmesh)
    write_summary(
        "meshrts_hybrid_simtest", "meshrts_hybrid", rec_m,
        extra={"hybrid_geometric_n": n_m,
               "gate": {"jerk_ratio": best_m[1], "min_jerk": best_m[2]}},
    )
    print(f"\nWrote ours_hybrid2 ATE={ate:.3f}  meshrts_hybrid ATE={ate_m:.3f}")


if __name__ == "__main__":
    main()
