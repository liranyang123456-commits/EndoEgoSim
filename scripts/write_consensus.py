"""No-GT consensus: if MD-VGGT-v2 disagrees with 8-point AND MASt3R,
and those two agree, replace the learned translation.

Also writes a conservative variant that swaps in the lower-jerk of
{8-point, MASt3R} instead of a median aligned onto the failed v2 cloud.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from endosim.eval.align import umeyama_alignment
from recompute_and_fuse import (
    SOTA, eval_with_scale_correction, gt_of, hybrid_gate, load_est,
    ref_frac, seqs, traj_jerk, write_summary,
)


def sim3_resid(A, B):
    s, R, t = umeyama_alignment(A[:, :3, 3], B[:, :3, 3], with_scale=True)
    pred = (s * (R @ A[:, :3, 3].T).T) + t
    scale = float(np.std(B[:, :3, 3])) + 1e-6
    return float(np.mean(np.linalg.norm(pred - B[:, :3, 3], axis=1)) / scale)


def align_pts(src, dst):
    s, R, t = umeyama_alignment(src[:, :3, 3], dst[:, :3, 3], with_scale=True)
    return (s * (R @ src[:, :3, 3].T).T) + t


def main():
    rec_med, rec_swap = [], []
    n_med = n_swap = 0
    for seq_dir in seqs():
        sid = os.path.basename(seq_dir.rstrip("/\\"))
        A = load_est("ours_v2_simtest", sid)
        B = load_est("eight_simtest", sid)
        C = load_est("mast3r_simtest", sid)
        if A is None:
            rec_med.append({"seq_id": sid, "error": "no v2"})
            rec_swap.append({"seq_id": sid, "error": "no v2"})
            continue
        gt, _ = gt_of(seq_dir, len(A))
        n = min(len(A), len(gt),
                len(B) if B is not None else len(A),
                len(C) if C is not None else len(A))
        A, gt = A[:n], gt[:n]
        B = None if B is None else B[:n]
        C = None if C is None else C[:n]

        fused_med, src_med = A.copy(), "learned"
        fused_swap, src_swap = A.copy(), "learned"
        if B is not None and C is not None:
            r_bc = sim3_resid(B, C)
            r_ab = sim3_resid(A, B)
            r_ac = sim3_resid(A, C)
            if r_bc < 0.45 and r_ab > 0.80 and r_ac > 0.80:
                med = np.median(np.stack([align_pts(B, A), align_pts(C, A)]), axis=0)
                fused_med = A.copy()
                fused_med[:, :3, 3] = med
                src_med = "consensus"
                n_med += 1
                # conservative: take the smoother of B/C as a whole trajectory
                jb, jc = traj_jerk(B), traj_jerk(C)
                fused_swap = (B if jb <= jc else C).copy()
                src_swap = "eight" if jb <= jc else "mast3r"
                n_swap += 1
            else:
                fused_med, src_med = hybrid_gate(
                    A, B, jerk_ratio=2.0, min_jerk=0.16, min_geo_path=0.3)
                fused_swap, src_swap = fused_med.copy(), src_med
                if src_med != "learned":
                    n_med += 1
                    n_swap += 1
        else:
            fused_med, src_med = hybrid_gate(
                A, B, jerk_ratio=2.0, min_jerk=0.16, min_geo_path=0.3)
            fused_swap, src_swap = fused_med.copy(), src_med
            if src_med != "learned":
                n_med += 1
                n_swap += 1

        for recs, fused, src, tag in (
            (rec_med, fused_med, src_med, "ours_consensus_simtest"),
            (rec_swap, fused_swap, src_swap, "ours_swap_simtest"),
        ):
            res = eval_with_scale_correction(fused, gt)
            res["reference_fraction"] = ref_frac(seq_dir)
            res["seq_id"] = sid
            res["n_frames_used"] = int(n)
            res["hybrid_src"] = src
            recs.append(res)
            out_dir = os.path.join(SOTA, tag)
            os.makedirs(out_dir, exist_ok=True)
            np.savetxt(os.path.join(out_dir, f"{sid}_est_c2w.txt"),
                       fused.reshape(len(fused), 16), fmt="%.6f")

    write_summary("ours_consensus_simtest", "ours_consensus", rec_med,
                  extra={"n_replaced": n_med, "note": "median aligned to v2"})
    write_summary("ours_swap_simtest", "ours_swap", rec_swap,
                  extra={"n_replaced": n_swap,
                         "note": "replace with lower-jerk of 8-point/MASt3R"})


if __name__ == "__main__":
    main()
