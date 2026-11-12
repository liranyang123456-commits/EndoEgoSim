"""Merge WSL2 DROID-SLAM trajectories into r2_2_sim_ar_expanded.json."""
import json
import os
import sys
import cv2
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, "scripts")
import eval_ar_registration as ear  # noqa: E402

d = json.loads(open("results/revision_vrih/r2_2_sim_ar_expanded.json", encoding="utf-8").read())
droid_meds_ts = []
for r in d["per_sequence_medians"]:
    seq = os.path.join("sim_data_abs/test", r["sequence"])
    K, pose, traj = ear.load_clip(seq)
    Xm = traj[0]
    s = r["stride"]
    sid = r["sequence"]
    tr = np.loadtxt(f"results/revision_vrih/droid_abs9/{sid}_c2w.txt").reshape(-1, 4, 4)
    errs = []
    for t in range(0, len(pose) - s, s):
        g0 = cv2.imread(os.path.join(seq, "color", f"{t:06d}.png"), 0)
        g1 = cv2.imread(os.path.join(seq, "color", f"{t + s:06d}.png"), 0)
        Xc = (pose[t][:3, :3].T @ (traj[t] - pose[t][:3, 3]).T).T
        R0, t0 = ear.kabsch(Xm, Xc)
        src, za = ear.project_w(traj[t], pose[t], K)
        truth, zb = ear.project_w(traj[t + s], pose[t + s], K)
        vis = (za > 1) & (zb > 1)
        if vis.mean() < 0.7:
            continue
        Re, te = ear.cam_static(pose[t], pose[t + s])
        mag = float(np.linalg.norm(te))
        Rm, tm = ear.cam_static(tr[t], tr[t + s])
        tm = tm / max(np.linalg.norm(tm), 1e-8) * mag
        uvp, zp = ear.project_c(K, (Rm @ Xc.T).T + tm)
        pred, guess = ((src, (R0, t0)) if np.median(zp) < 1 else (uvp, (Rm @ R0, Rm @ t0 + tm)))
        got = ear.register_local(g0, g1, src, pred, Xm, K, guess, vis, 18)
        errs.append(ear.px(got[vis], truth[vis]))
    r["medians"]["local_droid_truescale"] = round(float(np.median(errs)), 2)
    droid_meds_ts.append((float(np.median(errs)), float(np.mean(np.array(errs) < 10.0))))
    print(f"{sid}: droid_truescale={r['medians']['local_droid_truescale']:.2f}  egolite={r['medians']['local_egolite']:.2f}")

arr = np.array([x[0] for x in droid_meds_ts])
ego_arr = np.array([r["medians"]["local_egolite"] for r in d["per_sequence_medians"]])
d["table"]["local_droid_truescale"] = {
    "median_px": float(np.median(arr)),
    "mean_px": float(np.mean(arr)),
    "within_10px": float(np.mean([x[1] for x in droid_meds_ts])),
    "wilcoxon_p_vs_egolite": float(wilcoxon(arr, ego_arr).pvalue),
}
open("results/revision_vrih/r2_2_sim_ar_expanded.json", "w", encoding="utf-8").write(json.dumps(d, indent=2))
print("DROID Sim-AR summary:", d["table"]["local_droid_truescale"])
