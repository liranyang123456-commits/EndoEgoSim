"""Compare camera-motion leakage in near-static-camera StereoMIS segments.

For each sequence, select the 20-s window with the smallest GT camera-centre spread.
Fit a Sim(3) transform from method trajectory to GT over the whole sequence, then
measure the estimated camera trajectory's maximum distance from its window mean.
These are relative leakage diagnostics, not pointwise tissue-motion labels.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from diag_breathing_leak import peak, video_fps
from endosim.eval.align import align_trajectories
from endosim.eval.metrics import load_pose_txt

METHODS = {
    "EgoLite": "egolite_{o}_stereomis_full",
    "EgoLite + pi3 keyframes": "hybrid_{o}_stereomis_full",
    "DROID-SLAM": "droid_stereomis_full",
    "ORB-SLAM3 (endo)": "orb_e_c2w_stereomis_full",
    "8-point + chaining": "eight_stereomis_full",
    "AF-SfMLearner": "afsfm_stereomis_full",
    "EndoDAC": "endodac_stereomis_full",
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ours",default="r_v2")
    ap.add_argument("--list",default="lists/stereomis.txt")
    ap.add_argument("--window-s",type=float,default=20.0)
    ap.add_argument("--max-gt-spread",type=float,default=1.0)
    ap.add_argument("--out",default="results/egolite/static_leak.json")
    args=ap.parse_args()
    seqs=[x.strip() for x in open(args.list,encoding="utf-8") if x.strip() and not x.startswith("#")]
    rows=[]
    for d in seqs:
        sid=os.path.basename(d.rstrip("/\\"))
        gt=load_pose_txt(os.path.join(d,"pose_c2w.txt"))
        fps=video_fps(d); W=int(args.window_s*fps); c=gt[:,:3,3]
        if len(gt)<=W: continue
        best,a0=None,0
        for a in range(0,len(gt)-W,max(1,W//10)):
            q=c[a:a+W]; spread=float(np.linalg.norm(q-q.mean(0),axis=1).max())
            if best is None or spread<best: best,a0=spread,a
        row={"seq":sid,"gt_spread_mm":best,"start":a0,"fps":fps}
        for name,tag in METHODS.items():
            p=os.path.join("results/sota",tag.format(o=args.ours),f"{sid}_est_c2w.txt")
            if not os.path.exists(p): continue
            est=np.loadtxt(p).reshape(-1,4,4)
            if len(est)!=len(gt) and abs(len(est)-len(gt))>2: continue
            n=min(len(est),len(gt))
            n=min(len(est),len(gt))
            aligned,(scale,_,_)=align_trajectories(est[:n],gt[:n],True)
            if not np.isfinite(scale) or scale<=0: continue
            e=aligned[:,:3,3]-aligned[0,:3,3]  # already in GT mm after Sim(3) alignment
            seg=e[a0:a0+W]
            if len(seg)<W//2: continue
            e0=seg-seg.mean(0); axis=np.linalg.svd(e0,full_matrices=False)[2][0]
            f,r=peak(e0@axis,fps)
            row[name]={"spread_mm":float(np.linalg.norm(e0,axis=1).max()),"peak_hz":f,"resp_power":r}
        rows.append(row)
        print(sid,round(best,3),{k:round(v["spread_mm"],2) for k,v in row.items() if isinstance(v,dict)},flush=True)
    rows=[r for r in rows if r["gt_spread_mm"]<args.max_gt_spread]
    print(f"{len(rows)} near-static sequences (GT spread < {args.max_gt_spread} mm)")
    summ={}
    for name in METHODS:
        v=[r[name] for r in rows if name in r]
        if v: summ[name]={"n":len(v),"median_spread_mm":float(np.median([x["spread_mm"] for x in v])),
                          "mean_spread_mm":float(np.mean([x["spread_mm"] for x in v])),
                          "median_resp_power":float(np.median([x["resp_power"] for x in v]))}
    summ["ground truth"]={"n":len(rows),"median_spread_mm":float(np.median([r["gt_spread_mm"] for r in rows]))}
    print(json.dumps(summ,indent=1))
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    json.dump({"summary":summ,"rows":rows},open(args.out,"w",encoding="utf-8"),indent=1)
if __name__=="__main__": main()
