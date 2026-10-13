"""Re-evaluate r_v2 under its declared fixed 50-mm depth unit."""
import json, numpy as np,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scipy.stats import wilcoxon
from endosim.eval.metrics import evaluate_trajectory,rpe,load_pose_txt
ROOT=Path('results/sota/egolite_r_v2_stereomis_full'); droid=Path('results/sota/droid_stereomis_full')
seqs=[x.strip() for x in open('lists/stereomis_half.txt',encoding='utf-8') if x.strip() and not x.startswith('#')]
rows=[]
for s in seqs:
 sid=Path(s).name; gt=load_pose_txt(str(Path(s)/'pose_c2w.txt'))
 est=np.loadtxt(ROOT/f'{sid}_est_c2w.txt').reshape(-1,4,4)
 if len(gt)!=len(est):raise ValueError((sid,len(gt),len(est)))
 est50=est.copy();est50[:,:3,3]*=50.0
 res=evaluate_trajectory(est50,gt)
 rows.append({'seq_id':sid,'n':len(gt),'ate_se3_mm':res['ate_se3']['rmse'],'ate_sim3_mm':res['ate_sim3']['rmse'],'fitted_sim3_scale_after_x50':res['ate_sim3']['scale'],'rpe_metric':{str(g):rpe(est50,gt,g) for g in [1,5,10]},'rpe_raw_model_units':{str(g):rpe(est,gt,g) for g in [1,5,10]}})
 print(sid,round(res['ate_se3']['rmse'],3),round(res['ate_sim3']['scale'],3),flush=True)
# DROID remains up-to-scale and is reported only with Sim3 unless an independently calibrated metric scale exists.
a=np.array([r['ate_se3_mm'] for r in rows]); b=np.array([r['ate_sim3_mm'] for r in rows]);
out={'unit_basis':'r_v2 checkpoint args abs_depth=true; training defines DEPTH_UNIT_MM=50 and normalizes GT pair translation by 50; convert chained estimator translations from depth units to mm before SE3 metric ATE/RPE.','n_seq':len(rows),'ate_se3_mm_mean':float(a.mean()),'ate_se3_mm_median':float(np.median(a)),'ate_se3_paired_wilcoxon_vs_droid_not_applicable':None,'ate_sim3_mm_mean':float(b.mean()),'ate_sim3_mm_median':float(np.median(b)),'sim3_scale_after_x50_median':float(np.median([r['fitted_sim3_scale_after_x50'] for r in rows])),'sim3_scale_after_x50_range':[float(min(r['fitted_sim3_scale_after_x50'] for r in rows)),float(max(r['fitted_sim3_scale_after_x50'] for r in rows))],'rpe_metric_mean_mm':{str(g):float(np.mean([r['rpe_metric'][str(g)]['trans_mm_mean'] for r in rows])) for g in [1,5,10]},'sequences':rows}
Path('results/egolite/stereomis_rv2_fixed50_unit_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='sequences'},indent=2))
