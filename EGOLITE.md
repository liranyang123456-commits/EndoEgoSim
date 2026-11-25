# EgoLite

Code, frozen cohort lists, per-sequence result ledgers and the trained checkpoint for the manuscript

**EgoLite: separating camera motion from tissue motion in monocular endoscopy by differentiable pose solving with learned pixel weights** (under review).

EgoLite is a 20.9 M-parameter monocular network. For a frame pair it predicts per-pixel flow, depth, motion layers
(static, deforming tissue, instrument) and a flow-error scale; a differentiable damped Gauss-Newton solver on se(3), weighted
by the learned influence field, turns these into the camera motion. The model runs in about 45 ms per frame pair at batch
size 1 on an RTX 5090 laptop GPU (unoptimised PyTorch).

> **Claim boundary.** The separation of camera and tissue motion is shown in simulation only. On real video the accuracy
> depends on the dataset: on StereoMIS DROID-SLAM and Endo-FASt3r are more accurate, on C3VD Endo-FASt3r and EndoDAC are.
> EgoLite gives no calibrated metric scale. It is a research prototype, not a medical device, and must not be used for
> clinical decisions.

## What is in this release

| Path | Contents |
|------|----------|
| `endosim/lite/` | Model (`egolite.py`), frame loading and pseudo-height (`data.py`), sequence inference (`infer.py`), optional window bundle adjustment (`window_ba.py`), optional MiDaS prior (`depth_prior.py`, not used by the reported checkpoint) |
| `scripts/train_egolite.py` | Training |
| `scripts/eval_egolite.py` | Sim(3) ATE and RPE under the uniform at-most-64-frame protocol (`--max-frames 0` for full videos) |
| `scripts/eval_decoupling.py` | Image-space separation of camera and tissue motion (tracked-displacement error, AUROC, direction cosine) |
| `scripts/run_gn_sensitivity.py`, `eval_label_ablation.py`, `run_pairlab_eval.py`, `run_pairlab_full_chain.py`, `paired_bootstrap_ci.py`, `rotation_rpe_summary.py`, `eval_static_leak.py` | Analyses reported in the manuscript |
| `scripts/audit_egolite_fixed50_scale.py`, `results/egolite/stereomis_rv2_fixed50_unit_audit.json` | Re-evaluation of the full StereoMIS videos in the fixed 50 mm depth unit (the script needs the `*_est_c2w.txt` files written by `eval_egolite.py`, which are not shipped) |
| `scripts/eval_ar_registration.py`, `results/ar_reg/ar_registration_summary.json` | Augmented-reality registration test: registering a tissue patch with and without ego-motion removal (see below) |
| `scripts/bench_egolite_streaming.py`, `bench_baselines_latency.py`, `make_accuracy_latency_figure.py` | Latency and accuracy-latency figure (the baseline wrappers expect local clones of the official baseline code; edit the paths at the top of `egomotion_baselines.py` and `eval_slam.py`) |
| `scripts/egolite_pipelines/` | PowerShell pipelines of the reported runs (stage 6 training and evaluation, ablations) |
| `configs/egolite_r_v2_args.json`, `results/egolite/chain/*.json` | Configuration of the reported checkpoint and the arguments of every training stage |
| `configs/abs.json`, `endosim/dataset/r2s_generator.py`, `scripts/label_r2s_scenes.py` | EndoEgoSim-Abs generator config, SCARED-Real2Sim generator and labels |
| `lists/` | Frozen cohorts: `simtest92`, `sim_confirmatory265`, `sim_confirmatory_extension93`, `abs_test94`, `r2s_test`, `scared_official`, `c3vd_real`, `stereomis`, `egomo_gtraj` and the validation lists |
| `results/sota/egolite_r_v2_*/summary.json` | Per-sequence ledgers of the reported checkpoint and its inference-time ablations, label ablations and seeds |
| `results/sota/*.json` | Paired bootstrap intervals, rotation RPE, Gauss-Newton step sensitivity, label ablation summaries |
| `results/latency/*.json` | Latency measurements (batch size 1 and 16, several GPUs) |
| `tests/test_egolite_smoke.py` | CPU smoke test of the model and of the checkpoint |

## Checkpoint

The reported model (stage 6, `r_v2`) is the release asset **`egolite_r_v2.pth`** of
[release `egolite-v1.0`](https://github.com/liranyang123456-commits/EndoEgoSim/releases/tag/egolite-v1.0)
(83,625,117 bytes). It is a PyTorch file with the keys `model` (state dict), `iter` and `args` (the model configuration).

```
SHA-256  f2c6def4a6e6c81d01a95a0c59150047a661c2f5537486cf30efaf967b6266a4
```

```bash
curl -L -o egolite_r_v2.pth https://github.com/liranyang123456-commits/EndoEgoSim/releases/download/egolite-v1.0/egolite_r_v2.pth
python -c "import hashlib;print(hashlib.sha256(open('egolite_r_v2.pth','rb').read()).hexdigest())"
```

(`gh release download egolite-v1.0 --pattern egolite_r_v2.pth` works as well.)

The weights were trained on simulated and real-to-sim scenes only (see below). They are released for research use.

## Quick start (one frame pair)

```bash
pip install -r requirements-egolite.txt
```

```python
import torch
from endosim.lite.data import load_frames
from endosim.lite.egolite import build_from_args

ck = torch.load("egolite_r_v2.pth", map_location="cpu")
model = build_from_args(dict(ck["args"]), pretrained=False)
model.load_state_dict(ck["model"])
model = model.cuda().eval()

imgs, phs, _ = load_frames(["frame0.png", "frame1.png"])     # 256 x 320 RGB in [0, 1] and pseudo-height maps
img = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda()
ph = torch.from_numpy(phs)[:, None].cuda()
K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]).cuda()   # intrinsics scaled to 256 x 320 (see data.scaled_K)
with torch.no_grad():
    out = model(img[:1], img[1:], K[None], ph[:1], ph[1:])
R, t = out["R"][0], out["t"][0]      # relative camera motion of the pair
```

Translations are in units of 50 mm, the fixed depth unit used in training; the model has no calibrated metric scale.
The `rpe_*` entries in the EgoLite ledgers are in this unit and must be multiplied by 50 to compare them with millimetres.
The Sim(3) ATE in the ledgers is scale-aligned and needs no conversion. `endosim/lite/egolite.py::chain` composes a
trajectory from the pair motions.

## Evaluate

```bash
python scripts/eval_egolite.py --list lists/simtest92.txt --ckpt egolite_r_v2.pth \
    --tag egolite_r_v2_simtest --out results/sota
```

The evaluation needs the generated sequences named in the list. List entries are the author's local paths; only the last
path component (`seq_<seed>`) identifies a sequence, because the generator derives the split and the content of
`seq_<seed>` from the seed. The shipped ledger `results/sota/egolite_r_v2_simtest/summary.json` holds the numbers of the
manuscript. In a re-run of four sequences from a fresh clone with the downloaded checkpoint, the per-sequence ATE differed
from the ledger by at most 0.06 mm (below 0.4 %); GPU kernels are not bit-deterministic.

## Augmented-reality registration test

```bash
python scripts/eval_ar_registration.py --root sim_data_abs/test --ckpt egolite_r_v2.pth
```

On the EndoEgoSim-Abs test sequences in which the camera travels at least 40 mm and respiration has an amplitude of at
least 4 mm (nine sequences with visible deforming tissue, 163 updates), a rigid patch on deforming tissue is registered
between frames that are far enough apart for the camera-induced displacement to exceed an 18 px search window. All
methods register the same patch; they differ only in the prior. Median over sequences of the per-sequence median error:

| Registration | Prior | Error (px) | Within 10 px |
|---|---|---:|---:|
| Pyramidal LK, then PnP | none | 20.3 | 37% |
| Dense optical flow, then PnP | none | 20.1 | 38% |
| Depth ICP (simulator depth) | previous pose | 25.9 | 20% |
| Local registration | previous position | 23.4 | 40% |
| Homography transfer | none | 23.0 | 46% |
| Local registration | whole-frame homography | 17.1 | 50% |
| Local registration | ORB ego-motion, true translation magnitude | 5.1 | 59% |
| Local registration | EgoLite ego-motion | **3.4** | **72%** |
| Local registration | true camera motion (reference) | 2.1 | 80% |

Paired Wilcoxon tests over the nine sequences against EgoLite give p = 0.004 for the four methods without ego-motion
separation, 0.039 for both homography rows, 0.027 for the ORB ego-motion and 0.25 for the true camera motion. The
summary file also holds the window sweep (10-34 px) and the per-sequence medians. The patch is simulated tissue, not a
preoperative model, and the test does not measure clinical overlay accuracy.

## Data

* **EndoEgoSim, EndoEgoSim-Abs.** Generated with `scripts/generate_dataset.py` (`configs/*.json`, `configs/abs.json` for
  the Abs sets, which add respiration, hover motion and absolute-motion ground truth). Exact regeneration of the
  manuscript sequences needs the appearance banks below; `configs/procedural_review.json` gives structurally similar scenes
  with procedural textures and different appearance.
* **SCARED-Real2Sim.** Built from SCARED by `endosim/dataset/r2s_generator.py` (set `SCARED_ROOT`) and labelled with
  `scripts/label_r2s_scenes.py`. SCARED, StereoMIS, C3VD, CholecSeg8k and HyperKvasir must be obtained from their providers.
* **Appearance banks and ethics.** The texture pool of the training scenes combines frames from CholecSeg8k, HyperKvasir,
  SCARED and a private, de-identified set of surgical videos provided by Henan Provincial People's Hospital after the
  hospital's review of the data. These images are used only as textures for rendering simulated scenes. None of them is
  redistributed here, and the checkpoint contains network weights only.

## Training chain of the reported checkpoint

Six stages, 55,400 iterations in total (`results/egolite/chain/<stage>_args.json`, `python scripts/_init_chain.py <run>`):

| # | Run | Iterations | Batch x clip | LR | Data (EndoEgoSim : Abs : Real2Sim) | Parameters |
|---|-----|-----------:|-------------:|---:|------------------------------------|-----------:|
| 1 | `v1` | 20,000 | 6 x 6 | 2e-4 | EndoEgoSim, from scratch | 4.6 M |
| 2 | `v2` | 10,000 | 5 x 8 | 1e-4 | EndoEgoSim | 4.6 M |
| 3 | `v3abs` | 10,000 | 5 x 8 | 1e-4 | EndoEgoSim, fixed 50 mm depth unit | 4.6 M |
| 4 | `s_big_ft` | 6,600 | 3 x 8 | 1e-4 | EndoEgoSim, RAFT-large fine-tuned, `--head-extra 6 --ctx` | 20.9 M |
| 5 | `r_v1` | 6,600 | 3 x 8 | 1e-4 | 1 : 1 : 2 | 20.9 M |
| 6 | `r_v2` (released) | 2,200 | 3 x 8 | 5e-5 | 2 : 1 : 1 | 20.9 M |

Seed 0 throughout. `scripts/egolite_pipelines/pipeline_r2.ps1` is the exact command of stage 6 and its evaluations;
`scripts/run_pairlab_full_chain.py` repeats stages 5 and 6 with per-pair motion masks (label ablation).

## Revision Benchmarks & Ledgers (`egolite-v1.2`)

Release `egolite-v1.2` adds all reproducible scripts (`scripts/rev_*.py`) and JSON result ledgers (`results/revision_vrih/*.json`) for the revised manuscript:

* **Source-disjoint real-video subsets (`r1_5_disjoint_lineage_audit.json`, `scripts/rev_audit_disjoint_lineage.py`)**: specimen-disjoint SCARED datasets 2-3 (8 sequences) and mesh/mould-disjoint C3VD subsets under Protocols A and B (Table S1 / Table R5).
* **Same-corpus 12.5-M PoseNet, retrained module/loss ablations, and single-stage 3-seed training**:
  * `scripts/rev_fix_stage1_posenet.py` -> `r1_7_samedata_posenet_fixed.json` (12.5-M ResNet-18 PoseNet trained on the exact 2:1:1 synthetic + R2S mixture; Table 3, Table S4, Table R7).
  * `scripts/rev_fix_stage2_single_stage.py` -> `r1_4_single_stage_generic_raft.json` (3-seed single-stage from-scratch training with generic torchvision RAFT-large; Table S2, Table R4a).
  * `scripts/rev_local_step4_fast.py` -> `r1_4_seeds_and_stages_real.json` (six-stage progression and 3-seed stage-6 variance across all 8 cohorts; Table S2, Table R4a).
  * `scripts/rev_run_local_retrained_fast.py`, `scripts/rev_finish_instw_and_ar_posenet.py`, `scripts/rev_complete_remaining_audits.py` -> `r1_3_4_6_7_retrained_ablations.json` and `r1_6_r2_3_stereomis_fullvideo_ablations.json` (matched 1,500-step stage-6 module & loss-term ablations and full-video Protocol-B StereoMIS ablations; Table 6b, Table S4, Tables R4b/R6b).
  * `scripts/rev_fix_stage3_lossless_and_calib.py` & `scripts/rev_bench_solver_batch1.py` -> `r1_6_r2_3_table6_lossless.json` and `r1_3_design_choices_lossless.json` (lossless-PNG Table 6a, solver stride/occlusion ablations, and synchronized fp32 batch-1 CUDA-event solver timings; Table 6a, Table S3, Tables R3/R6a).
* **Expanded AR registration, static-hold respiratory leakage, and windowed SE(3)/Sim(3) scale drift**:
  * `scripts/rev_eval_stereomis_real_ar.py`, `scripts/rev_run_stir_ar_cpu.py`, `scripts/rev_run_droid_9abs.py`, `scripts/rev_run_droid_stir20.py`, `scripts/rev_fix_stage4_ar_seeded.py` -> `r2_2_sim_ar_expanded.json`, `r2_1_stereomis_real_ar_clean.json`, `r2_1_stir_real_ar.json` (seeded `cv2.setRNGSeed(0)` AR landmark registration across 9 EndoEgoSim-Abs sequences, 10 in vivo StereoMIS sequences, and 20 in vivo STIR 2024 clips; Table 9 / Table R9, Fig. 17).
  * `scripts/rev_eval_stereomis_static_and_scale.py` -> `r2_1_r1_2_stereomis_static_and_scale.json` ($n=23$ automated near-static 20-s StereoMIS holds across 7 sequences, $0.1$-$0.6$ Hz respiratory power fractions, and piecewise 60/30/15/10-s global-scale SE(3) vs. Sim(3) drift diagnostics across all 11 StereoMIS sequences; Sections 5.2 & 5.5, Fig. 15).
* **Monocular depth accuracy, motion-layer IoU, and GFLOPs/CUDA-Graph profiling**:
  * `scripts/rev_eval_depth_and_mechanisms.py` (+ `scripts/rev_fix_stage3_lossless_and_calib.py`) -> `r1_1_label_learning_audit.json` and `r1_2_depth_accuracy.json` (Table S5, Table S6, Tables R1/R2).
  * `scripts/rev_count_flops_cpu.py`, `scripts/rev_bench_latency_and_flops.py`, `scripts/rev_bench_cudagraphs.py` -> `r1_8_flops_and_ops_cpu.json` and `r1_8_latency_and_flops_audit.json` (153.11 GFLOPs, 2,877 GPU kernels, and 11.82 ms / 84.6 fps batch-1 static CUDA Graph timings; Section 5.3, Table R8).
* **Optional path overrides**: all scripts resolve `ROOT` relative to their own location and honour `EGOLITE_ROOT`, `EGOLITE_SIM_TRAIN_DIR`, `EGOLITE_HAMLYN_R2S_DIR`, `EGOLITE_STIR_ROOT`, `EGOLITE_C3VD_RAW_ROOT`, `DROID_SLAM_DIR`, and `DROID_WEIGHTS`.

## Citation and licence

Manuscript under review; please cite the repository until it is published. Code: MIT (see `LICENSE`). RAFT-large and
ResNet-18 weights used for initialisation come from torchvision under their own licences.
