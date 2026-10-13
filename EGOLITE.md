# EgoLite

Code, frozen cohort lists, per-sequence result ledgers and the trained checkpoint for the manuscript

**EgoLite: separating camera motion from tissue motion in monocular endoscopy by anchor-weighted differentiable pose solving** (under review).

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
gh release download egolite-v1.0 --repo liranyang123456-commits/EndoEgoSim --pattern egolite_r_v2.pth
python -c "import hashlib;print(hashlib.sha256(open('egolite_r_v2.pth','rb').read()).hexdigest())"
```

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
manuscript. In a re-run of three sequences from this repository, the per-sequence ATE differed from the ledger by at most
0.03 mm (GPU kernels are not bit-deterministic).

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

## Citation and licence

Manuscript under review; please cite the repository until it is published. Code: MIT (see `LICENSE`). RAFT-large and
ResNet-18 weights used for initialisation come from torchvision under their own licences.
