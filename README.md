# EndoEgoSim / MD-VGGT

Code and frozen evaluation ledgers for the manuscript:

**Disentangling Camera and Scene Motion for Ego-Motion Estimation in Monocular Endoscopy**

Repository: https://github.com/liranyang123456-commits/EndoEgoSim

> **Claim boundary.** We do **not** claim a universal real-domain SOTA.
> Primary simulation evidence is protocol-matched Sim(3) ATE. The development
> route threshold is post hoc; frozen confirmatory sets are mixed (core mean
> favourable; extension mean trails Reloc3r-512 on the mean but favours
> MD-VGGT-R on the median and on the task-level reconstruction metric).
> StereoMIS (64-frame) is the independent external real protocol.

## Headline numbers (sequence-macro Sim(3) ATE, mm)

| Protocol | Method | Mean | Notes |
|----------|--------|------|-------|
| `lists/simtest92.txt` (n=92) | **MD-VGGT-R** (τ=0.12) | **3.49** | post-hoc τ on this list |
| same | DROID-SLAM | 6.04 | full coverage |
| frozen core (n=265) | MD-VGGT-R | 2.70 | vs DROID 3.26; Holm n.s. |
| frozen extension (n=93) | MD-VGGT-R | 23.15 | Reloc3r-512 mean **22.77** |
| StereoMIS 64-frame | MD-VGGT-G | 13.44 | vs DROID 27.68 |
| EGO-Mo gtraj (n=18, global-GT) | MD-VGGT-R | **13.54** | lowest; vs VGGT p=0.0028 |

**Task-level reconstruction (Chamfer, mm)**: MD-VGGT-R 11.46 (simtest92) / 6.51 (frozen core) / 29.96 (extension) — lowest on every cohort; on the extension it reverses the ATE-mean ordering (Reloc3r-512 44.32, p=5.8e-06).

**Ablations** (simtest92, v7 baseline 3.495): +pseudo-height head 3.39 (frozen core 2.49, p=0.027; extension 20.91 surpasses Reloc3r, p=4.9e-06); +motion-layer head 3.39 (instrument/static IoU 0.50/0.65); +input channel 4.07 (harmful); +GN pose graph 3.51 (neutral).

Ledgers: `results/sota/*/summary.json`, `results/sota/confirmatory_analysis.json`.

## For reviewers

| Item | Location |
|------|----------|
| Frozen eval lists | `lists/simtest92.txt`, `sim_confirmatory265.txt`, `sim_confirmatory_extension93.txt` |
| Procedural-only config | `configs/procedural_review.json` (`texture_source=procedural`, `barrel_prob=0`) |
| Build reviewer pack | `python scripts/build_reviewer_procedural_pack.py --generate-demo 8` |
| Paper package (local) | `submission_archive/MIA_submission_*_candidate.zip` |

Private hospital textures and third-party appearance banks are **not** redistributed.
Procedural generation does not require those assets.

```bash
# Small procedural demo (no real texture bank)
python scripts/generate_dataset.py \
  --config configs/procedural_review.json \
  --n-seq 8 --seed-start 900001 --workers 4 --no-bank \
  --out review_demo_data
```

## Repository layout

| Path | Contents |
|------|----------|
| `endosim/` | Generator, rasterizer, metrics, `geometry/pose_graph.py`, `pseudoheight.py`, `eval/` (MD-VGGT infer, decouple, static support) |
| `scripts/` | Generation, MD-VGGT eval, reconstruction/motion-layer/board-pose eval, EGO-Mo converters, leaderboard/trajectory plots, packaging |
| `configs/` | Default / hard / keyframe / lowref / href / disaster / ego_protocol / **procedural_review** |
| `lists/` | Frozen identities used in the paper + EGO-Mo real-capture lists |
| `results/sota/` | Summary JSON ledgers (not full RGB) |
| `mia_paper/` | Manuscript sources (local workspace; may not be on GitHub) |

## Real-capture evaluation (EGO-Mo)

A global-reference rig (fixed 4K global camera + checkerboard on the stereo endoscope + scene checkerboard + dual IMU) gives absolute 6-DoF ground truth. Converters: `scripts/convert_egomo_to_eval.py`, `scripts/convert_gtraj_to_eval.py`. Evaluation: `scripts/eval_mdvggt_infer.py`, `scripts/eval_slam.py`, `scripts/eval_board_pose.py` (dual-rigid-body: camera + scene-board trajectories).

## Naming

- **MD-VGGT**: trained adaptation of VGGT-1B
- **MD-VGGT-G**: same checkpoint, global window only
- **MD-VGGT-R**: same checkpoint + collapse-aware local route
- Internal dirs `ours_v7_*` are ledger tags only, not paper method names

## License / third-party data

Code is for research reproduction.
Obtain C3VD, SCARED, CholecSeg8k, and HyperKvasir from their providers.
Public model weights remain under their original licenses.

## Contact

Corresponding authors listed in the manuscript (Henan University of Technology / Beihang University).
