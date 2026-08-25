# EndoEgoSim

Code for **monocular endoscopic camera egomotion**: a controllable simulation factory (EndoEgoSim), a geometric pose proxy (Mesh-RTS), and motion-decomposition fine-tuning of VGGT (MD-VGGT).

Paper: *Differentiable Pseudo-3D Mesh Alignment for Endoscopic Camera Pose under Deformable Tissue* (submitted to Medical Image Analysis).

**This repository does not claim a real-domain SOTA.** On the official simulation list `lists/simtest92.txt` ($n=92$), the single learned model **MD-VGGT-v3** reaches **5.74 mm** mean Sim(3) ATE, below the VGGT / MASt3R / DUSt3R / Endo3R / $\pi^3$ checkpoints we ran. Mesh-RTS-v3 is **10.47 mm** (single pairwise path, slightly below 8-point at 10.52 mm). Official SCARED left-camera videos remain metres-scale because native hops are already 12–40 cm.

## What is included

| Path | Contents |
|---|---|
| `endosim/` | Dataset generator, rasterizer, Umeyama ATE/RPE, real-data indexers |
| `scripts/` | Generation, SOTA eval, Mesh-RTS, $\pi^3$, fusion gates |
| `configs/` | Default / hard / keyframe generation configs |
| `lists/` | `simtest92.txt` and reference-fraction splits |
| `docs/` | Design notes and the performance table used in the paper |
| `results/sota/*/summary.json` | Per-run macros that match the paper tables |

Not included (too large or third-party): rendered `sim_data/`, C3VD/SCARED pixels, VGGT/$\pi^3$ weights.

## Metrics

`endosim/eval/metrics.py` reports:

- **ATE Sim(3)**: RMSE of translation (mm) after Umeyama 7-DoF alignment (headline, monocular / SCARED convention).
- **ATE SE(3)**: same with scale fixed.
- **RPE$_1$**: mean translational relative pose error (mm), **not** RMSE.

Degenerate Sim(3) solutions with vanishing scale variance are forced to scale 1.

## Reproduce the 92-sequence table

```bash
# Identity / 8-point / RGB-D PnP
python scripts/baseline_classical.py --method eight --seq-list lists/simtest92.txt --out results/sota

# Published feed-forward checkpoints (needs local weights)
python scripts/baseline_sota.py --method vggt --seq-list lists/simtest92.txt --out results/sota
python scripts/eval_slam.py --method pi3 --seq-list lists/simtest92.txt --out results/sota

# Mesh-RTS v3 (single path)
python scripts/eval_meshrts.py --variant v3 --seq-list lists/simtest92.txt --out results/sota

# Rebuild the markdown table from summary.json files
python scripts/compare_sota.py
```

C3VD and SCARED: index official videos with `scripts/index_real_datasets.py` (paths stay local; pixels are not copied), then run the same evaluators with `--protocol` as in `docs/06_改进方案.md`.

## Generate EndoEgoSim

```bash
python scripts/generate_dataset.py --config configs/default.json --n-seq 100 --out sim_data
python scripts/verify_gt.py --seq sim_data/train/seq_00000101
```

GPU rasterization (`--gpu`, nvdiffrast) is optional. The released 2,400-sequence set is about 115 GB and is not uploaded here.

## Citation

```bibtex
@article{li2026endoegosim,
  title={Differentiable Pseudo-3D Mesh Alignment for Endoscopic Camera Pose under Deformable Tissue},
  author={Li, Ranyang and Zhu, Ziyu and Guo, Xiaodong and Wei, Nan and Liu, Wufeng and Fan, Chao and Pan, Junjun},
  journal={Medical Image Analysis},
  note={under review},
  year={2026}
}
```

## Licence

MIT for this repository's original code. VGGT, DUSt3R, MASt3R, Endo3R, and $\pi^3$ remain under their own licences. C3VD and SCARED remain under their original data agreements.
