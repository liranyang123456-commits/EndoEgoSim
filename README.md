# EndoEgoSim / MD-VGGT

Code and frozen evaluation ledgers for the Medical Image Analysis manuscript:

**Disentangling Camera and Scene Motion in Monocular Endoscopy: EndoEgoSim and Motion-Decomposed VGGT**

Repository: https://github.com/liranyang123456-commits/EndoEgoSim

> **Claim boundary.** We do **not** claim a universal real-domain SOTA.
> Primary simulation evidence is protocol-matched Sim(3) ATE. The development
> route threshold is post hoc; frozen confirmatory sets are mixed (core mean
> favourable; extension mean trails Reloc3r-512). StereoMIS (64-frame) is the
> independent external real protocol.

## Headline numbers (sequence-macro Sim(3) ATE, mm)

| Protocol | Method | Mean | Notes |
|----------|--------|------|-------|
| `lists/simtest92.txt` (n=92) | **MD-VGGT-R** (τ=0.12) | **3.49** | post-hoc τ on this list |
| same | DROID-SLAM | 6.04 | full coverage |
| frozen core (n=265) | MD-VGGT-R | 2.70 | vs DROID 3.26; Holm n.s. |
| frozen extension (n=93) | MD-VGGT-R | 23.15 | Reloc3r-512 mean **22.77** |
| StereoMIS 64-frame | MD-VGGT-G | 13.44 | vs DROID 27.68 |
| StereoMIS full-rate sliding | MD-VGGT-G | 29.93 | composition + denser frames |

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
| `endosim/` | Generator, rasterizer, metrics |
| `scripts/` | Generation, MD-VGGT eval, confirmatory analysis, packaging |
| `configs/` | Default / hard / keyframe / **procedural_review** |
| `lists/` | Frozen identities used in the paper |
| `results/sota/` | Summary JSON ledgers (not full RGB) |
| `mia_paper/` | Manuscript sources (local workspace; may not be on GitHub) |

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
