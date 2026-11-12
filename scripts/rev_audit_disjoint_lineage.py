"""Pure-CPU audit for VRIH R1.5 (strictly disjoint SCARED and C3VD subsets) and R1.4/R1.2 existing ledgers.

1. Scans sim_data/train, sim_data_abs/train, sim_data_r2s/train meta.json to verify exact training asset lineage
   (which C3VD meshes, which SCARED keyframes, which texture sources were actually used).
2. Splits SCARED (11 seqs) into:
   - overlapped / same-specimen (dataset_1: k1 in texture bank, k2-k3 same porcine specimen)
   - strictly specimen- & source-disjoint (datasets 2 & 3: 8 sequences, zero overlap with d1 texture or d5-7 R2S)
   - also Endo-FASt3r out-of-training subset if applicable
   and computes Protocol A and Protocol B Sim(3) ATE + paired Wilcoxon p-values for all methods.
3. Splits C3VD (10 seqs) by mesh/trajectory lineage:
   - exact mesh+pose overlap (cecum_t1_a)
   - mold-anatomy overlap (cecum, sigmoid, transverse, descending)
   - strictly anatomy-mesh disjoint (c1_ascending_t4_v4: ascending colon has NO mesh in ORGAN_MESHES)
   and checks whether any other C3VD ascending sequences exist on disk!
"""
from __future__ import annotations
import collections
import glob
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(r"D:\ego_motiion_Camera")
OUT = ROOT / "results" / "revision_vrih"
OUT.mkdir(parents=True, exist_ok=True)


def scan_lineage():
    out = {}
    for split_dir in ("sim_data/train", "sim_data_abs/train", "sim_data_r2s/train"):
        p = ROOT / split_dir
        organs = collections.Counter()
        scenes = collections.Counter()
        tex_sources = collections.Counter()
        r2s_scenes = collections.Counter()
        n = 0
        for mp in p.glob("seq_*/meta.json"):
            n += 1
            m = json.loads(mp.read_text(encoding="utf-8"))
            scenes[str(m.get("scene_kind"))] += 1
            if m.get("organ"):
                organs[str(m.get("organ"))] += 1
            if m.get("texture_source"):
                tex_sources[str(m.get("texture_source"))] += 1
            if m.get("scene_name"):
                r2s_scenes[str(m.get("scene_name"))] += 1
            elif m.get("source_scene"):
                r2s_scenes[str(m.get("source_scene"))] += 1
        out[split_dir] = {
            "n_sequences": n,
            "scene_kinds": dict(scenes),
            "organs": dict(organs),
            "texture_sources": dict(tex_sources),
            "r2s_scenes_sample": dict(list(r2s_scenes.items())[:30]),
            "r2s_unique_prefixes": sorted({k.split("_k")[0] for k in r2s_scenes}),
        }
    return out


def load_per_seq_ate(tag: str) -> dict[str, float]:
    sp = ROOT / "results" / "sota" / tag / "summary.json"
    if not sp.is_file():
        return {}
    d = json.loads(sp.read_text(encoding="utf-8"))
    recs = d.get("records") or d.get("sequences") or d.get("per_sequence") or []
    res = {}
    for r in recs:
        sid = r.get("seq_id")
        ate = (r.get("ate_sim3") or {}).get("rmse")
        if sid and ate is not None and np.isfinite(ate):
            res[sid] = float(ate)
    return res


def paired_subset_stats(methods_tags: dict[str, str], ref_name: str, subset_ids: list[str]):
    vals = {m: load_per_seq_ate(tag) for m, tag in methods_tags.items()}
    ref = vals[ref_name]
    ids = [s for s in subset_ids if s in ref]
    table = {}
    for m, d in vals.items():
        common = [s for s in ids if s in d]
        if not common:
            continue
        arr = np.array([d[s] for s in common])
        rarr = np.array([ref[s] for s in common])
        p = None
        if m != ref_name and len(common) >= 5 and not np.allclose(arr, rarr):
            p = float(wilcoxon(arr, rarr).pvalue)
        table[m] = {
            "n": len(common),
            "mean_mm": float(arr.mean()),
            "median_mm": float(np.median(arr)),
            "wilcoxon_p_vs_egolite": p,
            "per_seq": {s: round(d[s], 3) for s in common},
        }
    return table


def main():
    lineage = scan_lineage()
    scared_all = [Path(x.strip()).name for x in (ROOT / "lists" / "scared_official.txt").read_text().splitlines() if x.strip()]
    scared_d23 = [s for s in scared_all if "_dataset_2_" in s or "_dataset_3_" in s]
    scared_no_d1k1 = [s for s in scared_all if s != "scared_dataset_1_keyframe_1"]

    c3vd_all = [Path(x.strip()).name for x in (ROOT / "lists" / "c3vd_full.txt").read_text().splitlines() if x.strip()]
    c3vd_no_exact = [s for s in c3vd_all if "cecum_t1_a" not in s]
    c3vd_ascending = [s for s in c3vd_all if "ascending" in s]

    # Also check raw C3VD folder on E: for any additional ascending sequences!
    raw_c3vd_asc = []
    c3vd_root = Path(r"E:\World_Agent_Enoscopy\datasets\C3VD")
    if c3vd_root.is_dir():
        for p in sorted(c3vd_root.rglob("pose.txt")):
            if "ascending" in str(p).lower():
                raw_c3vd_asc.append(str(p.parent))

    protA_scared = {
        "AF-SfMLearner": "afsfm_scared_u64",
        "EndoDAC": "endodac_scared_u64",
        "Endo-FASt3r": "endofast3r_scared_u64",
        "DROID-SLAM": "droid_scared_u64",
        "Pi3": "pi3_scared_u64",
        "Reloc3r-512": "reloc3r_scared_u64",
        "EgoLite": "egolite_r_v2_scared",
        "EgoLite (no R2S)": "egolite_abl_nor2s_scared",
    }
    protB_scared = {
        "AF-SfMLearner": "afsfm_scared_full",
        "EndoDAC": "endodac_scared_full",
        "Endo-FASt3r": "endofast3r_scared_full",
        "ORB-SLAM3": "orb_e_c2w_scared_full",
        "DROID-SLAM": "droid_scared_full",
        "EgoLite": "egolite_r_v2_scared_full",
    }
    protA_c3vd = {
        "AF-SfMLearner": "afsfm_c3vd_u64",
        "EndoDAC": "endodac_c3vd_u64",
        "Endo-FASt3r": "endofast3r_c3vd_u64",
        "DROID-SLAM": "droid_c3vd_u64",
        "Pi3": "pi3_c3vd_u64",
        "Reloc3r-512": "reloc3r_c3vd_u64",
        "EgoLite": "egolite_r_v2_c3vd",
    }
    protB_c3vd = {
        "AF-SfMLearner": "afsfm_c3vd_full",
        "EndoDAC": "endodac_c3vd_full",
        "Endo-FASt3r": "endofast3r_c3vd_full",
        "ORB-SLAM3": "orb_e_c2w_c3vd_full",
        "8-point": "eight_c3vd_full",
        "DROID-SLAM": "droid_c3vd_full",
        "EgoLite": "egolite_r_v2_c3vd_full",
    }

    report = {
        "lineage": lineage,
        "raw_c3vd_ascending_on_disk": raw_c3vd_asc,
        "scared_subsets": {
            "all_11": {"A": paired_subset_stats(protA_scared, "EgoLite", scared_all),
                       "B": paired_subset_stats(protB_scared, "EgoLite", scared_all)},
            "no_d1k1_10": {"A": paired_subset_stats(protA_scared, "EgoLite", scared_no_d1k1),
                           "B": paired_subset_stats(protB_scared, "EgoLite", scared_no_d1k1)},
            "specimen_disjoint_d2_d3_8": {"A": paired_subset_stats(protA_scared, "EgoLite", scared_d23),
                                          "B": paired_subset_stats(protB_scared, "EgoLite", scared_d23)},
        },
        "c3vd_subsets": {
            "all_10": {"A": paired_subset_stats(protA_c3vd, "EgoLite", c3vd_all),
                       "B": paired_subset_stats(protB_c3vd, "EgoLite", c3vd_all)},
            "no_cecum_t1_a_9": {"A": paired_subset_stats(protA_c3vd, "EgoLite", c3vd_no_exact),
                                "B": paired_subset_stats(protB_c3vd, "EgoLite", c3vd_no_exact)},
            "ascending_only": {"A": paired_subset_stats(protA_c3vd, "EgoLite", c3vd_ascending),
                               "B": paired_subset_stats(protB_c3vd, "EgoLite", c3vd_ascending)},
        },
    }
    out_file = OUT / "r1_5_disjoint_lineage_audit.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Lineage:", json.dumps(lineage, indent=2))
    print("Raw C3VD ascending dirs:", raw_c3vd_asc)
    for split_name, sub in report["scared_subsets"].items():
        print(f"\n--- SCARED {split_name} ---")
        for prot in ("A", "B"):
            print(f"  Protocol {prot}: " + ", ".join(
                f"{m}: {v['mean_mm']:.2f} (med {v['median_mm']:.2f}, p={v['wilcoxon_p_vs_egolite']})"
                for m, v in sub[prot].items()))
    for split_name, sub in report["c3vd_subsets"].items():
        print(f"\n--- C3VD {split_name} ---")
        for prot in ("A", "B"):
            print(f"  Protocol {prot}: " + ", ".join(
                f"{m}: {v['mean_mm']:.2f} (med {v['median_mm']:.2f}, p={v['wilcoxon_p_vs_egolite']})"
                for m, v in sub[prot].items()))


if __name__ == "__main__":
    main()
