"""Fleet QA task: independent supplementary statistics for the MIA resubmission.

Runs on a fleet worker (no GPU needed). Reads a bundled copy of
results/sota per-sequence summaries and produces:

1. Effect sizes (rank-biserial correlation) and median-difference bootstrap
   CIs for the frozen cohorts (reviewer R2-2).
2. Stratified paired tests on the extension cohort (low/mid/high reference).
3. A number-consistency check between key manuscript values and the JSON
   ledgers.
4. A language-pattern scan of the manuscript (sentence length, passive
   voice density, paragraph-initial word frequency).

Outputs fleet_qa_report.json and fleet_qa_report.md in the working dir.
"""
from __future__ import annotations

import json
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "fleet_qa_bundle.zip"
OUT_JSON = ROOT / "fleet_qa_report.json"
OUT_MD = ROOT / "fleet_qa_report.md"

METHODS = {
    "MD-VGGT-R": "ours_v7",
    "MD-VGGT Stage 6": "ours_v6",
    "DROID-SLAM": "droid",
    "pi3": "pi3",
    "Reloc3r-512": "reloc3r",
    "ORB-E": "orb_e",
}
COHORTS = {
    "core-complement": "confirmatory265",
    "development-extension": "confirmatory_extension93",
}
BOOT_DRAWS = 20000
SEED = 20260930


def load_records(bundle_dir: Path, method_tag: str, cohort_tag: str):
    path = bundle_dir / f"{method_tag}_{cohort_tag}" / "summary.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    for rec in data["records"]:
        records[rec["seq_id"]] = {
            "ate": rec["ate_sim3"]["rmse"],
            "ref": rec.get("reference_fraction"),
        }
    return records


def rank_biserial(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation from signed ranks."""
    diffs = diffs[np.abs(diffs) > 0]
    n = len(diffs)
    if n == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.abs(diffs))) + 1.0
    w_plus = float(ranks[diffs > 0].sum())
    w_minus = float(ranks[diffs < 0].sum())
    total = n * (n + 1) / 2.0
    return (w_plus - w_minus) / total


def wilcoxon_p_asymptotic(diffs: np.ndarray) -> float:
    diffs = diffs[np.abs(diffs) > 0]
    n = len(diffs)
    if n < 10:
        return float("nan")
    ranks = np.argsort(np.argsort(np.abs(diffs))) + 1.0
    w = float(min(ranks[diffs > 0].sum(), ranks[diffs < 0].sum()))
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w - mean) / math.sqrt(var)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def bootstrap_median_diff(diffs: np.ndarray, rng: np.random.Generator):
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    meds = np.empty(BOOT_DRAWS)
    for b in range(BOOT_DRAWS):
        sample = diffs[rng.integers(0, n, n)]
        meds[b] = np.median(sample)
    return (
        float(np.median(diffs)),
        float(np.percentile(meds, 2.5)),
        float(np.percentile(meds, 97.5)),
    )


def paired_stats(ours, other):
    common = sorted(set(ours) & set(other))
    d = np.array(
        [ours[k]["ate"] - other[k]["ate"] for k in common], dtype=float
    )
    wins = int((d < 0).sum())
    med, lo, hi = bootstrap_median_diff(d, RNG)
    return {
        "n": len(common),
        "win_rate": wins / len(common) if common else float("nan"),
        "mean_diff_mm": float(d.mean()),
        "median_diff_mm": med,
        "median_diff_95ci_mm": [lo, hi],
        "rank_biserial": rank_biserial(d),
        "wilcoxon_p_asymptotic": wilcoxon_p_asymptotic(d),
    }


def stratified_extension(ours, other):
    common = sorted(set(ours) & set(other))
    out = {}
    for name, lo, hi in (
        ("low", None, 0.3),
        ("mid", 0.3, 0.7),
        ("high", 0.7, None),
    ):
        keys = []
        for k in common:
            r = ours[k]["ref"]
            if r is None:
                continue
            if lo is not None and r < lo:
                continue
            if hi is not None and r >= hi:
                continue
            keys.append(k)
        d = np.array(
            [ours[k]["ate"] - other[k]["ate"] for k in keys], dtype=float
        )
        if len(d) >= 5:
            med, lo_ci, hi_ci = bootstrap_median_diff(d, RNG)
            out[name] = {
                "n": len(keys),
                "win_rate": float((d < 0).mean()),
                "median_diff_mm": med,
                "median_diff_95ci_mm": [lo_ci, hi_ci],
                "rank_biserial": rank_biserial(d),
            }
    return out


def number_consistency(bundle_dir: Path):
    analysis = json.loads(
        (bundle_dir / "confirmatory_analysis.json").read_text(encoding="utf-8")
    )
    tex = (bundle_dir / "mia_paper.tex").read_text(encoding="utf-8")
    checks = []
    for cohort in analysis["cohorts"]:
        for method in cohort["methods"]:
            mean = method["mean_ate_mm"]
            name = method["method"]
            rounded = f"{mean:.2f}"
            present = rounded in tex
            checks.append(
                {
                    "cohort": cohort["cohort"],
                    "method": name,
                    "json_mean": round(mean, 4),
                    "rounded": rounded,
                    "rounded_value_present_in_tex": present,
                }
            )
    return checks


def language_scan(bundle_dir: Path):
    tex = (bundle_dir / "mia_paper.tex").read_text(encoding="utf-8")
    body = re.sub(r"%.*", "", tex)
    body = re.sub(r"\\[A-Za-z@]+(?:\[[^]]*\])?(?:\{[^}]*\})?", " ", body)
    sentences = re.split(r"(?<=[.!?])\s+", body)
    lengths = [
        len(re.findall(r"\b[A-Za-z0-9'-]+\b", s)) for s in sentences
    ]
    lengths = [n for n in lengths if n >= 3]
    passive_hits = re.findall(
        r"\b(?:is|are|was|were|be|been|being)\s+[a-z]+(?:ed|en)\b",
        body,
        flags=re.I,
    )
    words = re.findall(r"\b[A-Za-z0-9'-]+\b", body)
    long_sentences = sorted(
        (n for n in lengths if n > 45), reverse=True
    )[:10]
    return {
        "n_sentences": len(lengths),
        "mean_sentence_words": float(np.mean(lengths)),
        "p95_sentence_words": float(np.percentile(lengths, 95)),
        "sentences_over_45_words": long_sentences,
        "passive_voice_hits": len(passive_hits),
        "passive_per_1000_words": 1000.0 * len(passive_hits) / max(len(words), 1),
        "total_words": len(words),
    }


RNG = np.random.default_rng(SEED)


def main() -> None:
    bundle_dir = ROOT / "fleet_qa_bundle"
    bundle_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(BUNDLE) as zf:
        zf.extractall(bundle_dir)

    report = {"paired": {}, "stratified_extension": {}, "checks": {}, "language": {}}
    for cohort_name, cohort_tag in COHORTS.items():
        ours = load_records(bundle_dir, METHODS["MD-VGGT-R"], cohort_tag)
        for method, tag in METHODS.items():
            if method == "MD-VGGT-R":
                continue
            other = load_records(bundle_dir, tag, cohort_tag)
            if ours and other:
                report["paired"].setdefault(cohort_name, {})[method] = paired_stats(
                    ours, other
                )
        if cohort_name == "development-extension" and ours:
            reloc = load_records(bundle_dir, METHODS["Reloc3r-512"], cohort_tag)
            droid = load_records(bundle_dir, METHODS["DROID-SLAM"], cohort_tag)
            if reloc:
                report["stratified_extension"]["vs Reloc3r-512"] = (
                    stratified_extension(ours, reloc)
                )
            if droid:
                report["stratified_extension"]["vs DROID-SLAM"] = (
                    stratified_extension(ours, droid)
                )

    report["checks"]["number_consistency"] = number_consistency(bundle_dir)
    report["language"] = language_scan(bundle_dir)

    OUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    lines = ["# Fleet QA report", ""]
    for cohort, rows in report["paired"].items():
        lines.append(f"## {cohort}")
        for method, s in rows.items():
            lines.append(
                f"- MD-VGGT-R vs {method}: n={s['n']}, win={s['win_rate']:.2f}, "
                f"median diff {s['median_diff_mm']:.3f} mm "
                f"[{s['median_diff_95ci_mm'][0]:.3f}, {s['median_diff_95ci_mm'][1]:.3f}], "
                f"rank-biserial {s['rank_biserial']:.3f}, "
                f"Wilcoxon p~{s['wilcoxon_p_asymptotic']:.4g}"
            )
        lines.append("")
    lines.append("## Stratified extension (MD-VGGT-R)")
    for comp, strata in report["stratified_extension"].items():
        lines.append(f"### {comp}")
        for name, s in strata.items():
            lines.append(
                f"- {name}: n={s['n']}, win={s['win_rate']:.2f}, "
                f"median diff {s['median_diff_mm']:.3f} mm "
                f"[{s['median_diff_95ci_mm'][0]:.3f}, {s['median_diff_95ci_mm'][1]:.3f}], "
                f"rank-biserial {s['rank_biserial']:.3f}"
            )
        lines.append("")
    missing = [
        c for c in report["checks"]["number_consistency"]
        if not c["rounded_value_present_in_tex"]
    ]
    lines.append("## Number consistency")
    lines.append(
        f"- {len(report['checks']['number_consistency']) - len(missing)}"
        f"/{len(report['checks']['number_consistency'])} rounded means present in tex"
    )
    for c in missing:
        lines.append(
            f"- MISSING: {c['cohort']} {c['method']} {c['rounded']}"
        )
    lang = report["language"]
    lines.append("## Language scan")
    lines.append(
        f"- sentences={lang['n_sentences']}, mean words/sentence="
        f"{lang['mean_sentence_words']:.1f}, p95={lang['p95_sentence_words']:.0f}"
    )
    lines.append(
        f"- passive hits per 1000 words: {lang['passive_per_1000_words']:.1f}"
    )
    lines.append(
        f"- sentences >45 words (top10): {lang['sentences_over_45_words']}"
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    sys.exit(main())
