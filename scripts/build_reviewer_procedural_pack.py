"""Assemble a reviewer-facing procedural-only EndoEgoSim pack (no private textures)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "submission_archive" / "EndoEgoSim_reviewer_procedural_pack"

LISTS = [
    "simtest92.txt",
    "simtest92_low.txt",
    "simtest92_mid.txt",
    "simtest92_high.txt",
    "sim_confirmatory265.txt",
    "sim_confirmatory_extension93.txt",
]

LEDGERS = [
    "confirmatory_analysis.json",
    "confirmatory_manifest.json",
    "distortion_sensitivity.json",
    "FINDINGS.md",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--generate-demo", type=int, default=0, help="N procedural sequences")
    parser.add_argument("--seed-start", type=int, default=900001)
    args = parser.parse_args()

    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    copy_file(ROOT / "configs" / "procedural_review.json", out / "configs" / "procedural_review.json")
    copy_file(ROOT / "configs" / "default.json", out / "configs" / "default.json")

    for name in LISTS:
        src = ROOT / "lists" / name
        if src.is_file():
            copy_file(src, out / "lists" / name)

    for name in LEDGERS:
        src = ROOT / "results" / "sota" / name
        if src.is_file():
            copy_file(src, out / "result_ledgers" / name)

    for rel in (
        "ours_v7_final/summary.json",
        "ours_v7_confirmatory265/summary.json",
        "ours_v7_confirmatory_extension93/summary.json",
    ):
        src = ROOT / "results" / "sota" / rel
        if src.is_file():
            copy_file(src, out / "result_ledgers" / rel)

    for p in sorted((ROOT / "endosim").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        copy_file(p, out / "code" / p.relative_to(ROOT))
    for name in ("generate_dataset.py", "verify_gt.py", "build_reviewer_procedural_pack.py"):
        p = ROOT / "scripts" / name
        if p.is_file():
            copy_file(p, out / "code" / "scripts" / name)

    readme = out / "README_REVIEWERS.md"
    readme.write_text(
        "\n".join(
            [
                "# EndoEgoSim reviewer pack (procedural-only)",
                "",
                "This pack excludes private hospital textures and third-party appearance banks.",
                "It is intended for peer review of labels, generator behaviour, and frozen metrics.",
                "",
                "## Contents",
                "- `configs/procedural_review.json` — `texture_source=procedural`, `barrel_prob=0`",
                "- `lists/` — frozen evaluation identities used in the manuscript",
                "- `result_ledgers/` — sequence-macro summaries matching paper tables",
                "- `code/` — generator sources needed to rebuild procedural RGB",
                "- `demo_data/` — optional short procedural sequences (if generated)",
                "",
                "## Rebuild a small procedural demo",
                "```bash",
                "python code/scripts/generate_dataset.py \\",
                "  --config configs/procedural_review.json \\",
                "  --n-seq 8 --seed-start 900001 --workers 4 --no-bank \\",
                "  --out demo_data",
                "```",
                "",
                "No real texture bank path is required for this config.",
                "",
                f"Built UTC: {datetime.now(timezone.utc).isoformat()}",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    demo_note = ""
    if args.generate_demo > 0:
        demo_dir = out / "demo_data"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "generate_dataset.py"),
            "--config",
            str(ROOT / "configs" / "procedural_review.json"),
            "--n-seq",
            str(args.generate_demo),
            "--seed-start",
            str(args.seed_start),
            "--workers",
            "2",
            "--no-bank",
            "--out",
            str(demo_dir),
        ]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(ROOT))
        demo_note = f"demo_sequences={args.generate_demo}"

    files = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            files.append(
                {
                    "path": p.relative_to(out).as_posix(),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
            )
    (out / "MANIFEST.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "demo_note": demo_note,
                "files": files,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    zip_path = out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print(f"Pack: {out}")
    print(f"Zip:  {zip_path}")
    print(f"SHA-256: {sha256_file(zip_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
