"""Publication figure: fair EndoEgoSim / real-domain comparison.

Ranking pool = single estimator, 92/92 poses. ORB-SLAM3, depth-GT controls, and
multi-estimator systems are shown only in the completeness / footnote panels.
Numbers match mia_paper.tex and results/sota/*/summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sota"
OUT_DIR = Path(r"D:\MIS_Pose_Track_Re3D")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(tag):
    with (RESULTS / tag / "summary.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def summary(tag):
    return load(tag)["summary"]


def sim3(tag, scale=1.0):
    return summary(tag)["ate_sim3_rmse_mean"] / scale


def bucket(tag, name):
    rows = summary(tag).get("stratified", [])
    row = next(item for item in rows if name in item["bucket"])
    return row["ate_sim3_mean"]

# Colour-blind Okabe–Ito (no rainbow)
C = {
    "ours": "#0072B2",
    "slam": "#E69F00",
    "recon": "#999999",
    "ft": "#56B4E9",
    "geo": "#D55E00",
    "ctrl": "#BBBBBB",
    "system": "#CC79A7",
    "ink": "#222222",
    "muted": "#666666",
}

# Full-list single estimators only (n=92); values are read from run summaries.
RANK_SOURCES = [
    ("MD-VGGT-R*", "ours_v7_final", "ours"),
    ("Stage 6 route + instrument", "ours_v6_decouple_v3", "ours"),
    ("MD-VGGT-R τ=.30", "ours_v7_infer_simtest", "ours"),
    ("DROID-SLAM", "droid_simtest", "slam"),
    ("π³ zs", "pi3_simtest", "slam"),
    ("Reloc3r-512 zs", "reloc3r_simtest", "recon"),
    ("VGGT global FT", "vggt_ft2_simtest", "ft"),
    ("CUT3R zs", "cut3r_simtest", "recon"),
    ("Mesh-RTS", "meshrts_v3_c2w", "geo"),
    ("ORB-E", "orb_e_c2w_simtest", "geo"),
    ("VGGT heads FT", "vggt_ft_simtest", "ft"),
    ("VGGT zs", "vggt_simtest", "recon"),
    ("Identity", "identity_simtest", "ctrl"),
]
RANK = sorted(
    [(name, sim3(tag), kind) for name, tag, kind in RANK_SOURCES],
    key=lambda item: item[1],
)

STRAT_SOURCES = [
    ("Identity", "identity_simtest"),
    ("ORB-E", "orb_e_c2w_simtest"),
    ("Mesh-RTS", "meshrts_v3_c2w"),
    ("VGGT zs", "vggt_simtest"),
    ("Reloc3r", "reloc3r_simtest"),
    ("π³", "pi3_simtest"),
    ("DROID", "droid_simtest"),
    ("Stage 6 + route", "ours_v6_decouple_v3"),
    ("MD-VGGT-R*", "ours_v7_final"),
]
STRAT_NAMES = [name for name, _ in STRAT_SOURCES]
STRAT = {
    "low": [bucket(tag, "低") for _, tag in STRAT_SOURCES],
    "mid": [bucket(tag, "中") for _, tag in STRAT_SOURCES],
    "high": [bucket(tag, "高") for _, tag in STRAT_SOURCES],
}

TRACK_SOURCES = [
    ("MD-VGGT-R*", "ours_v7_final", "ours"),
    ("DROID-SLAM", "droid_simtest", "slam"),
    ("π³", "pi3_simtest", "slam"),
    ("CUT3R", "cut3r_simtest", "recon"),
    ("VGGT zs", "vggt_simtest", "recon"),
    ("ORB-E", "orb_e_c2w_simtest", "geo"),
    ("Mesh-RTS", "meshrts_v3_c2w", "geo"),
    ("ORB-SLAM3", "orbslam3_simtest", "system"),
]
TRACK = [
    (
        name,
        int(summary(tag)["n_seq"]),
        92 - int(summary(tag)["n_seq"]),
        kind,
    )
    for name, tag, kind in TRACK_SOURCES
]


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(ax, letter):
    ax.text(
        -0.08,
        1.08,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=C["ink"],
    )


def panel_rank(ax):
    names = [r[0] for r in RANK][::-1]
    vals = [r[1] for r in RANK][::-1]
    cols = [C[r[2]] for r in RANK][::-1]
    y = np.arange(len(names))
    bars = ax.barh(y, vals, color=cols, height=0.72, edgecolor="none", zorder=2)
    for i, (name, v, kind) in enumerate(RANK[::-1]):
        if name == "MD-VGGT-R*":
            bars[i].set_edgecolor(C["ink"])
            bars[i].set_linewidth(1.1)
            ax.text(v + 0.28, y[i], f"{v:.2f}  lowest mean/median", va="center", fontsize=7, color=C["ours"], fontweight="bold")
        else:
            ax.text(v + 0.28, y[i], f"{v:.2f}", va="center", fontsize=7, color=C["muted"])
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Sim(3) ATE mean (mm), lower is better")
    ax.set_xlim(0, max(vals) * 1.2)
    primary_mean = sim3("ours_v7_final")
    droid_mean = sim3("droid_simtest")
    ax.axvline(primary_mean, color=C["ours"], ls="--", lw=0.8, zorder=1, alpha=0.85)
    ax.axvline(droid_mean, color=C["slam"], ls=":", lw=0.8, zorder=1, alpha=0.85)
    ax.set_title("Ranking pool only: single estimator, 92/92 poses", loc="left", pad=4)
    ax.grid(axis="x", color="#eeeeee", lw=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [
        mpatches.Patch(color=C["ours"], label="Proposed estimator"),
        mpatches.Patch(color=C["slam"], label="Zero-shot SLAM / π³"),
        mpatches.Patch(color=C["ft"], label="Fine-tuned baselines"),
        mpatches.Patch(color=C["recon"], label="Zero-shot reconstruction"),
        mpatches.Patch(color=C["geo"], label="Geometric controls"),
        mpatches.Patch(color=C["ctrl"], label="Identity control"),
    ]
    ax._rank_handles = handles
    _panel_label(ax, "a")


def panel_strat(ax):
    x = np.arange(len(STRAT_NAMES))
    w = 0.26
    ax.bar(x - w, STRAT["low"], w, color="#C44E52", label="Low ref. <0.3 (n=16)", zorder=2)
    ax.bar(x, STRAT["mid"], w, color="#E69F00", label="Mid ref. 0.3–0.7 (n=44)", zorder=2)
    ax.bar(x + w, STRAT["high"], w, color="#009E73", label="High ref. >0.7 (n=32)", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(STRAT_NAMES, rotation=25, ha="right")
    ax.set_ylabel("Sim(3) ATE mean (mm)")
    ax.set_ylim(0, max(max(values) for values in STRAT.values()) * 1.12)
    ax.set_title("ATE by static-reference fraction", loc="left", pad=4)
    ax.legend(frameon=False, loc="upper right")
    ax.axhline(bucket("ours_v7_final", "低"), color=C["ink"], ls="--", lw=0.7, alpha=0.8)
    ax.grid(axis="y", color="#eeeeee", lw=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _panel_label(ax, "b")


def panel_track(ax):
    names = [t[0] for t in TRACK]
    ok = np.array([t[1] for t in TRACK], dtype=float)
    lost = np.array([t[2] for t in TRACK], dtype=float)
    cols = [C[t[3]] for t in TRACK]
    x = np.arange(len(names))
    ax.bar(x, ok, color=cols, zorder=2, label="Tracked (≥5 poses)")
    ax.bar(x, lost, bottom=ok, color="#F0E6E6", edgecolor="#AA5555", linewidth=0.6, zorder=2, label="Lost (not imputed)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Sequences (of 92)")
    ax.set_ylim(0, 108)
    ax.set_title("Tracking completeness", loc="left", pad=4)
    for i, (name, n_ok, n_lost, _) in enumerate(TRACK):
        if n_lost:
            ax.text(i, n_ok + n_lost + 3, f"{n_ok}/92\nATE on successes\nonly: 1.40 mm", ha="center", va="bottom", fontsize=6.2, color="#AA3333")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.5, zorder=0)
    _panel_label(ax, "c")


def panel_real(ax):
    sources = [
        ("SCARED full\nStage 3", "ours_v3_scared_official", 1000.0, "ours"),
        ("SCARED full\nMD-VGGT-G", "ours_v7_scared_official", 1.0, "ours"),
        ("SCARED full\nORB-E", "orb_e_c2w_scared_full", 1.0, "geo"),
        ("C3VD† full\nMD-VGGT-G", "ours_v7_c3vd_full", 1.0, "ours"),
        ("C3VD† full\nORB-E", "orb_e_c2w_c3vd_full", 1.0, "geo"),
        ("StereoMIS 64\nMD-VGGT-G", "ours_v7_plain_u64", 1.0, "ours"),
        ("StereoMIS 64\nDROID", "droid_stereomis", 1.0, "slam"),
        ("StereoMIS 64\nORB-E", "orb_e_c2w_stereomis_u64", 1.0, "geo"),
    ]
    labels = [item[0] for item in sources]
    vals = [sim3(tag, scale) for _, tag, scale, _ in sources]
    cols = [C[kind] for *_, kind in sources]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=cols, zorder=2)
    for i in (0, 3, 5):
        bars[i].set_edgecolor(C["ink"])
        bars[i].set_linewidth(1.0)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.45, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=C["ink"])
    ax.axvline(2.5, color="#dddddd", lw=0.8)
    ax.axvline(4.5, color="#dddddd", lw=0.8)
    y_top = max(vals) * 1.10
    ax.text(1.0, y_top, "hop 0.239 mm", ha="center", fontsize=6.5, color=C["muted"])
    ax.text(3.5, y_top, "hop 0.23 mm", ha="center", fontsize=6.5, color=C["muted"])
    ax.text(6.0, y_top, "hop 8.50 mm", ha="center", fontsize=6.5, color=C["muted"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.6, rotation=38, ha="right")
    ax.set_ylabel("Sim(3) ATE mean (mm)")
    ax.set_ylim(0, max(vals) * 1.27)
    ax.set_title("Real-video results (within protocol)", loc="left", pad=4)

    paired_differences = []
    paired_relative = []
    pair_tags = [
        ("ours_v7_plain_u64", "orb_e_c2w_stereomis_u64"),
        ("ours_v7_c3vd_full", "orb_e_c2w_c3vd_full"),
        ("ours_v7_scared_official", "orb_e_c2w_scared_full"),
    ]
    for ours_tag, orb_tag in pair_tags:
        ours = {
            row["seq_id"]: row["ate_sim3"]["rmse"]
            for row in load(ours_tag)["records"] if "error" not in row
        }
        orb = {
            row["seq_id"]: row["ate_sim3"]["rmse"]
            for row in load(orb_tag)["records"] if "error" not in row
        }
        common = sorted(set(ours) & set(orb))
        x_ours = np.array([ours[key] for key in common])
        x_orb = np.array([orb[key] for key in common])
        paired_differences.append(x_ours - x_orb)
        paired_relative.append((x_ours - x_orb) / x_orb)
    differences = np.concatenate(paired_differences)
    relative = np.concatenate(paired_relative)
    wins = int(np.sum(differences < 0))
    pooled_p = stats.wilcoxon(differences, alternative="two-sided").pvalue

    frame_values = [
        sim3("ours_v7_plain_u64"),
        sim3("ours_v7_plain_u128"),
        sim3("ours_v7_plain_u256"),
        sim3("ours_v7_noroute_stereomis_full"),
    ]
    ax.text(
        0.0,
        -0.34,
        "Exploratory pooled MD-VGGT-G vs ORB-E:\n"
        f"lower ATE on {wins}/{len(differences)};\n"
        f"median relative reduction {-np.median(relative) * 100:.1f}%;\n"
        f"two-sided Wilcoxon p={pooled_p:.4f}.\n"
        "†C3VD is training-exposed;\n"
        "clips are not patient-independent.\n"
        "StereoMIS MD-VGGT-G (64/128/256/full):\n"
        + "/".join(f"{value:.2f}" for value in frame_values)
        + " mm.",
        transform=ax.transAxes,
        fontsize=5.8,
        color=C["muted"],
        va="top",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.5, zorder=0)
    _panel_label(ax, "d")


def main():
    _style()
    fig = plt.figure(figsize=(7.16, 7.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], wspace=0.32, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    panel_rank(ax_a)
    panel_strat(ax_b)
    panel_track(ax_c)
    panel_real(ax_d)
    fig.legend(
        handles=ax_a._rank_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.55, 0.945),
    )
    fig.suptitle(
        "Simulation: uniform ≤64 frames (actual 40–64), 92/92 single-estimator methods; *MD-VGGT-R is benchmark-tuned.\n"
        "Real videos: compare only within dataset and frame budget.",
        fontsize=7.0,
        color=C["muted"],
        y=0.992,
    )
    fig.subplots_adjust(left=0.225, right=0.985, top=0.855, bottom=0.20,
                        wspace=0.42, hspace=0.48)
    paper = ROOT / "mia_paper"
    for dest in (OUT_DIR, paper, OUT_DIR / "mia_paper"):
        dest.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest / "fig_sota_leaderboard.pdf", dpi=300)
        fig.savefig(dest / "fig_sota_leaderboard.jpg", dpi=300)
        print(f"wrote {dest / 'fig_sota_leaderboard.jpg'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
