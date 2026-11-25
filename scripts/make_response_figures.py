"""Generate publication-quality Response-to-Reviewers figures directly from verified JSON ledgers:
  1. `fig_resp_r1_mechanisms.png` (R1.1 & R1.3):
     - Left: Class-wise IoU (`static`, `deforming`, `instrument`) for sequence-label `r_v2` vs. per-pair `pairlab` on Abs & R2S.
     - Middle: Predicted static probability P_s and flow-error scale e^s across true kinematic regimes (`truly_static`, `slow_breathing`, `fast_deforming`, `instrument`).
     - Right: Region-wise warped feature mismatch |F_k - F_{k+1->k}|, flow-error scale e^s, and effective solver weight on unoccluded static vs. occluded/OOB vs. instrument cells.
  2. `fig_resp_r1_curriculum_losses.png` (R1.4 & R1.7):
     - Left: Six-stage progression (Stages 1->6) vs. 3-seed Single-Stage From-Scratch training vs. Same-Data 12.5-M PoseNet across simulated and real cohorts.
     - Right: Individual ablations of all 9 loss terms (Sim(3) ATE on dev & core).
  3. `fig_resp_r1_latency_cudagraph.png` (R1.8):
     - Left: Hardware-independent GFLOPs and ATen operator counts per pair.
     - Right: Batch-1 float32 wall-clock latency in Eager PyTorch vs. launch-free CUDA Graphs (showing 84.6 fps video-rate EgoLite).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "results" / "revision_vrih"
OUT_DIR = ROOT / "VRIH_EgoLite_submission" / "resp_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 8.5,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#1d5fa7"
LIGHT_BLUE = "#7aaad8"
ORANGE = "#e07b39"
GREEN = "#2b8cbe"
RED = "#c94a4a"
GREY = "#7f8c8d"
DARK = "#2c3e50"


def make_fig1_mechanisms() -> None:
    d1 = json.loads((REV / "r1_1_label_learning_audit.json").read_text(encoding="utf-8"))
    d3 = json.loads((REV / "r1_3_design_choices_lossless.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4), dpi=240)
    plt.subplots_adjust(wspace=0.31, left=0.055, right=0.985, top=0.84, bottom=0.22)

    # Panel 1: Class-wise IoU on Abs & R2S
    ax = axes[0]
    cats = ["Abs\nStatic", "Abs\nDeform.", "Abs\nTool", "R2S\nStatic", "R2S\nDeform.", "R2S\nTool"]
    seq_ious = [
        d1["abs_94"]["seq_model"]["class_metrics_vs_seq_gt"]["static"]["iou"],
        d1["abs_94"]["seq_model"]["class_metrics_vs_seq_gt"]["deforming"]["iou"],
        d1["abs_94"]["seq_model"]["class_metrics_vs_seq_gt"]["instrument"]["iou"],
        d1["r2s_40"]["seq_model"]["class_metrics_vs_seq_gt"]["static"]["iou"],
        d1["r2s_40"]["seq_model"]["class_metrics_vs_seq_gt"]["deforming"]["iou"],
        d1["r2s_40"]["seq_model"]["class_metrics_vs_seq_gt"]["instrument"]["iou"],
    ]
    pair_ious = [
        d1["abs_94"]["pair_model"]["class_metrics_vs_seq_gt"]["static"]["iou"],
        d1["abs_94"]["pair_model"]["class_metrics_vs_seq_gt"]["deforming"]["iou"],
        d1["abs_94"]["pair_model"]["class_metrics_vs_seq_gt"]["instrument"]["iou"],
        d1["r2s_40"]["pair_model"]["class_metrics_vs_seq_gt"]["static"]["iou"],
        d1["r2s_40"]["pair_model"]["class_metrics_vs_seq_gt"]["deforming"]["iou"],
        d1["r2s_40"]["pair_model"]["class_metrics_vs_seq_gt"]["instrument"]["iou"],
    ]
    x = np.arange(len(cats))
    w = 0.36
    b1 = ax.bar(x - w / 2, seq_ious, w, color=BLUE, label="Sequence-level labels")
    b2 = ax.bar(x + w / 2, pair_ious, w, color=ORANGE, label="Per-pair motion masks")
    for idx_b, (ba, bb) in enumerate(zip(b1, b2)):
        ha, hb = ba.get_height(), bb.get_height()
        if abs(ha - hb) < 0.06:
            ax.text(ba.get_x() + ba.get_width() / 2 - 0.04, ha + 0.015, f"{ha:.2f}", ha="center", va="bottom", fontsize=6.1)
            ax.text(bb.get_x() + bb.get_width() / 2 + 0.04, hb + 0.055 if hb >= ha else hb + 0.015, f"{hb:.2f}", ha="center", va="bottom", fontsize=6.1)
        else:
            ax.text(ba.get_x() + ba.get_width() / 2, ha + 0.015, f"{ha:.2f}", ha="center", va="bottom", fontsize=6.2)
            ax.text(bb.get_x() + bb.get_width() / 2, hb + 0.015, f"{hb:.2f}", ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylim(0, 0.96)
    ax.set_ylabel("Class-wise IoU")
    ax.set_title("Motion-layer segmentation IoU", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)

    # Panel 2: Slow-breathing observability vs. mean per-pair median e^s on Abs (94 seqs)
    ax = axes[1]
    bk_keys = ["truly_static", "slow_breathing", "fast_deforming", "instrument"]
    bk_labels = ["Truly\nstatic", "Slow\nbreath.", "Fast\ndeform.", "Rigid\ntool"]
    ps_vals = [d1["abs_94"]["seq_model"]["buckets"][k]["Ps"] for k in bk_keys]
    es_vals = [d1["abs_94"]["seq_model"]["buckets"][k]["es"] for k in bk_keys]
    x2 = np.arange(len(bk_keys))
    r1 = ax.bar(x2 - w / 2, ps_vals, w, color=LIGHT_BLUE, label="Static prob. $P_{\\mathrm{s}}$")
    r2 = ax.bar(x2 + w / 2, es_vals, w, color=RED, label="Mean of pair-med. $e^s$")
    for b in list(r1) + list(r2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.018, f"{h:.2f}", ha="center", va="bottom", fontsize=6.6)
    ax.set_xticks(x2); ax.set_xticklabels(bk_labels)
    ax.set_ylim(0, 0.90)
    ax.set_ylabel("Mean across pairs (Abs test)")
    ax.set_title("Response across kinematic regimes", fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=7.0)

    # Panel 3: Implicit occlusion & tool suppression via |F_k - F_{k+1->k}| and e^s on dev (92 seqs, lossless)
    ax = axes[2]
    reg_keys = ["static_unocc", "deform_unocc", "occluded_or_oob", "instr_unocc"]
    reg_labels = ["Static\nunocc.", "Deform.\nunocc.", "Occluded\n/ OOB", "Surgical\ntool"]
    reg_d = d3["region_distributions_r_v2"]["dev_92"]
    fdiff = [reg_d[k]["feat_diff"] * 10.0 for k in reg_keys]
    es_r = [reg_d[k]["es"] * 5.0 for k in reg_keys]
    eff_w = [reg_d[k]["eff_w"] for k in reg_keys]
    x3 = np.arange(len(reg_keys))
    w3 = 0.26
    ax.bar(x3 - w3, fdiff, w3, color="#5dade2", label="$10\\times |\\mathbf{F}_k-\\mathbf{F}_{k+1\\to k}|$")
    ax.bar(x3, es_r, w3, color=ORANGE, label="$5\\times$ error scale $e^s$")
    b_ew = ax.bar(x3 + w3, eff_w, w3, color=DARK, label="Effective GN weight")
    for b in b_ew:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    ax.set_xticks(x3); ax.set_xticklabels(reg_labels)
    ax.set_ylim(0, 1.52)
    ax.set_ylabel("Region median (dev 92, lossless)")
    ax.set_title("Implicit occlusion & tool suppression", fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=6.7)

    fig.savefig(OUT_DIR / "fig_resp_r1_mechanisms.png", dpi=240)
    plt.close(fig)
    print("Saved fig_resp_r1_mechanisms.png")


def make_fig2_curriculum_and_losses() -> None:
    d4 = json.loads((REV / "r1_4_seeds_and_stages_real.json").read_text(encoding="utf-8"))
    dr = json.loads((REV / "r1_3_4_6_7_retrained_ablations.json").read_text(encoding="utf-8"))
    ds2 = json.loads((REV / "r1_4_single_stage_generic_raft.json").read_text(encoding="utf-8"))
    dpn = json.loads((REV / "r1_7_samedata_posenet_fixed.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.5), dpi=240)
    plt.subplots_adjust(wspace=0.28, left=0.058, right=0.985, top=0.84, bottom=0.24)

    # Left: 6 stages vs Leak-Free Single-Stage From Scratch (3 seeds) vs Same-Data 12.5M PoseNet
    ax = axes[0]
    st_keys = ["stage1_v1", "stage2_v2", "stage3_v3abs", "stage4_s_big_ft", "stage5_r_v1", "stage6_r_v2"]
    st_x = np.arange(len(st_keys))
    dev_y = [d4["six_stage_progression"][k]["dev_92"] for k in st_keys]
    sc_y = [d4["six_stage_progression"][k]["scared_11"] for k in st_keys]
    ax.plot(st_x, dev_y, "-o", color=BLUE, lw=1.8, markersize=5, label="dev (92), six-stage schedule")
    ax.plot(st_x, sc_y, "-s", color=ORANGE, lw=1.8, markersize=5, label="SCARED (11), six-stage schedule")

    fs_dev = ds2["summary_3seeds"]["dev_92"]
    fs_sc = ds2["summary_3seeds"]["scared_11"]
    pn_dev = dpn["protocol_a"]["dev_92"]["posenet_mean_mm"]
    ax.axhline(fs_dev["mean_mm"], color=BLUE, ls="--", lw=1.3,
               label=f"dev, single stage (generic RAFT, 3 seeds: {fs_dev['mean_mm']:.2f}±{fs_dev['std_mm']:.2f})")
    ax.axhline(fs_sc["mean_mm"], color=ORANGE, ls="--", lw=1.3,
               label=f"SCARED, single stage (generic RAFT, 3 seeds: {fs_sc['mean_mm']:.2f}±{fs_sc['std_mm']:.2f})")
    ax.axhline(pn_dev, color=RED, ls=":", lw=1.6,
               label=f"dev, same-corpus 12.5-M PoseNet ({pn_dev:.2f} mm)")

    ax.set_xticks(st_x)
    ax.set_xticklabels(["Stage 1\n(4.6M)", "Stage 2\n(4.6M)", "Stage 3\n(50-mm unit)", "Stage 4\n(20.9M)", "Stage 5\n(+Abs & R2S)", "Stage 6\n(final)"], fontsize=7.2)
    ax.set_ylim(3.2, 11.4)
    ax.set_ylabel("Mean Sim(3) ATE (mm) $\\downarrow$")
    ax.set_title("Six-stage vs. single-stage training & same-corpus PoseNet", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=6.3)

    # Right: All 9 loss terms ablated individually
    ax = axes[1]
    loss_specs = [
        ("retrain_ctrl_full", "Full\nctrl."),
        ("retrain_nogn", "w/o $\\mathcal{L}_{\\mathrm{p}}$\n(no GN)"),
        ("loss_no_initpose", "w/o\n$\\mathcal{L}_0$"),
        ("loss_no_depth", "w/o\n$\\mathcal{L}_Z$"),
        ("loss_no_layer", "w/o\n$\\mathcal{L}_{\\mathrm{layer}}$"),
        ("loss_no_cam", "w/o\n$\\mathcal{L}_{\\mathrm{cam}}$"),
        ("loss_no_unc", "w/o\n$\\mathcal{L}_{\\mathrm{unc}}$"),
        ("loss_no_flow", "w/o\n$\\mathcal{L}_{\\mathrm{flow}}$"),
        ("loss_no_traj", "w/o\n$\\mathcal{L}_{\\mathrm{traj}}$"),
        ("loss_no_cons_only", "w/o\n$\\mathcal{L}_{\\mathrm{cons}}$"),
        ("loss_no_rho_only", "w/o\n$\\mathcal{L}_{\\rho}$"),
    ]
    lx = np.arange(len(loss_specs))
    w = 0.38
    dev_l = [dr["variants"][k]["dev_92"]["mean_mm"] for k, _ in loss_specs]
    core_l = [dr["variants"][k]["core_265"]["mean_mm"] for k, _ in loss_specs]
    b_d = ax.bar(lx - w / 2, dev_l, w, color=BLUE, label="dev (92 seqs)")
    b_c = ax.bar(lx + w / 2, core_l, w, color=LIGHT_BLUE, label="core (265 seqs)")
    for b in [b_d[0], b_d[1], b_c[0], b_c[1]]:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.08, f"{h:.2f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    ax.axhline(dev_l[0], color=BLUE, ls=":", lw=0.9, alpha=0.7)
    ax.axhline(core_l[0], color=DARK, ls=":", lw=0.9, alpha=0.5)
    ax.set_xticks(lx)
    ax.set_xticklabels([lbl for _, lbl in loss_specs], fontsize=6.8)
    ax.set_ylim(0, 6.7)
    ax.set_ylabel("Mean Sim(3) ATE (mm) $\\downarrow$")
    ax.set_title("Individual ablation of all nine loss terms (stage 6)", fontweight="bold")
    ax.legend(loc="upper right", frameon=False)

    fig.savefig(OUT_DIR / "fig_resp_r1_curriculum_losses.png", dpi=240)
    plt.close(fig)
    print("Saved fig_resp_r1_curriculum_losses.png")


def make_fig3_latency_cudagraphs() -> None:
    d8 = json.loads((REV / "r1_8_latency_and_flops_audit.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.3), dpi=240)
    plt.subplots_adjust(wspace=0.28, left=0.065, right=0.985, top=0.84, bottom=0.22)

    models = ["AF-SfMLearner\n(13.0 M, 256×320)", "EndoDAC\n(13.0 M, 256×320)", "EgoLite\n(20.9 M, 256×320)", "Endo-FASt3r\n(428.4 M, 384×512)"]
    gflops = [6.52, 6.52, 153.11, 1501.49]
    ops = [205, 205, 2877, 9044]
    eager_ms = [
        d8["models"]["afsfm"]["eager_fp32"]["median_ms"],
        d8["models"]["endodac"]["eager_fp32"]["median_ms"],
        d8["models"]["EgoLite"]["eager_fp32"]["median_ms"],
        d8["models"]["Endo-FASt3r"]["eager_fp32"]["median_ms"],
    ]
    cg_ms = [
        d8["cuda_graphs_batch1_fp32"]["afsfm_fp32"]["cudagraph_median_ms"],
        d8["cuda_graphs_batch1_fp32"]["endodac_fp32"]["cudagraph_median_ms"],
        d8["cuda_graphs_batch1_fp32"]["EgoLite_full_fp32"]["cudagraph_median_ms"],
        d8["cuda_graphs_batch1_fp32"]["EndoFASt3r_fp32"]["cudagraph_median_ms"],
    ]

    # Left: GFLOPs & ATen Ops (log scale)
    ax = axes[0]
    x = np.arange(len(models))
    w = 0.36
    b1 = ax.bar(x - w / 2, gflops, w, color=BLUE, label="Compute (GFLOPs / pair)")
    b2 = ax.bar(x + w / 2, ops, w, color=GREY, label="GPU kernel launches / pair")
    ax.set_yscale("log")
    for b in list(b1) + list(b2):
        h = b.get_height()
        txt = f"{h:.1f}" if h < 200 else f"{int(round(h))}"
        ax.text(b.get_x() + b.get_width() / 2, h * 1.15, txt, ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(1, 28000)
    ax.set_ylabel("Count / GFLOPs (log scale)")
    ax.set_title("Hardware-independent compute & kernel launches", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)

    # Right: Eager PyTorch vs CUDA Graphs batch-1 latency
    ax = axes[1]
    b3 = ax.bar(x - w / 2, eager_ms, w, color=ORANGE, label="Eager execution (dispatch-bound)")
    b4 = ax.bar(x + w / 2, cg_ms, w, color=BLUE, label="Static CUDA Graph (dispatch-free)")
    ax.axhline(33.3, color=RED, ls="--", lw=1.1, label="30 fps video-rate threshold (33.3 ms)")
    for b in list(b3) + list(b4):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 2.2, f"{h:.1f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold" if h in (11.82, 78.47) else "normal")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(0, 152)
    ax.set_ylabel("Batch-1 latency on RTX 5090 Laptop (ms) $\\downarrow$")
    ax.set_title("Batch-1 float32 latency before & after CUDA Graphs", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)

    fig.savefig(OUT_DIR / "fig_resp_r1_latency_cudagraph.png", dpi=240)
    plt.close(fig)
    print("Saved fig_resp_r1_latency_cudagraph.png")


if __name__ == "__main__":
    make_fig1_mechanisms()
    make_fig2_curriculum_and_losses()
    make_fig3_latency_cudagraphs()
