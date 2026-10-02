"""Build extra qualitative image panels for the MIA manuscript."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from endosim.eval.align import align_trajectories
from endosim.eval.metrics import load_pose_txt
from endosim.eval.protocol import list_color_frames, select_frame_indices

SOTA = ROOT / "results" / "sota"
PAPER = ROOT / "mia_paper"
LISTS = ROOT / "lists"

LAYER_COLORS = {
    1: (34, 139, 230),
    2: (242, 166, 59),
    3: (210, 76, 125),
}

SIM_ROWS = [
    ("seq_00000150", "High-reference"),
    ("seq_00002344", "Mid-reference"),
    ("seq_00002069", "Low-reference"),
    ("seq_00001305", "High-error case"),
]


def _seq_map() -> dict[str, Path]:
    out = {}
    lst = LISTS / "simtest92.txt"
    if lst.is_file():
        for ln in lst.read_text(encoding="utf-8").splitlines():
            p = Path(ln.strip())
            if p.is_dir():
                out[p.name] = p
    for split in ("test", "val", "train"):
        for p in (ROOT / "sim_data" / split).glob("seq_*"):
            out.setdefault(p.name, p)
    return out


def _colorize_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    palette = np.zeros_like(rgb)
    for lab, color in LAYER_COLORS.items():
        palette[mask == lab] = color
    overlay = np.round(0.40 * rgb + 0.60 * palette).astype(np.uint8)
    boundary = np.zeros(mask.shape, np.uint8)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    overlay[boundary > 0] = 255
    return overlay


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    vis = np.zeros((*depth.shape, 3), np.uint8)
    valid = depth > 0
    if not valid.any():
        return vis
    lo, hi = np.percentile(depth[valid], [2, 98])
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    turbo = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[valid] = cv2.cvtColor(turbo, cv2.COLOR_BGR2RGB)[valid]
    return vis


def _resize(im: np.ndarray, w: int = 220) -> np.ndarray:
    h = int(round(im.shape[0] * w / im.shape[1]))
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)


def _mid_frame(seq: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    colors = sorted((seq / "color").glob("*.png"))
    if not colors:
        raise FileNotFoundError(seq / "color")
    fp = colors[len(colors) // 2]
    rgb = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
    mask_p = seq / "motion_mask" / fp.name
    depth_p = seq / "depth" / fp.name
    mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED) if mask_p.is_file() else None
    depth = None
    if depth_p.is_file():
        raw = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            depth = raw.astype(np.float32)
            if raw.dtype == np.uint16:
                depth = depth / 1000.0
    return rgb, mask, depth, fp.stem


def plot_motion_gallery(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seqs = _seq_map()
    fig, axes = plt.subplots(4, 4, figsize=(11.4, 8.6))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.04, wspace=0.03, hspace=0.18)
    titles = ["RGB", "Motion layers", "Layer overlay", "Metric depth"]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=9)
    for i, (sid, bucket) in enumerate(SIM_ROWS):
        seq = seqs[sid]
        rgb, mask, depth, frame_id = _mid_frame(seq)
        if mask is None:
            raise FileNotFoundError(seq / "motion_mask")
        layers = np.zeros_like(rgb)
        for lab, color in LAYER_COLORS.items():
            layers[mask == lab] = color
        overlay = _colorize_mask(rgb, mask)
        depth_vis = _colorize_depth(depth) if depth is not None else np.full_like(rgb, 230)
        images = [rgb, layers, overlay, depth_vis]
        for j, im in enumerate(images):
            axes[i, j].imshow(_resize(im))
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for sp in axes[i, j].spines.values():
                sp.set_linewidth(0.4)
                sp.set_color("#888")
        axes[i, 0].set_ylabel(
            f"{bucket}\n{sid}\nframe {frame_id}",
            fontsize=7.5,
            rotation=0,
            ha="right",
            va="center",
            labelpad=28,
        )
    fig.suptitle(
        "EndoEgoSim motion-layer supervision used by MD-VGGT. "
        "Blue: stationary support ($m{=}1$); amber: deforming tissue ($m{=}2$); "
        "magenta: independent object ($m{=}3$). Depth remains metric and unwarped.",
        fontsize=9,
        y=0.98,
    )
    fig.text(
        0.53,
        0.012,
        "Independent objects are excluded from depth loss; deforming tissue remains in the depth mask.",
        ha="center",
        fontsize=8,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


def _aligned_xy(est: np.ndarray, gt: np.ndarray):
    n = min(len(est), len(gt))
    aligned, _ = align_trajectories(est[:n], gt[:n], with_scale=True)
    return gt[:n, :3, 3], aligned[:, :3, 3]


def plot_real_gallery(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        (
            "C3VD",
            ROOT / "sim_data" / "real_refs" / "c1_transverse1_t1_v4_c1_transverse1_t1_v4",
            SOTA / "ours_v7_c3vd_full",
            SOTA / "orb_e_c2w_c3vd_full",
            "MD-VGGT-G",
            "ORB-E",
        ),
        (
            "StereoMIS 64-frm",
            ROOT / "sim_data" / "real_refs" / "stereomis_extracted_StereoMIS_0_0_1_P1",
            SOTA / "ours_v7_plain_u64",
            SOTA / "droid_stereomis",
            "MD-VGGT-G",
            "DROID-SLAM",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.4))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.88, bottom=0.08, wspace=0.18, hspace=0.32)
    for row, (name, seq, ours_dir, base_dir, ours_lab, base_lab) in enumerate(panels):
        if not seq.is_dir():
            for ax in axes[row]:
                ax.set_axis_off()
            continue
        gt_all = load_pose_txt(str(seq / "pose_c2w.txt"))
        idx = select_frame_indices(gt_all, "uniform", max_frames=64)
        frames = list_color_frames(str(seq), indices=idx)
        show = [frames[k] for k in np.linspace(0, len(frames) - 1, 4).astype(int) if frames]
        tiles = []
        for fp in show:
            im = cv2.imread(fp)
            if im is None:
                continue
            tiles.append(cv2.resize(im, (160, 128), interpolation=cv2.INTER_AREA))
        mont = np.hstack(tiles) if tiles else np.full((128, 160, 3), 230, np.uint8)
        axes[row, 0].imshow(cv2.cvtColor(mont, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_axis_off()
        axes[row, 0].set_title(f"{name} frames", fontsize=8)

        sid = seq.name
        gt = gt_all[idx]
        ours_p = ours_dir / f"{sid}_est_c2w.txt"
        base_p = base_dir / f"{sid}_est_c2w.txt"
        axes[row, 1].set_title("Sim(3)-aligned XY (mm)", fontsize=8)
        axes[row, 2].set_title("Per-frame translation error", fontsize=8)
        if ours_p.is_file():
            est = load_pose_txt(str(ours_p))
            gt_t, xy = _aligned_xy(est, gt)
            err = np.linalg.norm(xy - gt_t, axis=1)
            axes[row, 1].plot(gt_t[:, 0], gt_t[:, 1], color="#222222", lw=2.0, label="GT")
            axes[row, 1].plot(xy[:, 0], xy[:, 1], color="#0072B2", lw=1.8, label=ours_lab)
            axes[row, 2].plot(err, color="#0072B2", lw=1.4, label=ours_lab)
        if base_p.is_file():
            est = load_pose_txt(str(base_p))
            gt_t, xy = _aligned_xy(est, gt)
            err = np.linalg.norm(xy - gt_t, axis=1)
            axes[row, 1].plot(xy[:, 0], xy[:, 1], color="#E69F00", lw=1.3, label=base_lab)
            axes[row, 2].plot(err, color="#E69F00", lw=1.1, label=base_lab)
        axes[row, 1].set_aspect("equal", adjustable="datalim")
        axes[row, 1].tick_params(labelsize=6)
        axes[row, 1].set_xlabel("X (mm)", fontsize=7)
        axes[row, 1].set_ylabel("Y (mm)", fontsize=7)
        axes[row, 1].legend(fontsize=6, frameon=False)
        axes[row, 2].tick_params(labelsize=6)
        axes[row, 2].set_xlabel("selected frame", fontsize=7)
        axes[row, 2].set_ylabel("trans. err (mm)", fontsize=7)
        axes[row, 2].grid(alpha=0.25)
        axes[row, 2].legend(fontsize=6, frameon=False)
    fig.suptitle(
        "Real-video qualitative transfer on one C3VD clip and one StereoMIS clip. "
        "Trajectories are Sim(3)-aligned; protocols match Table 5.",
        fontsize=9,
        y=0.98,
    )
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    plot_motion_gallery(PAPER / "fig_motion_layers.jpg")
    plot_real_gallery(PAPER / "fig_real_qual.jpg")


if __name__ == "__main__":
    main()
