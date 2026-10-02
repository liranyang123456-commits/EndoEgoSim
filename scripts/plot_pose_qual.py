"""Qualitative pose figure for the MIA paper.

Uses already-written simtest92 trajectories (Sim(3) aligned). Does not invent scores.

  python scripts/plot_pose_qual.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.align import align_trajectories
from endosim.eval.metrics import load_pose_txt
from endosim.eval.protocol import list_color_frames, select_frame_indices

ROOT = Path(__file__).resolve().parents[1]
SOTA = ROOT / "results" / "sota"
PAPER = ROOT / "mia_paper"
MIS = Path(r"D:\MIS_Pose_Track_Re3D")

C = {
    "gt": "#222222",
    "ours": "#0072B2",
    "droid": "#E69F00",
    "reloc": "#999999",
    "ink": "#222222",
    "muted": "#666666",
}

METHODS = [
    ("ours_v7_final", "MD-VGGT-R*", C["ours"], 1.8),
    ("droid_simtest", "DROID-SLAM", C["droid"], 1.4),
    ("reloc3r_simtest", "Reloc3r-512", C["reloc"], 1.2),
]


def _seq_map() -> dict[str, Path]:
    out = {}
    lst = ROOT / "lists" / "simtest92.txt"
    if lst.is_file():
        for ln in lst.read_text(encoding="utf-8").splitlines():
            p = Path(ln.strip())
            if p.is_dir():
                out[p.name] = p
    for split in ("test", "val", "train"):
        for p in (ROOT / "sim_data" / split).glob("seq_*"):
            out.setdefault(p.name, p)
    return out


def _load_summary(tag: str) -> dict[str, dict]:
    p = SOTA / tag / "summary.json"
    if not p.is_file():
        return {}
    recs = json.loads(p.read_text(encoding="utf-8")).get("records", [])
    return {r["seq_id"]: r for r in recs if r.get("seq_id") and "error" not in r}


def _est_path(tag: str, sid: str) -> Path:
    return SOTA / tag / f"{sid}_est_c2w.txt"


def _aligned_xy(est: np.ndarray, gt: np.ndarray):
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]
    aligned, _ = align_trajectories(est, gt, with_scale=True)
    err = np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=1)
    return gt[:, :3, 3], aligned[:, :3, 3], err


def _montage(seq_dir: Path, idx: np.ndarray, n_show: int = 4) -> np.ndarray:
    show = np.unique(np.linspace(0, len(idx) - 1, min(n_show, len(idx))).astype(int))
    frames = list_color_frames(str(seq_dir), indices=idx[show])
    tiles = []
    for i, fp in enumerate(frames):
        im = cv2.imread(fp)
        if im is None:
            continue
        mk_p = seq_dir / "object_mask" / Path(fp).name
        if mk_p.is_file():
            mk = cv2.imread(str(mk_p), cv2.IMREAD_UNCHANGED)
            if mk is not None and mk.shape[:2] == im.shape[:2]:
                ov = im.copy()
                ov[mk == 2] = (0.55 * ov[mk == 2] + np.array([0, 180, 255])).astype(np.uint8)
                ov[mk == 3] = (0.55 * ov[mk == 3] + np.array([255, 70, 70])).astype(np.uint8)
                im = ov
        im = cv2.resize(im, (160, 128), interpolation=cv2.INTER_AREA)
        tiles.append(im)
    if not tiles:
        return np.full((128, 160, 3), 230, np.uint8)
    return np.hstack(tiles)


def _colorize_height(h: np.ndarray) -> np.ndarray:
    hn = h.copy()
    m = hn > 0
    if m.any():
        lo, hi = np.percentile(hn[m], [2, 98])
        hn = np.clip((hn - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _flow_hsv(flow: np.ndarray) -> np.ndarray:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = 255
    mx = float(np.percentile(mag, 95)) if mag.size else 1.0
    hsv[..., 2] = np.clip(mag / max(mx, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _load_selection() -> dict:
    p = SOTA / "qual_selection.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _panel_traj(ax, gt_t, tracks, title, ate_txt):
    ax.plot(gt_t[:, 0], gt_t[:, 1], color=C["gt"], lw=2.0, label="GT")
    ax.scatter(gt_t[0, 0], gt_t[0, 1], c="#009E73", s=22, zorder=5)
    ax.scatter(gt_t[-1, 0], gt_t[-1, 1], c="#D55E00", s=22, zorder=5)
    for name, xy, col, lw in tracks:
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=lw, label=name)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("X (mm)", fontsize=7)
    ax.set_ylabel("Y (mm)", fontsize=7)
    ax.text(0.02, 0.98, ate_txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=6.5, color=C["ink"],
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))


def plot_sim(out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seqs = _seq_map()
    sel = _load_selection()
    labels = [
        ("high_success", "High-reference"),
        ("mid_typical", "Mid-reference"),
        ("low_recovery", "Low-reference example"),
        ("disaster", "High-error example"),
    ]
    summaries = {tag: _load_summary(tag) for tag, _, _, _ in METHODS}
    fig, axes = plt.subplots(4, 3, figsize=(11.2, 10.6))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.05,
                        wspace=0.28, hspace=0.42)

    for row, (key, bucket) in enumerate(labels):
        sid = sel.get(key)
        ax0, ax1, ax2 = axes[row]
        if not sid or sid not in seqs:
            for ax in (ax0, ax1, ax2):
                ax.set_axis_off()
            continue
        seq_dir = seqs[sid]
        gt_all = load_pose_txt(str(seq_dir / "pose_c2w.txt"))
        idx = select_frame_indices(gt_all, "uniform", max_frames=64)
        gt = gt_all[idx]
        mont = _montage(seq_dir, idx)
        ax0.imshow(cv2.cvtColor(mont, cv2.COLOR_BGR2RGB))
        ax0.set_axis_off()
        meta = {}
        mp = seq_dir / "meta.json"
        if mp.is_file():
            meta = json.loads(mp.read_text(encoding="utf-8"))
        refs = [r for r in meta.get("reference_fraction", []) if r is not None]
        ref = float(np.mean(refs)) if refs else summaries["ours_v7_final"].get(sid, {}).get("reference_fraction")
        ax0.set_title(
            f"{bucket}: {sid}\n"
            f"ref={ref:.2f}  {meta.get('motion_type', '')}/{meta.get('scene_kind', '')}",
            fontsize=8)

        tracks = []
        bits = []
        err_ours = None
        for tag, name, col, lw in METHODS:
            ep = _est_path(tag, sid)
            if not ep.is_file():
                continue
            est = load_pose_txt(str(ep))
            n = min(len(est), len(gt))
            gt_t, xy, err = _aligned_xy(est[:n], gt[:n])
            tracks.append((name, xy, col, lw))
            rec = summaries[tag].get(sid, {})
            ate = rec.get("ate_sim3", {}).get("rmse")
            if ate is None:
                ate = float(np.sqrt((err ** 2).mean()))
            bits.append(f"{name} {ate:.2f}")
            if tag == "ours_v7_final":
                err_ours = err
                ax2.plot(err, color=col, lw=1.4, label=name)
            else:
                ax2.plot(err, color=col, lw=1.0, alpha=0.9, label=name)
        if tracks:
            _panel_traj(ax1, gt_t, tracks, "Sim(3)-aligned XY (mm)", "  |  ".join(bits))
        else:
            ax1.set_axis_off()
        ax2.set_xlabel("selected frame", fontsize=7)
        ax2.set_ylabel("trans. err (mm)", fontsize=7)
        ax2.set_title("Per-frame translation error", fontsize=8)
        ax2.tick_params(labelsize=6)
        ax2.grid(alpha=0.25)
        if err_ours is not None:
            ax2.legend(fontsize=6, loc="upper right", frameon=False)
        if row == 0:
            ax1.legend(fontsize=6, loc="best", frameon=False)

    fig.suptitle(
        "Qualitative camera trajectories on EndoEgoSim simtest92. "
        "Alignment is Sim(3); *MD-VGGT-R uses the benchmark-tuned routing threshold.",
        fontsize=9, y=0.995)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("wrote", out_png)


def plot_meshrts(out_png: Path, sid: str | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_meshrts import dense_flow, flow_motion_scores, height_field, _resize_max

    seqs = _seq_map()
    if not sid or sid not in seqs:
        sid = next(iter(seqs))
    seq_dir = seqs[sid]
    frames = list_color_frames(str(seq_dir))
    if len(frames) < 2:
        return
    i0, i1 = 0, 1
    im0 = cv2.imread(frames[i0])
    im1 = cv2.imread(frames[i1])
    im0s, _ = _resize_max(im0, 384)
    im1s, _ = _resize_max(im1, 384)
    h0, m0 = height_field(im0s, alpha=68.0, s_z=40.0)
    g0 = cv2.cvtColor(im0s, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(im1s, cv2.COLOR_BGR2GRAY)
    flow = dense_flow(g0, g1)
    meta = json.loads((seq_dir / "meta.json").read_text(encoding="utf-8")) if (seq_dir / "meta.json").is_file() else {}
    cam = meta.get("camera") or {}
    cx = float(cam.get("cx", im0s.shape[1] / 2))
    cy = float(cam.get("cy", im0s.shape[0] / 2))
    rad, tang, mag = flow_motion_scores(flow, cx, cy, mask=m0)
    rot_like = mag > 0.8 and rad < 0.35 * mag

    fig, axes = plt.subplots(1, 4, figsize=(11.2, 2.85))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.78, bottom=0.08, wspace=0.08)
    titles = [
        f"RGB  {sid}",
        "Height field $h$ (p=68)",
        "DIS flow",
        f"Gate  mag={mag:.2f}  rad={rad:.2f}\n"
        f"{'rotation (t=0)' if rot_like else 'translation pair'}",
    ]
    images = [
        cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(_colorize_height(h0), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(_flow_hsv(flow), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(im1s, cv2.COLOR_BGR2RGB),
    ]
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title(t, fontsize=8)
    fig.suptitle(
        "Mesh-RTS scoring path on one pair (Sobel+percentile height, DIS+FB, rotation gate). "
        "Not metric depth.",
        fontsize=9)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("wrote", out_png)


def _copy(src: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / src.name
    target.write_bytes(src.read_bytes())


def main():
    sel = _load_selection()
    pose = PAPER / "fig_pose_qual.jpg"
    mesh = PAPER / "fig_meshrts_qual.jpg"
    plot_sim(pose)
    plot_meshrts(mesh, sel.get("high_success") or sel.get("mid_typical"))
    if MIS.is_dir():
        for p in (pose, mesh):
            if p.is_file():
                _copy(p, MIS)
                _copy(p, MIS / "mia_paper") if (MIS / "mia_paper").is_dir() else None
    print("done")


if __name__ == "__main__":
    main()
