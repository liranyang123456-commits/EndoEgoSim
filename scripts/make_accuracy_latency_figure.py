"""Accuracy-latency trade-off figure (protocol-A accuracy vs. GPU latency measured on one machine).

x: model time per frame in ms (log scale), RTX 5090 laptop GPU: batch size 1 for frame-pair models,
   64-frame clip divided by 64 for window and SLAM methods (results/latency/<tag>_<method>.json).
y: Sim(3) ATE mean (mm), protocol A: (a) simulated development list, (b) mean of the SCARED, StereoMIS and
   C3VD cohort means (methods without all three real cohorts are omitted there).
Writes bspc_paper/figs/fig_accuracy_latency.png and results/latency/accuracy_latency_points.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
TAG = sys.argv[1] if len(sys.argv) > 1 else "x18"
src = (ROOT / "scripts" / "audit_table_markers.py").read_text(encoding="utf-8").split("\nres = {}")[0]
ns = {"__file__": str(ROOT / "scripts" / "audit_table_markers.py")}
exec(compile(src, "hdr", "exec"), ns)
S, EGO, METHODS, per_seq = ns["S"], ns["EGO"], ns["METHODS"], ns["per_seq"]

# name in the paper -> (key in METHODS, latency json method, marker)
ENTRIES = {
    "EgoLite": (None, "egolite", "o"),
    "AF-SfMLearner": ("AF-SfMLearner", "afsfm", "o"),
    "EndoDAC": ("EndoDAC", "endodac", "o"),
    "Endo-FASt3r": ("Endo-FASt3r", "endofast3r", "o"),
    "Reloc3r-512": ("Reloc3r-512", "reloc3r", "o"),
    "$\\pi^3$": ("pi3", "pi3", "^"),
    "CUT3R": ("CUT3R", "cut3r", "^"),
    "DROID-SLAM": ("DROID-SLAM", "droid", "s"),
}
COL = {"EgoLite": "#2878B5", "AF-SfMLearner": "#59A14F", "EndoDAC": "#F28E2B", "Endo-FASt3r": "#B07AA1",
       "Reloc3r-512": "#9A9A9A", "$\\pi^3$": "#9467BD", "CUT3R": "#8C564B", "DROID-SLAM": "#E8A202"}


def cohort_mean(method_key, col):
    tag = EGO[col] if method_key is None else METHODS[method_key].get(col)
    if tag is None:
        return None
    d = per_seq(tag)
    return float(np.mean(list(d.values()))) if d else None


pts = {}
for name, (mk, lat, marker) in ENTRIES.items():
    p = ROOT / "results" / "latency" / f"{TAG}_{lat}.json"
    if not p.exists():
        print("no latency for", name)
        continue
    L = json.load(open(p, encoding="utf-8"))
    if "ms_per_frame" not in L:
        print("latency failed for", name, L.get("error"))
        continue
    dev = cohort_mean(mk, "dev")
    real = [cohort_mean(mk, c) for c in ("SCARED_A", "StereoMIS_A", "C3VD_A")]
    pts[name] = {"ms": L["ms_per_frame"], "dev": dev, "real": float(np.mean(real)) if all(r is not None for r in real) else None,
                 "marker": marker, "params_M": L.get("params_M"), "unit": L.get("unit"), "peak_mem_MB": L.get("peak_mem_MB")}
json.dump(pts, open(ROOT / "results" / "latency" / "accuracy_latency_points.json", "w", encoding="utf-8"), indent=1)

plt.rcParams.update({"font.size": 7.5})
fig, axs3 = plt.subplots(1, 3, figsize=(8.8, 3.05), gridspec_kw={"width_ratios": [1, 1, 0.72]})
axs = axs3[:2]
panels = [("dev", "Simulation: development list (92 sequences)"),
          ("real", "Real video: mean of SCARED, StereoMIS and C3VD")]
for ax, (key, title) in zip(axs, panels):
    xs, ys, names = [], [], []
    for name, v in pts.items():
        if v[key] is None:
            continue
        xs.append(v["ms"]); ys.append(v[key]); names.append(name)
        ax.scatter(v["ms"], v[key], marker=v["marker"], s=52 if name == "EgoLite" else 30, color=COL[name],
                   edgecolor="k" if name == "EgoLite" else "none", linewidth=1.2, zorder=3)
    # Pareto front: no other method is both faster and more accurate
    order = np.argsort(xs)
    best, front = np.inf, []
    for i in order:
        if ys[i] < best:
            front.append(i); best = ys[i]
    ax.plot([xs[i] for i in front], [ys[i] for i in front], ls="--", lw=0.9, color="#555", zorder=2)
    # label offsets in points; a few labels are moved so that neighbouring points stay readable
    OFF = {("dev", "Endo-FASt3r"): (-7, 6, "right"), ("dev", "DROID-SLAM"): (7, -8, "left"),
           ("real", "EndoDAC"): (7, 6, "left"), ("real", "AF-SfMLearner"): (7, -8, "left")}
    for x, y, n in zip(xs, ys, names):
        dx, dy, ha = OFF.get((key, n), (7, 0, "left"))
        ax.annotate(n, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.6, va="center", ha=ha,
                    fontweight="bold" if n == "EgoLite" else "normal")
    ax.set_xscale("log")
    ax.set_xlabel("model time per frame (ms, log)")
    ax.set_ylabel("Sim(3) ATE mean (mm)")
    ax.set_title(title, fontsize=7.8)
    ax.grid(alpha=0.25, which="both")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi * 4.5)
# (c) EgoLite on the fleet GPUs, fp32, ms per frame pair at batch size 1 and 16
gpus = [("RTX 5090\nlaptop", "rtx5090laptop"), ("RTX 2080 Ti", "rtx2080ti"), ("GTX 1070", "gtx1070")]
b1, b16, labs = [], [], []
for lab, tag in gpus:
    p = ROOT / "results" / "latency" / f"{tag}_egolite.json"
    if not p.exists():
        continue
    rows = json.load(open(p, encoding="utf-8"))["rows"]
    pick = {r["batch"]: r["median_ms_per_pair"] for r in rows if r.get("dtype") == "fp32" and "median_ms_per_pair" in r}
    if 1 in pick and 16 in pick:
        labs.append(lab); b1.append(pick[1]); b16.append(pick[16])
ax = axs3[2]
xx = np.arange(len(labs))
bars1 = ax.bar(xx - 0.2, b1, 0.38, color="#2878B5", label="batch size 1")
bars16 = ax.bar(xx + 0.2, b16, 0.38, color="#9CC3E6", label="batch size 16")
for bars in (bars1, bars16):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.0f}", ha="center", fontsize=6.6)
ax.set_xticks(xx, labs, fontsize=6.6)
ax.set_ylabel("EgoLite ms per frame pair (fp32)")
ax.set_title("EgoLite on other GPUs", fontsize=7.8)
ax.set_ylim(0, max(b1) * 1.2)
ax.legend(fontsize=6.4, frameon=False, loc="upper left")
ax.grid(axis="y", alpha=0.25)
json.dump({"gpus": labs, "batch1_ms": b1, "batch16_ms": b16}, open(ROOT / "results" / "latency" / "egolite_gpus.json", "w"), indent=1)
handles = [plt.Line2D([], [], marker="o", ls="", color="#888", label="frame-pair model, batch size 1"),
           plt.Line2D([], [], marker="^", ls="", color="#888", label="64-frame clip, per-frame average"),
           plt.Line2D([], [], marker="s", ls="", color="#888", label="SLAM over a 64-frame clip"),
           plt.Line2D([], [], ls="--", color="#555", label="fastest-and-most-accurate front")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.4, frameon=False)
fig.tight_layout(rect=(0, 0.075, 1, 1))
out = ROOT / "bspc_paper" / "figs" / "fig_accuracy_latency.png"
fig.savefig(out, dpi=200)
print("written", out)
for n, v in pts.items():
    print(f"{n:16s} {v['ms']:8.2f} ms/frame  dev {v['dev']}  real {v['real']}  params {v['params_M']}  mem {v['peak_mem_MB'] and round(v['peak_mem_MB'])} MB")
