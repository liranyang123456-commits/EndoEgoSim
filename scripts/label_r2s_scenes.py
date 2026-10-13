"""Anatomy labels for the real-to-sim scenes (SCARED keyframes) and a few test frames.

Writes results/anatomy_seg/scenes/<scene>.png (class ids, full keyframe resolution)
and a colour overview figure.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from train_anatomy_seg import CLASSES, Head, features, load_backbone, prep  # noqa: E402
from endosim.dataset.r2s_generator import scared_keyframes, scene_name  # noqa: E402
from endosim.eval.protocol import list_color_frames  # noqa: E402

PALETTE = np.array([[0, 0, 0], [255, 220, 120], [160, 30, 30], [230, 140, 60], [250, 240, 80],
                    [120, 120, 255], [80, 200, 120], [200, 0, 120], [0, 255, 255], [60, 60, 200],
                    [80, 160, 40], [0, 120, 200], [255, 120, 200]], np.uint8)


@torch.no_grad()
def main():
    out = "results/anatomy_seg/scenes"
    os.makedirs(out, exist_ok=True)
    bb = load_backbone().cuda()
    head = Head().cuda()
    head.load_state_dict(torch.load("results/anatomy_seg/head.pth", map_location="cpu")["head"])
    head.eval()
    items = [(scene_name(k), os.path.join(k, "Left_Image.png")) for k in scared_keyframes((5, 6, 7))]
    for q in (1, 4, 7):
        d = f"sim_data/real_refs/stereomis_extracted_StereoMIS_0_0_1_P2_{q}"
        items.append((f"stereomis_P2_{q}", list_color_frames(d, indices=[800])[0]))
    tiles = []
    for name, f in items:
        bgr = cv2.imread(f)
        x = torch.from_numpy(prep(bgr).transpose(2, 0, 1)[None].copy()).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fe = features(bb, x)
        lab = F.interpolate(head(fe.float()), size=bgr.shape[:2], mode="bilinear",
                            align_corners=False).argmax(1)[0].byte().cpu().numpy()
        cv2.imwrite(os.path.join(out, f"{name}.png"), lab)
        frac = {CLASSES[c]: round(float((lab == c).mean()), 2) for c in np.unique(lab) if (lab == c).mean() > 0.02}
        print(name, frac, flush=True)
        col = PALETTE[lab][..., ::-1]
        vis = cv2.addWeighted(bgr, 0.5, col, 0.5, 0)
        tiles.append(cv2.resize(np.concatenate([bgr, vis], 1), (480, 192)))
    rows = [np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
    cv2.imwrite("results/anatomy_seg/scenes_overview.jpg", np.concatenate(rows, 0))


if __name__ == "__main__":
    main()
