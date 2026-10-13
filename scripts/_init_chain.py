"""Follow the `init` chain of an EgoLite run and print the key training arguments per stage."""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
run = sys.argv[1]
seen = 0
while run and seen < 10:
    d = os.path.dirname(run) if run.endswith(".pth") else run
    p = os.path.join(d, "args.json")
    if not os.path.exists(p):
        print(f"[no args.json in {d}]")
        break
    a = json.load(open(p, encoding="utf-8"))
    keep = {k: a.get(k) for k in ("out", "iters", "batch", "clip", "lr", "gn_iters", "init", "data", "stride_p",
                                   "train_flow", "flow_net", "abs_depth", "no_ph", "seed", "params_M", "freeze_raft",
                                   "lam_p", "w_layer", "loss_w", "pose_loss", "flow_lr_mult") if k in a}
    extra = {k: v for k, v in a.items() if k not in keep and k not in ("workers", "val_every", "val_n", "val_abs",
                                                                     "temporal", "anchor_t", "head_extra", "ctx")}
    print(json.dumps(keep, ensure_ascii=False))
    if extra:
        print("   other:", json.dumps(extra, ensure_ascii=False)[:400])
    run = a.get("init")
    seen += 1
