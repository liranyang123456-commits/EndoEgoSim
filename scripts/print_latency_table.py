"""Print the latency JSONs of results/latency/<tag>_<method>.json as a table."""
import json
import os
import sys

tag = sys.argv[1] if len(sys.argv) > 1 else "x18"
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "latency")
for m in ["egolite", "afsfm", "endodac", "endofast3r", "reloc3r", "pi3", "cut3r", "droid"]:
    p = os.path.join(root, f"{tag}_{m}.json")
    if not os.path.exists(p):
        print(f"{m:11s} (no file)")
        continue
    d = json.load(open(p, encoding="utf-8"))
    if "error" in d:
        print(f"{m:11s} ERROR {d['error']}")
        continue
    pm = d.get("params_M")
    mem = d.get("peak_mem_MB")
    print(f"{m:11s} {d.get('unit', '-'):6s} ms/frame {d['ms_per_frame']:8.2f}  p95/call {d['p95_ms']:9.1f}  "
          f"params {pm if pm is None else round(pm, 1)}  mem {mem if mem is None else round(mem)} MB  in {d.get('input')}"
          f"  gpu {d.get('gpu')}")
