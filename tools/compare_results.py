"""Deep-compare a re-run results JSON against the recorded one.

    python tools/compare_results.py ~/multilingual_icm/data/uPRM_work_/runs/<new>.json ~/multilingual_icm/data/uPRM_work_/recorded_results/<recorded>.json

Exit 0 if identical (floats compared exactly unless --tol is given).
"""
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("new")
ap.add_argument("recorded")
ap.add_argument("--tol", type=float, default=0.0)
args = ap.parse_args()

diffs = []


def walk(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a:
                diffs.append(f"{path}.{k}: missing in re-run")
            elif k not in b:
                diffs.append(f"{path}.{k}: not in recorded")
            else:
                walk(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > args.tol:
            diffs.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        diffs.append(f"{path}: {a!r} vs {b!r}")


walk(json.load(open(args.new)), json.load(open(args.recorded)), "$")
if diffs:
    print(f"DIFFERS: {len(diffs)} values")
    for d in diffs[:50]:
        print("  " + d)
    sys.exit(1)
print(f"IDENTICAL: {args.new} == {args.recorded}")
