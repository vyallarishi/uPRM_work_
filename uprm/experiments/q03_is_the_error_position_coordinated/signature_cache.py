"""Stage 43 helper — ONE-TIME equation-signature cache.

WHY: stage43's P3 term needs a per-chain structural signature. Computing it
inline re-reads and re-regexes the whole 1500-problem generation root on
every run (minutes of scratch I/O, and it sat on the critical path before
any C1/C2 output appeared). This writes it once; stage43 --sig-cache then
reads a small local file.

Uses rung3_equations.signature(kind="values") -- the granularity rung3
measured as separating best (both-right pairs 0.859 mean Jaccard vs
both-wrong 0.561). No gold, no model.

Output schema, one line per chain:
    {"iid": ..., "k": ..., "sig": ["12", "36", ...]}

Usage: python stage43_sigcache.py [--root ROOT] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17            # patches gate.TIER to 37 langs
from icm.experiments.q08_earlier_8b_study.rung3_reasoning_structure import signature
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import load_pools
from ...paths import RUN_DIR


def safe_signature(text):
    """rung3.normalize_number raises decimal.InvalidOperation on absurd-
    magnitude digit runs: its try/except covers the Decimal() construction
    but NOT the later .quantize(Decimal(1)) (rung3_equations.py:76).
    stage8.norm guards this same case and returns None; rung3 does not.
    Measured: it killed this pass at ~30k/55k pools on the 1500 root.

    rung3_equations.py is existing experiment code and is left untouched;
    the guard lives here. A chain whose signature cannot be computed gets
    an EMPTY signature, which jaccard() already treats as unusable
    (returns None), so such chains are excluded from P3 rather than
    silently scored as dissimilar."""
    try:
        return signature(text or "", "values")
    except Exception:                                  # noqa: BLE001
        return frozenset()

ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
OUT = RUN_DIR / "stage43_signatures.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    print(f"loading pools from {args.root} ...", flush=True)
    pools = load_pools(args.root)
    print(f"  {len(pools)} pools", flush=True)

    written = empty = 0
    with open(args.out, "w", encoding="utf-8") as handle:
        for i, (iid, pool) in enumerate(sorted(pools.items())):
            for k, chain in enumerate(pool["chains"]):
                sig = safe_signature(chain["text"])
                if not sig:
                    empty += 1
                handle.write(json.dumps(
                    {"iid": iid, "k": k, "sig": sorted(sig)}) + "\n")
                written += 1
            if i % 5000 == 0:
                print(f"  {i}/{len(pools)} pools", flush=True)
    print(f"wrote {args.out}: {written} chains "
          f"({empty} with an empty signature, "
          f"{100*empty/max(written,1):.1f}%)")


if __name__ == "__main__":
    main()
