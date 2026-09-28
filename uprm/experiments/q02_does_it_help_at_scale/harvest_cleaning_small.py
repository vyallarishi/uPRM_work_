"""Stage 34 — CAN THE STEP-LEVEL SCORE CLEAN THE HARVEST?

WHERE THIS SITS:
  Toy 1 (stage32): per-chain step scoring, p_clean = P(j = T+1), gave AUC
    0.835 raw / 0.891 length-matched over 20,782 same-length pairs on
    clean-vs-broken. Length alone scores 0.448, i.e. the confound runs
    AGAINST the result. Best judge signal measured in this project.
  Toy 2 (stage33): uPRM's joint scoring did NOT reproduce (joint 0.848 vs
    independent 0.847 length-matched) and showed herding (poisoned
    neighbours: 0.724). Independent scoring only from here on.

THE TARGET: harvest contamination -- chains that agree with a WRONGLY
committed answer and therefore enter SFT with a wrong label. The local vote
cannot see them (they ARE the majority). Current cleaners:
    world-agreement (exact, no model)      AUC 0.926
    verdict judge (whole-chain)            AUC 0.677
    verdict judge on the double-consensus
      residue, where BOTH votes are wrong  AUC 0.731  (n=47)
The residue is the interesting part: world-agreement is blind there by
construction, so 0.731 is the number to beat.

THIS RUN: score harvest chains with Toy 1's independent p_clean and measure
  H1 genuine-vs-contaminated AUC overall (beat 0.677?)
  H2 the same on the double-consensus residue (beat 0.731?)  <- the point
  H3 two-stage: world-agreement filter first, then p_clean on survivors --
     purity/retention frontier vs the shipped 97.03% @ 0.7% cost
  H4 length-matched versions of H1/H2, since length-matching was the
     stricter (and better) number in Toy 1

Gold is used only to label genuine vs contaminated in evaluation.

Usage: --self-test | --dry-run | full via stage34_uprm_harvest.sbatch
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc, load_pools, match
from ..q01_does_step_scoring_work.toy1_first_broken_step import (CHAIN_MAX_CHARS, MIN_STEPS, SYSTEM,
                               score_from_logprobs, split_steps)

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
LANDSLIDE = 0.70
MAX_PROMPT_TOKENS = 3000
BATCH_SIZE = 16
N_GENUINE = 1200           # subsample: contamination is ~4.6%, so cap the
N_CONTAM = 100000          # majority class and keep every contaminated chain
SEED = 20260824


def harvest_rows(pools):
    """Chains agreeing with each landslide pool's committed answer, labelled
    genuine (commit correct) or contaminated (commit wrong). Mirrors
    harvest_evidence.py / harvest_frontier.py."""
    lang_plur = defaultdict(dict)
    for iid, pool in pools.items():
        lang_plur[iid.rsplit("_", 1)[-1]][pool["lang"]] = pool["plurality"]
    rows = []
    for iid in sorted(pools):
        pool = pools[iid]
        if pool["share"] < LANDSLIDE:
            continue
        commit = pool["plurality"]
        commit_right = match(commit, pool["gold"])
        problem = iid.rsplit("_", 1)[-1]
        others = [a for l, a in lang_plur[problem].items()
                  if l != pool["lang"]]
        world = (sum(match(commit, a) for a in others) / len(others)
                 if others else 0.0)
        for chain in pool["chains"]:
            if not match(chain["answer"], commit):
                continue
            if len(chain["text"]) > CHAIN_MAX_CHARS:
                continue
            steps = split_steps(chain["text"])
            if len(steps) < MIN_STEPS:
                continue
            rows.append({"iid": iid, "lang": pool["lang"],
                         "question": pool["question"], "steps": steps,
                         "genuine": commit_right, "world": world})
    return rows


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- harvest_rows labelling ---")
    def mk(lang, prob, plur, gold, answers):
        return (f"train_{lang}_{prob}",
                {"lang": lang, "gold": gold, "question": "q", "share": 1.0,
                 "plurality": plur,
                 "chains": [{"answer": a, "text": f"s1 line\ns2 line\n#### {a}"}
                            for a in answers]})
    pools = dict([mk("en", 1, "10", "10", ["10", "10", "20"]),
                  mk("de", 1, "99", "10", ["99", "99"]),
                  mk("fr", 1, "10", "10", ["10"])])
    rows = harvest_rows(pools)
    check("only commit-agreeing chains harvested", len(rows), 5)
    gen = [r for r in rows if r["genuine"]]
    con = [r for r in rows if not r["genuine"]]
    check("genuine from correct commits", len(gen), 3)
    check("contaminated from the wrong commit", len(con), 2)
    check("contaminated rows are the de pool",
          all(r["lang"] == "de" for r in con), True)
    check("world share computed for de",
          round(con[0]["world"], 3), 0.0)
    check("world share computed for en", round(gen[0]["world"], 3), 0.5)

    print("--- non-landslide pools excluded ---")
    split = dict([mk("en", 2, "10", "10", ["10", "20", "30", "40"])])
    split["train_en_2"]["share"] = 0.25
    check("split pool contributes nothing", len(harvest_rows(split)), 0)

    print("--- Eq. 6 reuse ---")
    lp = [(-0.1, -2.0), (-0.2, -1.5)]
    check("j=T+1 sums the pluses",
          round(score_from_logprobs(lp, 3, 2), 4), -0.3)

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def report(rows):
    scored = [r for r in rows if r.get("p_clean") is not None]
    gen = [r for r in scored if r["genuine"]]
    con = [r for r in scored if not r["genuine"]]
    results = {"n_scored": len(scored), "n_genuine": len(gen),
               "n_contaminated": len(con)}
    print("=" * 70)
    print(f"scored {len(scored)} harvest chains "
          f"({len(gen)} genuine / {len(con)} contaminated)")

    def length_matched(subset):
        strata = defaultdict(list)
        for r in subset:
            strata[len(r["steps"])].append(r)
        pairs = conc = 0.0
        for sub in strata.values():
            g = [r["p_clean"] for r in sub if r["genuine"]]
            c = [r["p_clean"] for r in sub if not r["genuine"]]
            if g and c:
                pairs += len(g) * len(c)
                conc += auc(g, c) * len(g) * len(c)
        return conc / pairs if pairs else None

    a1 = auc([r["p_clean"] for r in gen], [r["p_clean"] for r in con])
    lm1 = length_matched(scored)
    print(f"\nH1 overall genuine-vs-contaminated")
    print(f"   p_clean AUC {a1:.4f}   length-matched {lm1 if lm1 else float('nan'):.4f}"
          f"   [verdict judge 0.677]")
    results["H1"] = {"auc": round(a1, 4) if a1 else None,
                     "auc_length_matched": round(lm1, 4) if lm1 else None}

    aw = auc([r["world"] for r in gen], [r["world"] for r in con])
    print(f"   world-agreement AUC {aw:.4f} (reference, no model)")
    results["H1_world"] = {"auc": round(aw, 4) if aw else None}

    print(f"\nH2 double-consensus residue (world >= 0.5: both votes wrong)")
    for thresh in (0.3, 0.5, 0.7):
        sub = [r for r in scored if r["world"] >= thresh]
        g = [r["p_clean"] for r in sub if r["genuine"]]
        c = [r["p_clean"] for r in sub if not r["genuine"]]
        a = auc(g, c) if g and c else None
        lm = length_matched(sub)
        print(f"   world>={thresh}: {len(g)} genuine / {len(c)} contaminated"
              f"   AUC {a if a else float('nan'):.4f}"
              f"   length-matched {lm if lm else float('nan'):.4f}"
              f"{'   <- verdict judge 0.731 here' if thresh == 0.5 else ''}")
        results[f"H2_world_{thresh}"] = {
            "n_genuine": len(g), "n_contaminated": len(c),
            "auc": round(a, 4) if a else None,
            "auc_length_matched": round(lm, 4) if lm else None}

    print(f"\nH3 two-stage frontier: world >= 0.3, then drop lowest p_clean")
    kept = sorted([r for r in scored if r["world"] >= 0.3],
                  key=lambda r: r["p_clean"])
    base_con = sum(1 for r in kept if not r["genuine"])
    print(f"   {'drop':>6}{'kept':>8}{'contam':>8}{'purity':>9}{'genuine lost':>14}")
    frontier = []
    for drop in (0.0, 0.02, 0.05, 0.10, 0.20):
        k = int(drop * len(kept))
        survivors = kept[k:]
        c = sum(1 for r in survivors if not r["genuine"])
        lost = sum(1 for r in kept[:k] if r["genuine"])
        purity = 100 * (1 - c / max(len(survivors), 1))
        frontier.append({"drop": drop, "kept": len(survivors),
                         "contaminated": c, "purity": round(purity, 2),
                         "genuine_lost": lost})
        print(f"   {100*drop:>5.0f}%{len(survivors):>8}{c:>8}{purity:>8.2f}%"
              f"{lost:>14}")
    results["H3_frontier"] = frontier
    results["H3_baseline_contaminated"] = base_con
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    pools = load_pools(args.root)
    rows = harvest_rows(pools)
    rng = random.Random(SEED)
    gen = [r for r in rows if r["genuine"]]
    con = [r for r in rows if not r["genuine"]]
    rng.shuffle(gen)
    sample = gen[:N_GENUINE] + con[:N_CONTAM]
    rng.shuffle(sample)
    print(f"pools {len(pools)}; harvest chains {len(rows)} "
          f"({len(gen)} genuine / {len(con)} contaminated, "
          f"{100*len(con)/max(len(rows),1):.2f}% contamination)")
    print(f"scoring {len(sample)} "
          f"({sum(r['genuine'] for r in sample)} genuine / "
          f"{sum(not r['genuine'] for r in sample)} contaminated)")
    print(f"languages: {dict(sorted(Counter(r['lang'] for r in sample).items()))}")

    if args.dry_run:
        c = next(r for r in sample if not r["genuine"])
        print(f"\n--- a contaminated chain ({c['lang']}, world "
              f"{c['world']:.2f}) ---")
        for i, s in enumerate(c["steps"], 1):
            print(f"  [{i}] {s[:90]}")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    plus_id = tokenizer.encode(" +", add_special_tokens=False)[0]
    minus_id = tokenizer.encode(" -", add_special_tokens=False)[0]
    print("loading model in bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    import inspect
    last_logit = ({"logits_to_keep": 1}
                  if "logits_to_keep" in inspect.signature(model.forward).parameters
                  else {})

    entries = []
    for index, r in enumerate(sample):
        prefix = SYSTEM + "Problem: " + r["question"].strip()
        for t, step in enumerate(r["steps"], start=1):
            prompt = prefix + f"\n\nStep {t}: {step}\nJudge:"
            n_tokens = len(tokenizer.encode(prompt))
            if n_tokens <= MAX_PROMPT_TOKENS:
                entries.append((n_tokens, index, t, prompt))
            prefix = prefix + f"\n\nStep {t}: {step}\nJudge: +"
    entries.sort(key=lambda e: e[0])
    print(f"scoring {len(entries)} marker slots", flush=True)

    slots = defaultdict(dict)
    for start in range(0, len(entries), BATCH_SIZE):
        batch = entries[start:start + BATCH_SIZE]
        enc = tokenizer([p for _, _, _, p in batch], return_tensors="pt",
                        padding=True, pad_to_multiple_of=256).to(model.device)
        with torch.inference_mode():
            logits = model(**enc, use_cache=False, **last_logit).logits
            tail = logits[:, -1, :].float()
            pair = torch.stack([tail[:, plus_id], tail[:, minus_id]], dim=-1)
            lp = torch.log_softmax(pair, dim=-1)
        for row, (_, index, t, _) in enumerate(batch):
            slots[index][t] = (float(lp[row, 0]), float(lp[row, 1]))
        if start % (BATCH_SIZE * 100) == 0:
            print(f"  {start}/{len(entries)}", flush=True)
            torch.cuda.empty_cache()

    for index, r in enumerate(sample):
        got = slots.get(index, {})
        n_steps = len(r["steps"])
        if len(got) < n_steps:
            continue
        step_lp = [got[t] for t in range(1, n_steps + 1)]
        scores = {j: score_from_logprobs(step_lp, j, n_steps)
                  for j in range(1, n_steps + 2)}
        top = max(scores.values())
        lse = top + math.log(sum(math.exp(v - top) for v in scores.values()))
        r["p_clean"] = scores[n_steps + 1] - lse

    results = report(sample)
    (OUT_DIR / "stage34_results.json").write_text(json.dumps(results, indent=2))
    with (OUT_DIR / "stage34_chains.jsonl").open("w", encoding="utf-8") as h:
        for r in sample:
            if r.get("p_clean") is None:
                continue
            h.write(json.dumps({"iid": r["iid"], "lang": r["lang"],
                                "genuine": r["genuine"], "world": r["world"],
                                "n_steps": len(r["steps"]),
                                "p_clean": r["p_clean"]},
                               ensure_ascii=False) + "\n")
    print("\nwrote stage34_results.json and stage34_chains.jsonl")


if __name__ == "__main__":
    main()
