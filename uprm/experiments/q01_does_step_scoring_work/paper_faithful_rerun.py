"""Stage 42 — uPRM TOY 1 RE-RUN: two isolated deviations from the paper.

WHY: stage32 reached p_clean AUC 0.835 (readout A) but localization at
CHANCE (readout B: 17.6% exact vs 21.7% chance). Re-reading uPRM
(arXiv 2605.10158) against stage32 found three deviations from the paper.
This file tests the two that are fixable on our hardware, each behind its
own flag so the contributions are separable.

DEVIATION 1 — the absorbing-error clause (paper Appendix C.1, p.16).
  The paper's system prompt contains a rule stage32's does not:
    "For each new problem, once you determine that a step is incorrect,
     you must consider all subsequent steps for that problem to also be
     incorrect, and respond with '-' for them as well."
  That clause is what makes '-' ABSORBING, which is exactly the property
  argmax_j needs and p_clean does not. Its absence is a candidate
  explanation for the A-works/B-fails asymmetry.
  --system paper | stage32

DEVIATION 2 — step partitioning (measured, not from the paper).
  stage32 splits steps on '\n' alone. Measured over all 35,802 phaseA
  chains: only 1.7% have zero newlines, BUT 17.2% yield < MIN_STEPS steps
  and are dropped from every PRM measurement, and the drop is
  tier-correlated (ml 34.5%, eu 34.3% vs en 6.7%, am 3.1%).
  Those chains are NOT junk: of the 610 zero-newline chains, 451 carry
  >= 2 sentence breaks and 279 carry >= 2 '<<a*b=c>>' annotations
  (e.g. train_ar_118 is a complete correct 5-step derivation on one line).
  --split newline | fallback
  `fallback` splits on newlines first; if that yields < MIN_STEPS, it
  re-splits the chain on sentence boundaries and '<<...>>' annotation
  boundaries. MIN_STEPS is still enforced afterwards, so genuine
  one-liners (train_ar_116 = '**المثلة 1**:') stay dropped.

NOT ADDRESSED (deviation 3, not fixable here): the paper uses
Qwen2.5-14B-INSTRUCT with a multi-turn chat template (Figure C1) where each
marker is an assistant turn. We use Qwen3-14B-Base with a flat transcript.
No instruct sibling is on scratch. Any null here remains partly a statement
about base models.

Eq. 6 itself is NOT touched: audit_eq6.py proved stage32's indexing,
both boundaries, the all-plus-prefix reuse (provably exact) and the
logsumexp correct. This file imports score_from_logprobs unchanged.

ARMS (2x2, all four run in one job; identical chains across arms by
construction -- the sample is built ONCE under the union of both splitters
so every arm scores the SAME chain set and AUCs are comparable):
    s32_newline    stage32 replication anchor  (expect A ~= 0.835)
    s32_fallback   split fix only
    paper_newline  system-prompt fix only
    paper_fallback both

PRE-REGISTERED PREDICTIONS (gold used only in scoring, never in prompts):
  P1  paper_* localization (readout B exact rate) > chance by >= 5 points,
      where stage32 was 4.1 points BELOW chance. This is the claim: the
      absorbing clause is what argmax_j was missing.
  P2  |paper_* p_clean AUC - s32_* p_clean AUC| < 0.03. The clause should
      move localization, not the whole-chain signal. If p_clean moves a
      lot instead, the clause is acting as a generic prompt improvement
      and P1's interpretation is unsafe.
  P3  *_fallback recovers >= 10% more chains into the scored population
      than *_newline, and the low-tier share of the sample rises.
  P4  *_fallback p_clean AUC within 0.05 of *_newline. The recovered
      chains are harder (shorter, low-resource); a large DROP means the
      0.835 headline was partly a survivorship artifact -- which is
      itself the finding, and would be reported as such.

FAILURE OF P1 closes the localization hypothesis: if the paper's own
system prompt does not lift argmax_j above chance, then the base model
cannot localize errors regardless of prompt, and only p_clean survives.

Gold-blind: gold appears in build_sample class balance and in report()
only, never in any prompt. Sample construction is seeded and identical
across arms.

Usage: --self-test | --dry-run | full via stage42_uprm_faithful.sbatch
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from icm.experiments.q07_audits.fix_arithmetic_precedence import first_bad_step          # FIXED precedence checker
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc, load_pools, match
from .toy1_first_broken_step import score_from_logprobs   # Eq. 6, audited correct

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
LANGS = ("en", "de", "es", "fr", "ru", "hi", "ar", "ko", "fa",
         "am", "sw", "kn", "eu", "ml", "is")
MAX_STEPS = 12
MIN_STEPS = 2
CHAIN_MAX_CHARS = 1400
MAX_PROMPT_TOKENS = 3000
BATCH_SIZE = 16
N_PER_CLASS = 350
N_SLIP_MIN = 250
SEED = 20260824                      # same as stage32, for anchor comparability

# stage32's prompt: no absorbing-error clause.
SYSTEM_S32 = ("You are a strict mathematical reasoning judge. After each step "
              "you respond with + if the step is correct and - if it is "
              "incorrect.\n\n")

# The paper's system prompt, Appendix C.1 p.16, transposed out of chat format
# into our flat transcript. The final clause is the thing under test.
SYSTEM_PAPER = (
    "You are a strict mathematical reasoning judge.\n"
    "Your task is to evaluate one individual reasoning step of a math "
    "problem at a time.\n"
    "- If the step is mathematically correct, respond with +.\n"
    "- If the step is mathematically incorrect or logically flawed, "
    "respond with -.\n"
    "- Do not provide any explanation, comment, or feedback - only respond "
    "with + or -, and nothing else.\n"
    "- Each input is either a single reasoning step or a new problem "
    "followed by its first reasoning step. In both cases, evaluate only the "
    "validity of the reasoning step.\n"
    "- For each new problem, once you determine that a step is incorrect, "
    "you must consider all subsequent steps for that problem to also be "
    "incorrect, and respond with - for them as well.\n"
    "Your response must only be one of these two symbols: + or -.\n\n")

SYSTEMS = {"stage32": SYSTEM_S32, "paper": SYSTEM_PAPER}

# Sentence-boundary fallback. Split AFTER a terminator (so the terminator
# stays with its step) when followed by whitespace. Includes the Arabic full
# stop, Devanagari danda, and CJK/ideographic stop, since the drop is
# concentrated in non-Latin scripts.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。۔।؟])\s+")
# An annotation boundary: split after a '<<...>>NNN' calculator span when the
# next character starts new prose. Used only if sentence splitting failed.
_CALC_SPLIT_RE = re.compile(r"(?<=>>)(?=\s*\S)")


def _clean_steps(parts):
    """Strip, drop empties and the '#### N' answer line, cap at MAX_STEPS."""
    steps = [p.strip() for p in parts
             if p.strip() and not p.strip().lstrip().startswith("####")]
    return steps[:MAX_STEPS]


def split_steps_newline(chain_text):
    """stage32's rule, reproduced exactly: non-empty lines, minus '#### N'."""
    return _clean_steps(chain_text.split("\n"))


def split_steps_fallback(chain_text):
    """Newlines first; if that under-segments, fall back to sentence
    boundaries, then to calculator-annotation boundaries. MIN_STEPS is
    enforced by the caller, so genuine one-liners still drop out."""
    steps = split_steps_newline(chain_text)
    if len(steps) >= MIN_STEPS:
        return steps
    # Strip the answer line before re-splitting, so '#### N' cannot become
    # a sentence of its own and manufacture a spurious second step.
    body = "\n".join(line for line in chain_text.split("\n")
                     if not line.strip().lstrip().startswith("####"))
    sent = _clean_steps(_SENT_SPLIT_RE.split(body))
    if len(sent) >= MIN_STEPS:
        return sent
    calc = _clean_steps(_CALC_SPLIT_RE.split(body))
    if len(calc) >= MIN_STEPS:
        return calc
    return steps


SPLITTERS = {"newline": split_steps_newline, "fallback": split_steps_fallback}


def build_transcript(question, steps, j, system, pos=" +", neg=" -"):
    """Grading transcript asserting 'first error at step j' (j = len+1 means
    no error). Ends right after the final marker."""
    lines = [system + "Problem: " + question.strip()]
    for index, step in enumerate(steps, start=1):
        marker = neg if index == j else pos
        lines.append(f"Step {index}: {step}\nJudge:{marker}")
        if index == j:
            break
    return "\n\n".join(lines)


def marker_slots(question, steps, j, system):
    """Prompts whose NEXT token is the marker for each graded step."""
    slots = []
    prefix = system + "Problem: " + question.strip()
    for index, step in enumerate(steps, start=1):
        prompt = prefix + f"\n\nStep {index}: {step}\nJudge:"
        slots.append((prompt, index == j))
        marker = " -" if index == j else " +"
        prefix = prefix + f"\n\nStep {index}: {step}\nJudge:{marker}"
        if index == j:
            break
    return slots


def build_sample(pools, rng, splitters):
    """ONE sample, shared by every arm. A chain is admitted only if EVERY
    splitter yields >= MIN_STEPS for it, so all four arms score exactly the
    same chains and their AUCs are directly comparable. Per-splitter step
    lists are carried on the row.

    Readout P3 (how many chains a splitter recovers) is measured separately
    in coverage_report(), over the whole population -- not on this sample,
    which is deliberately restricted to the intersection."""
    clean, broken = [], []
    for iid in sorted(pools):
        pool = pools[iid]
        if pool["lang"] not in LANGS:
            continue
        for chain in pool["chains"]:
            if len(chain["text"]) > CHAIN_MAX_CHARS:
                continue
            steps_by = {name: fn(chain["text"])
                        for name, fn in splitters.items()}
            if any(len(s) < MIN_STEPS for s in steps_by.values()):
                continue
            is_gold = match(chain["answer"], pool["gold"])
            row = {"iid": iid, "question": pool["question"],
                   "steps_by": steps_by, "is_gold": is_gold,
                   "answer": chain["answer"],
                   "bad_step_by": {name: first_bad_step(s)
                                   for name, s in steps_by.items()}}
            (clean if is_gold else broken).append(row)
    rng.shuffle(clean)
    rng.shuffle(broken)
    # Force-include verifiable-slip chains for readout B, as stage32 did.
    # A chain counts as a slip chain if ANY splitter localizes a false
    # equation in it (the splitters can disagree about which step it lands on).
    slips = [r for r in broken if any(r["bad_step_by"].values())][:N_SLIP_MIN]
    slip_ids = {id(r) for r in slips}
    rest = [r for r in broken if id(r) not in slip_ids]
    broken_sample = slips + rest[:max(N_PER_CLASS - len(slips), 0)]
    return clean[:len(broken_sample)] + broken_sample


def coverage_report(pools):
    """P3: how much of the population each splitter admits, by tier.
    This is the measurement that motivated deviation 2; it needs no GPU."""
    from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import TIER
    stats = defaultdict(Counter)
    for iid in sorted(pools):
        pool = pools[iid]
        if pool["lang"] not in LANGS:
            continue
        for chain in pool["chains"]:
            if len(chain["text"]) > CHAIN_MAX_CHARS:
                continue
            s = stats[pool["lang"]]
            s["chains"] += 1
            for name, fn in SPLITTERS.items():
                if len(fn(chain["text"])) >= MIN_STEPS:
                    s[f"kept_{name}"] += 1
    print("=" * 78)
    print("P3 — STEP-PARTITION COVERAGE (no model involved)")
    print("=" * 78)
    print(f"{'lang':<6}{'tier':<6}{'chains':>8}{'newline':>9}{'%':>7}"
          f"{'fallback':>10}{'%':>7}{'recovered':>11}")
    tot = Counter()
    by_tier = defaultdict(Counter)
    for lang in sorted(stats, key=lambda l: (TIER.get(l, "?"), l)):
        s = stats[lang]
        n = max(s["chains"], 1)
        rec = s["kept_fallback"] - s["kept_newline"]
        tier = TIER.get(lang, "?")
        tot.update(s)
        by_tier[tier].update(s)
        print(f"{lang:<6}{tier:<6}{s['chains']:>8}{s['kept_newline']:>9}"
              f"{100*s['kept_newline']/n:>6.1f}%{s['kept_fallback']:>10}"
              f"{100*s['kept_fallback']/n:>6.1f}%{rec:>11}")
    n = max(tot["chains"], 1)
    print(f"\n{'ALL':<12}{tot['chains']:>8}{tot['kept_newline']:>9}"
          f"{100*tot['kept_newline']/n:>6.1f}%{tot['kept_fallback']:>10}"
          f"{100*tot['kept_fallback']/n:>6.1f}%"
          f"{tot['kept_fallback']-tot['kept_newline']:>11}")
    rel = ((tot["kept_fallback"] - tot["kept_newline"])
           / max(tot["kept_newline"], 1))
    print(f"\nP3 bar: fallback admits >= 10% more chains than newline. "
          f"Got +{100*rel:.1f}%  -> {'PASS' if rel >= 0.10 else 'FAIL'}")
    print("\nby tier:")
    for tier in ("high", "mid", "low"):
        s = by_tier.get(tier)
        if not s:
            continue
        n = max(s["chains"], 1)
        print(f"  {tier:<5} newline {100*s['kept_newline']/n:5.1f}%  "
              f"fallback {100*s['kept_fallback']/n:5.1f}%  "
              f"(+{s['kept_fallback']-s['kept_newline']} chains)")
    return {lang: dict(s) for lang, s in stats.items()}


def report(sample, arms):
    """Readouts A (p_clean AUC), B (localization) and C (sanity) per arm."""
    results = {}
    for arm in arms:
        scored = [s for s in sample if s.get(f"p_clean::{arm}") is not None]
        if not scored:
            print(f"\n[{arm}] nothing scored")
            continue
        split_name = arm.split("_", 1)[1]
        pos = [s[f"p_clean::{arm}"] for s in scored if s["is_gold"]]
        neg = [s[f"p_clean::{arm}"] for s in scored if not s["is_gold"]]
        a = auc(pos, neg)
        loc = [s for s in scored
               if not s["is_gold"] and s["bad_step_by"][split_name]]
        hits = sum(1 for s in loc
                   if s[f"j_hat::{arm}"] == s["bad_step_by"][split_name])
        near = sum(1 for s in loc
                   if abs(s[f"j_hat::{arm}"]
                          - s["bad_step_by"][split_name]) <= 1)
        chance = (sum(1.0 / (len(s["steps_by"][split_name]) + 1) for s in loc)
                  / len(loc)) if loc else 0.0
        exact_rate = hits / max(len(loc), 1)
        cl = [s for s in scored if s["is_gold"]]
        at_end = sum(1 for s in cl
                     if s[f"j_hat::{arm}"] == len(s["steps_by"][split_name]) + 1)
        br = [s for s in scored if not s["is_gold"]]
        br_end = sum(1 for s in br
                     if s[f"j_hat::{arm}"] == len(s["steps_by"][split_name]) + 1)
        print("=" * 78)
        print(f"ARM {arm}   ({len(scored)} chains scored)")
        print("=" * 78)
        print(f"  A) p_clean AUC          {a:.4f}   "
              f"[{len(pos)} gold / {len(neg)} non-gold; stage32: 0.835]")
        print(f"  B) localization exact   {hits}/{len(loc)} "
              f"({100*exact_rate:.1f}%)   chance {100*chance:.1f}%   "
              f"lift {100*(exact_rate-chance):+.1f}pts   "
              f"[stage32: 17.6% vs 21.7% = -4.1pts]")
        print(f"     within one step      {near}/{len(loc)} "
              f"({100*near/max(len(loc),1):.1f}%)")
        print(f"  C) clean chains j=T+1   {at_end}/{len(cl)} "
              f"({100*at_end/max(len(cl),1):.1f}%)")
        print(f"     broken chains j=T+1  {br_end}/{len(br)} "
              f"({100*br_end/max(len(br),1):.1f}%)  (missed the error)")
        results[arm] = {
            "n_scored": len(scored),
            "A_auc": round(a, 4) if a is not None else None,
            "A_n_gold": len(pos), "A_n_rest": len(neg),
            "B_n": len(loc), "B_exact": hits, "B_within_one": near,
            "B_exact_rate": round(exact_rate, 4),
            "B_chance_rate": round(chance, 4),
            "B_lift": round(exact_rate - chance, 4),
            "C_clean_at_end": at_end, "C_n_clean": len(cl),
            "C_broken_at_end": br_end, "C_n_broken": len(br)}

    print()
    print("=" * 78)
    print("PRE-REGISTERED VERDICTS")
    print("=" * 78)
    verdicts = {}
    paper_arms = [a for a in results if a.startswith("paper_")]
    s32_arms = [a for a in results if a.startswith("s32_")]
    if paper_arms:
        best = max(results[a]["B_lift"] for a in paper_arms)
        verdicts["P1_localization_lift"] = {
            "value": round(best, 4), "bar": 0.05,
            "pass": bool(best >= 0.05)}
        print(f"  P1 paper localization lift >= +5pts: "
              f"{100*best:+.1f}pts -> {'PASS' if best >= 0.05 else 'FAIL'}")
    if paper_arms and s32_arms:
        deltas = [abs(results[p]["A_auc"] - results[s]["A_auc"])
                  for p, s in zip(sorted(paper_arms), sorted(s32_arms))
                  if results[p]["A_auc"] is not None
                  and results[s]["A_auc"] is not None]
        if deltas:
            worst = max(deltas)
            verdicts["P2_pclean_stability"] = {
                "value": round(worst, 4), "bar": 0.03,
                "pass": bool(worst < 0.03)}
            print(f"  P2 |paper - stage32| p_clean AUC < 0.03: "
                  f"{worst:.4f} -> {'PASS' if worst < 0.03 else 'FAIL'}")
    fb = [a for a in results if a.endswith("_fallback")]
    nl = [a for a in results if a.endswith("_newline")]
    if fb and nl:
        deltas = [abs(results[f]["A_auc"] - results[n]["A_auc"])
                  for f, n in zip(sorted(fb), sorted(nl))
                  if results[f]["A_auc"] is not None
                  and results[n]["A_auc"] is not None]
        if deltas:
            worst = max(deltas)
            verdicts["P4_split_stability"] = {
                "value": round(worst, 4), "bar": 0.05,
                "pass": bool(worst < 0.05)}
            print(f"  P4 |fallback - newline| p_clean AUC < 0.05: "
                  f"{worst:.4f} -> {'PASS' if worst < 0.05 else 'FAIL'}")
            print("     (a large DROP here means stage32's 0.835 was partly "
                  "a survivorship artifact -- report as such)")
    results["verdicts"] = verdicts
    return results


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- splitters: newline arm reproduces stage32 exactly ---")
    from .toy1_first_broken_step import split_steps as s32_split
    cases = ["a = 1\nb = 2\n#### 2", "a\n\n\nb", "one line only",
             "x\ny\n#### 3", "  \n a \n\n b \n#### 9"]
    for text in cases:
        check(f"newline == stage32 on {text!r}",
              split_steps_newline(text), s32_split(text))

    print("--- fallback: only fires when newline under-segments ---")
    check("multi-line unchanged",
          split_steps_fallback("a = 1\nb = 2\n#### 2"), ["a = 1", "b = 2"])
    one = ("She has 6 * 2 = <<6*2=12>>12 tubes. That serves "
           "12 * 3 = <<12*3=36>>36 people. #### 36")
    got = split_steps_fallback(one)
    check("one-line multi-sentence is split", len(got) >= 2, True)
    check("answer line is not a step",
          any(s.startswith("####") for s in got), False)
    check("newline splitter still fails on it",
          len(split_steps_newline(one)) < MIN_STEPS, True)

    print("--- fallback: genuine one-liners still drop ---")
    for junk in ("**Example 1**:", "Alyssa ate 20 nuggets.", "#### 7"):
        check(f"junk {junk!r} stays under MIN_STEPS",
              len(split_steps_fallback(junk)) < MIN_STEPS, True)

    print("--- fallback: real Arabic one-liner from the data (train_ar_118) ---")
    ar = ("جانيت بدأت بـ 3. "
          "في الأسبوع "
          "الأول، 3 + 2 = 5. "
          "في الأسبوع "
          "الثاني، 5 + 2 = 7. #### 7")
    got_ar = split_steps_fallback(ar)
    check("arabic one-liner splits into >= 3 steps", len(got_ar) >= 3, True)
    check("arabic answer line dropped",
          any("####" in s for s in got_ar), False)

    print("--- fallback: calculator-annotation boundary as last resort ---")
    nosent = "6 * 2 = <<6*2=12>>12 tubes 12 * 3 = <<12*3=36>>36 people"
    check("calc split fires when no sentence breaks",
          len(split_steps_fallback(nosent)) >= 2, True)

    print("--- MAX_STEPS cap holds in both splitters ---")
    many_nl = "\n".join(f"step {i}" for i in range(30))
    many_sent = " ".join(f"Step {i} happens." for i in range(30))
    check("newline capped", len(split_steps_newline(many_nl)), MAX_STEPS)
    check("fallback capped", len(split_steps_fallback(many_sent)), MAX_STEPS)

    print("--- system prompts ---")
    check("stage32 prompt has NO absorbing clause",
          "subsequent" in SYSTEM_S32, False)
    check("paper prompt HAS the absorbing clause",
          "subsequent steps for that problem to also be incorrect"
          in SYSTEM_PAPER, True)
    check("paper prompt states the two-symbol constraint",
          "only be one of these two symbols" in SYSTEM_PAPER, True)

    print("--- transcript / slots respect the chosen system prompt ---")
    steps = ["one", "two", "three"]
    for name, system in SYSTEMS.items():
        t = build_transcript("Q?", steps, 2, system)
        check(f"{name}: transcript carries its system prompt",
              t.startswith(system), True)
        check(f"{name}: marks step 1 correct",
              "Step 1: one\nJudge: +" in t, True)
        check(f"{name}: marks step 2 incorrect",
              "Step 2: two\nJudge: -" in t, True)
        check(f"{name}: stops after the first error", "Step 3" in t, False)
        clean = build_transcript("Q?", steps, 4, system)
        check(f"{name}: j=T+1 is all plus", clean.count("Judge: +"), 3)
        check(f"{name}: j=T+1 has no minus", "Judge: -" in clean, False)
        slots = marker_slots("Q?", steps, 2, system)
        check(f"{name}: slots stop at the error", len(slots), 2)
        check(f"{name}: slot prompt ends at the marker",
              slots[0][0].endswith("Step 1: one\nJudge:"), True)

    print("--- all-plus-prefix reuse is still EXACT under both prompts ---")
    # This is the property audit_eq6.py proved for stage32; it must survive
    # the system-prompt swap, since the prompt only changes the prefix head.
    for name, system in SYSTEMS.items():
        allplus = []
        prefix = system + "Problem: " + "Q?"
        for t, step in enumerate(steps, start=1):
            allplus.append(prefix + f"\n\nStep {t}: {step}\nJudge:")
            prefix = prefix + f"\n\nStep {t}: {step}\nJudge: +"
        ok = True
        for j in range(1, len(steps) + 2):
            for i, (prompt, _) in enumerate(marker_slots("Q?", steps, j, system)):
                ok &= (prompt == allplus[i])
        check(f"{name}: every hypothesis reuses the all-plus prompts", ok, True)

    print("--- Eq. 6 imported unchanged (audit_eq6 anchor values) ---")
    lp = [(-0.1, -2.0), (-0.2, -1.5), (-0.3, -1.0)]
    check("j=1 is the minus at step 1",
          round(score_from_logprobs(lp, 1, 3), 4), -2.0)
    check("j=2 is plus1 + minus2",
          round(score_from_logprobs(lp, 2, 3), 4), round(-0.1 - 1.5, 4))
    check("j=T+1 is all pluses",
          round(score_from_logprobs(lp, 4, 3), 4), round(-0.6, 4))

    print("--- arithmetic checker is the FIXED one (precedence) ---")
    # 2+3*4 IS 14 under precedence. The OLD checker folded left-to-right,
    # got 20, and falsely flagged this -- 66.7% of its flags were such
    # false alarms (fix_impact.log). The fixed checker must NOT flag it.
    check("true precedence chain is not flagged",
          first_bad_step(["Total is 2 + 3 * 4 = <<2+3*4=14>>14."]), None)
    check("the old left-to-right answer IS flagged",
          first_bad_step(["Total is 2 + 3 * 4 = <<2+3*4=20>>20."]), 1)
    check("80-8*5=40 (the af_1011 false alarm) is not flagged",
          first_bad_step(["Sy het 80 - 8 * 5 = <<80-8*5=40>>40 nodig."]), None)
    check("a genuinely broken step is still located",
          first_bad_step(["Clean: 6 * 2 = <<6*2=12>>12.",
                          "Broken: 12 * 3 = <<12*3=30>>30."]), 2)

    print("--- sample construction is arm-identical ---")
    fake = {
        "train_en_1": {"lang": "en", "gold": "12", "question": "q",
                       "chains": [
                           {"answer": "12", "text": "a = 1\nb = 2\n#### 12"},
                           {"answer": "13", "text": "6*2 = <<6*2=13>>13.\n"
                                                    "Then done.\n#### 13"}]},
        "train_am_1": {"lang": "am", "gold": "5", "question": "q",
                       "chains": [
                           {"answer": "5", "text": "One thing. Two thing. "
                                                   "#### 5"}]},
    }
    rng = random.Random(0)
    sample = build_sample(fake, rng, SPLITTERS)
    check("every row carries both splitters' steps",
          all(set(r["steps_by"]) == set(SPLITTERS) for r in sample), True)
    check("every row meets MIN_STEPS under both",
          all(len(s) >= MIN_STEPS
              for r in sample for s in r["steps_by"].values()), True)
    check("gold never enters a prompt",
          all("gold" not in r for r in sample), True)

    print()
    if failures:
        print(f"{len(failures)} FAILURES:")
        for f in failures:
            print(f"  {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--coverage-only", action="store_true",
                        help="P3 step-partition coverage; CPU only, login-safe")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    pools = load_pools(args.root)

    if args.coverage_only:
        cov = coverage_report(pools)
        (OUT_DIR / f"stage42_coverage{args.tag}.json").write_text(
            json.dumps(cov, indent=2))
        print(f"\nwrote stage42_coverage{args.tag}.json")
        return

    rng = random.Random(SEED)
    sample = build_sample(pools, rng, SPLITTERS)
    arms = [f"{s}_{p}" for s in ("s32", "paper") for p in SPLITTERS]
    n_gold = sum(s["is_gold"] for s in sample)
    langs = Counter(pools[s["iid"]]["lang"] for s in sample)
    print(f"pools {len(pools)}; sample {len(sample)} chains "
          f"({n_gold} gold / {len(sample)-n_gold} non-gold)")
    print(f"languages: {dict(sorted(langs.items()))}")
    for name in SPLITTERS:
        n_bad = sum(1 for s in sample
                    if not s["is_gold"] and s["bad_step_by"][name])
        med = sorted(len(s["steps_by"][name]) for s in sample)[len(sample)//2]
        print(f"  [{name}] non-gold with a verifiable slip: {n_bad}; "
              f"median steps {med}")
    print(f"arms: {arms}")
    print("CAVEAT: Qwen3-14B-Base is a base model; uPRM used an instruct "
          "grader in chat format. A null here is partly about base models.")
    coverage_report(pools)

    if args.dry_run:
        s = next(x for x in sample
                 if not x["is_gold"] and x["bad_step_by"]["newline"])
        print(f"\n--- broken chain, first bad step "
              f"(newline) = {s['bad_step_by']['newline']} ---")
        for i, step in enumerate(s["steps_by"]["newline"], 1):
            print(f"  [{i}] {step[:100]}")
        for arm in arms:
            system = SYSTEMS[arm.split("_", 1)[0].replace("s32", "stage32")]
            split_name = arm.split("_", 1)[1]
            print(f"\n--- {arm}: transcript asserting j=1 (first 700) ---")
            print(build_transcript(s["question"], s["steps_by"][split_name],
                                   1, system)[:700])
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    plus_id = tokenizer.encode(" +", add_special_tokens=False)[0]
    minus_id = tokenizer.encode(" -", add_special_tokens=False)[0]
    print(f"marker ids: '+' {plus_id}, '-' {minus_id}", flush=True)
    print("loading model in bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    import inspect
    last_logit = ({"logits_to_keep": 1}
                  if "logits_to_keep" in inspect.signature(model.forward).parameters
                  else {})

    for arm in arms:
        system = SYSTEMS["paper" if arm.startswith("paper") else "stage32"]
        split_name = arm.split("_", 1)[1]
        entries = []
        for index, s in enumerate(sample):
            steps = s["steps_by"][split_name]
            prefix = system + "Problem: " + s["question"].strip()
            for t, step in enumerate(steps, start=1):
                prompt = prefix + f"\n\nStep {t}: {step}\nJudge:"
                n_tokens = len(tokenizer.encode(prompt))
                if n_tokens <= MAX_PROMPT_TOKENS:
                    entries.append((n_tokens, index, t, prompt))
                prefix = prefix + f"\n\nStep {t}: {step}\nJudge: +"
        entries.sort(key=lambda e: e[0])
        print(f"\n[{arm}] scoring {len(entries)} marker slots", flush=True)

        per_chain = defaultdict(dict)
        for start in range(0, len(entries), BATCH_SIZE):
            batch = entries[start:start + BATCH_SIZE]
            enc = tokenizer([p for _, _, _, p in batch], return_tensors="pt",
                            padding=True, pad_to_multiple_of=256).to(model.device)
            with torch.inference_mode():
                logits = model(**enc, use_cache=False, **last_logit).logits
                tail = logits[:, -1, :].float()
                pair = torch.stack([tail[:, plus_id], tail[:, minus_id]], dim=-1)
                logprobs = torch.log_softmax(pair, dim=-1)
            for row, (_, index, t, _) in enumerate(batch):
                per_chain[index][t] = (float(logprobs[row, 0]),
                                       float(logprobs[row, 1]))
            if start % (BATCH_SIZE * 50) == 0:
                print(f"  {start}/{len(entries)}", flush=True)
                torch.cuda.empty_cache()

        for index, s in enumerate(sample):
            slots = per_chain.get(index, {})
            n_steps = len(s["steps_by"][split_name])
            if len(slots) < n_steps:
                continue
            step_logprobs = [slots[t] for t in range(1, n_steps + 1)]
            scores = {j: score_from_logprobs(step_logprobs, j, n_steps)
                      for j in range(1, n_steps + 2)}
            s[f"j_hat::{arm}"] = max(scores, key=lambda j: (scores[j], -j))
            top = max(scores.values())
            lse = top + math.log(sum(math.exp(v - top)
                                     for v in scores.values()))
            s[f"p_clean::{arm}"] = scores[n_steps + 1] - lse

    results = report(sample, arms)
    (OUT_DIR / f"stage42_results{args.tag}.json").write_text(
        json.dumps(results, indent=2))
    with (OUT_DIR / f"stage42_chains{args.tag}.jsonl").open(
            "w", encoding="utf-8") as h:
        for s in sample:
            row = {"iid": s["iid"], "is_gold": s["is_gold"],
                   "n_steps": {k: len(v) for k, v in s["steps_by"].items()},
                   "bad_step": s["bad_step_by"]}
            for arm in arms:
                row[f"j_hat::{arm}"] = s.get(f"j_hat::{arm}")
                row[f"p_clean::{arm}"] = s.get(f"p_clean::{arm}")
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote stage42_results{args.tag}.json and "
          f"stage42_chains{args.tag}.jsonl")


if __name__ == "__main__":
    main()
