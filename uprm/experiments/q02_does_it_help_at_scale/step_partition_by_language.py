"""Does newline-splitting actually partition steps in EVERY language, or
only the ones that happen to imitate the English few-shot format?

split_steps() treats one line as one step. That is an assumption, not a
measurement. If some languages emit a single unbroken paragraph, then for
those languages the PRM saw one giant 'step' and Eq. 6 degenerates to a
single yes/no judgement -- i.e. the whole step-level method silently
reverted to the verdict judge there.
"""
import re
import statistics as st
from collections import defaultdict

from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17
from icm.experiments.q02_can_the_model_judge_itself import verdict_judge_gate as gate
from icm.experiments.q04_full_scale_headline.full_scale_37x1500 import load
from ..q01_does_step_scoring_work.toy1_first_broken_step import CHAIN_MAX_CHARS, MIN_STEPS, split_steps

CALC = re.compile(r"<<[^>]*>>")
DIGIT_OP = re.compile(r"\d\s*[-+*/×÷=]\s*\d")

pools = load("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500")
stats = defaultdict(lambda: {"n": 0, "one_line": 0, "lines": [], "chars": [],
                             "calc_lines": 0, "tot_lines": 0, "too_short": 0})
for iid, pool in pools.items():
    s = stats[pool["lang"]]
    for c in pool["chains"]:
        if len(c["text"]) > CHAIN_MAX_CHARS:
            continue
        steps = split_steps(c["text"])
        s["n"] += 1
        if len(steps) <= 1:
            s["one_line"] += 1
        if len(steps) < MIN_STEPS:
            s["too_short"] += 1
        s["lines"].append(len(steps))
        for line in steps:
            s["tot_lines"] += 1
            s["chars"].append(len(line))
            if CALC.search(line) or DIGIT_OP.search(line):
                s["calc_lines"] += 1

print(f"{'lang':<7}{'tier':<6}{'chains':>8}{'1-line%':>9}{'<2 steps%':>11}"
      f"{'mean steps':>11}{'chars/step':>12}{'steps w/math%':>15}")
rows = []
for lang in sorted(stats, key=lambda l: (gate.TIER.get(l, "zzz"), l)):
    s = stats[lang]
    if not s["n"]:
        continue
    rows.append((lang, gate.TIER.get(lang, "?"), s["n"],
                 100 * s["one_line"] / s["n"],
                 100 * s["too_short"] / s["n"],
                 st.mean(s["lines"]),
                 st.mean(s["chars"]) if s["chars"] else 0,
                 100 * s["calc_lines"] / max(s["tot_lines"], 1)))
for r in rows:
    print(f"{r[0]:<7}{r[1]:<6}{r[2]:>8}{r[3]:>8.1f}%{r[4]:>10.1f}%"
          f"{r[5]:>11.2f}{r[6]:>12.0f}{r[7]:>14.1f}%")

worst = sorted(rows, key=lambda r: -r[3])[:6]
print("\nHIGHEST single-line rates (worst partitioning):")
for r in worst:
    print(f"  {r[0]:<6} {r[3]:.1f}% of chains are ONE line "
          f"(mean {r[5]:.2f} steps, {r[6]:.0f} chars/step)")
best = sorted(rows, key=lambda r: r[3])[:4]
print("\nLOWEST single-line rates (cleanest partitioning):")
for r in best:
    print(f"  {r[0]:<6} {r[3]:.1f}% one-line (mean {r[5]:.2f} steps)")
print("DONE")
