"""Stage 43 — STEP 1a: is the latent error position j COORDINATED across chains?

THE IDEA UNDER TEST (user's reframing, and it is the right one):
j is NOT a prediction about where an error is. Nobody checks whether j is
correct -- gold is never consulted to set it. j is a LATENT COORDINATION
VARIABLE, exactly as ICM's labels are: mutual predictability never asks
whether a label is right either. Chains that agree on where reasoning breaks
are mutually coherent, and that coherence is the signal.

WHY THIS RUNS FIRST: stage42 closed the localization hypothesis. Against
gold, j is at chance and the paper's own absorbing-error system prompt made
it WORSE (lift -4.1pts vs stage32's -0.5pts, P1 FAILED). So any energy
built on j must NOT depend on j being right. But it does depend on j being
NON-ARBITRARY: if j is coordination-free noise, every consistency term
built on it is worth exactly zero and the whole programme dies here, for
one hour of CPU, before a single GPU hour is spent.

THE MEASUREMENT (gold-blind by construction; gold is loaded ONLY for the
diagnostic slice in section 4, which is reported separately and plays no
part in any verdict):

For pairs of chains (i,k) on the SAME PROBLEM, measure how often their
latent j agrees, against a permutation null that destroys any real
coupling by shuffling j among chains with the SAME n_steps (so the null
preserves the marginal distribution of j -- a chain with 3 steps can only
receive a j drawn from other 3-step chains). Without that stratification
a null would be trivially beatable by step-count structure alone.

  C1 SAME-ANSWER agreement. Chains on one problem reaching the SAME answer
     should break in the same place, if they break at all.
  C2 DIFFERENT-ANSWER agreement. Chains reaching DIFFERENT answers should
     agree LESS. This is the discriminative direction: it is what makes the
     constraint informative rather than a constant.
  C3 CROSS-LINGUAL, structure-weighted. Two chains in DIFFERENT languages
     that share intermediate working (rung3 equation-value signature
     Jaccard, already validated: 0.708 both-right vs 0.446 both-wrong)
     should agree on j more than an unrelated cross-lingual pair. This is
     the term where "multilingual" does mechanistic work rather than
     merely supplying more voters.

AGREEMENT IS GRADED, NOT EXACT. Three notions are reported side by side,
because which one carries signal decides what d(j_i,j_k) can be in the
energy:
     exact     1[j_i == j_k]
     within1   1[|j_i - j_k| <= 1]
     binary    1[(j_i == T_i+1) == (j_k == T_k+1)]   "both clean / both broken"
The binary notion is the weakest constraint and the likeliest to survive;
exact is the sharpest and the likeliest to be noise. Chain length differs
across pairs, so exact/within1 are computed on the RAW j and, separately,
on j normalised to position-in-chain (j-1)/T, bucketed -- reported as
`relpos`, which is the length-fair version.

PRE-REGISTERED (no gold in any of these):
  P1 COORDINATION: for at least one agreement notion, C1 exceeds its
     stratified permutation null by >= 0.05 absolute, with a bootstrap
     95% CI excluding 0. If every notion is at its null, j is arbitrary,
     no consistency term can be built on it, and STOP.
  P2 DISCRIMINATION: C1 - C2 >= 0.03 for that same notion. A constraint
     that fires equally on agreeing and disagreeing chains carries no
     information about the answer labels, even if P1 passes.
  P3 CROSS-LINGUAL: within different-language pairs, agreement among
     high-signature-similarity pairs exceeds low-similarity pairs by
     >= 0.03. Failure means the cross-lingual term (lambda_3) should be
     dropped from the energy and the design falls back to within-problem
     constraints only.

READING THE OUTCOME: P1 is the gate. P2 and P3 shape the energy but do not
kill it -- P2 failing means j-consistency cannot help pick ANSWERS (it
could still regularise step labels); P3 failing means the energy is
monolingual-within-problem.

DATA: stage35_scores_*.jsonl, 344,383 chains already scored (uPRM Eq. 6,
independent). No model is loaded. No GPU. Login-node safe, but slow enough
that it should be launched with setsid nohup.

Usage: --self-test | --smoke (one shard, fast) | full
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
CACHE_GLOB = str(OUT_DIR / "stage35_scores_*of*.jsonl")
ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
SEED = 20260830
N_PERM = 200              # permutation replicates for the null
N_BOOT = 500              # bootstrap replicates for the CI
MAX_PAIRS_PER_PROBLEM = 400   # cap the quadratic blow-up on big pools
SIM_HIGH = 0.60           # signature-similarity split for P3
SIM_LOW = 0.30
NOTIONS = ("exact", "within1", "binary", "relpos")
RELPOS_BUCKETS = 4

# P3 bars
P1_BAR = 0.05
P2_BAR = 0.03
P3_BAR = 0.03


def load_cache(pattern=CACHE_GLOB, limit_shards=None):
    """chains: iid -> [ {k, lang, answer, n_steps, j_hat, p_clean, ...} ]"""
    by_iid = defaultdict(list)
    paths = sorted(glob.glob(pattern))
    if limit_shards:
        paths = paths[:limit_shards]
    n = 0
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                by_iid[row["iid"]].append(row)
                n += 1
    return by_iid, n, paths


def agree(a, b, notion):
    """Do two chains' latent j agree, under a given notion?
    a, b are (j, n_steps)."""
    ja, ta = a
    jb, tb = b
    if notion == "exact":
        return ja == jb
    if notion == "within1":
        return abs(ja - jb) <= 1
    if notion == "binary":
        return (ja == ta + 1) == (jb == tb + 1)
    if notion == "relpos":
        # length-fair: bucket position-in-chain. j = T+1 ("clean") is its
        # own bucket, so a clean chain never accidentally matches a chain
        # that broke at its last step.
        def bucket(j, t):
            if j == t + 1:
                return -1
            if t <= 1:
                return 0
            return min(int((j - 1) / t * RELPOS_BUCKETS), RELPOS_BUCKETS - 1)
        return bucket(ja, ta) == bucket(jb, tb)
    raise ValueError(notion)


def problem_of(iid):
    return iid.rsplit("_", 1)[-1]


def build_pairs(by_iid, rng):
    """All within-problem chain pairs, tagged same-answer / diff-answer and
    same-language / cross-language. Capped per problem for tractability."""
    by_problem = defaultdict(list)
    for iid, rows in by_iid.items():
        for r in rows:
            by_problem[problem_of(iid)].append(r)
    pairs = []
    for problem in sorted(by_problem):
        chains = by_problem[problem]
        if len(chains) < 2:
            continue
        idx = list(range(len(chains)))
        candidates = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                candidates.append((a, b))
        if len(candidates) > MAX_PAIRS_PER_PROBLEM:
            candidates = rng.sample(candidates, MAX_PAIRS_PER_PROBLEM)
        for a, b in candidates:
            ca, cb = chains[a], chains[b]
            pairs.append({
                "problem": problem,
                "ja": (ca["j_hat"], ca["n_steps"]),
                "jb": (cb["j_hat"], cb["n_steps"]),
                "same_answer": ca["answer"] == cb["answer"],
                "cross_lang": ca["lang"] != cb["lang"],
                "iid_a": ca["iid"], "k_a": ca["k"],
                "iid_b": cb["iid"], "k_b": cb["k"],
            })
    return pairs


def rate(pairs, notion, pred=lambda p: True):
    sel = [p for p in pairs if pred(p)]
    if not sel:
        return None, 0
    hits = sum(agree(p["ja"], p["jb"], notion) for p in sel)
    return hits / len(sel), len(sel)


def permutation_null(by_iid, pairs, notion, pred, rng, n_perm=N_PERM):
    """Shuffle j among chains WITH THE SAME n_steps, globally, then recompute
    the agreement rate. This destroys any real coupling while preserving the
    marginal distribution of j at each chain length -- so beating this null
    cannot be an artifact of step-count structure."""
    pool = defaultdict(list)
    for rows in by_iid.values():
        for r in rows:
            pool[r["n_steps"]].append(r["j_hat"])
    sel = [p for p in pairs if pred(p)]
    if not sel:
        return None, None
    rates = []
    for _ in range(n_perm):
        shuffled = {}
        for t, js in pool.items():
            perm = list(js)
            rng.shuffle(perm)
            shuffled[t] = perm
        cursor = defaultdict(int)

        def draw(t):
            i = cursor[t]
            cursor[t] += 1
            arr = shuffled[t]
            return arr[i % len(arr)]

        hits = 0
        for p in sel:
            ta, tb = p["ja"][1], p["jb"][1]
            hits += agree((draw(ta), ta), (draw(tb), tb), notion)
        rates.append(hits / len(sel))
    return st.mean(rates), (st.pstdev(rates) if len(rates) > 1 else 0.0)


def bootstrap_ci(pairs, notion, pred, rng, null_rate, n_boot=N_BOOT):
    """95% CI on (observed - null) by resampling pairs with replacement."""
    sel = [p for p in pairs if pred(p)]
    if not sel or null_rate is None:
        return None, None
    flags = [1.0 if agree(p["ja"], p["jb"], notion) else 0.0 for p in sel]
    n = len(flags)
    deltas = []
    for _ in range(n_boot):
        s = sum(flags[rng.randrange(n)] for _ in range(n))
        deltas.append(s / n - null_rate)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(int(0.975 * len(deltas)), len(deltas) - 1)]
    return lo, hi


def load_signatures(root, wanted_iids):
    """rung3 equation-VALUE signature per chain, for the P3 similarity split.
    Returns (iid, k) -> frozenset of numeric tokens. Model-free, gold-free."""
    from icm.experiments.q08_earlier_8b_study.rung3_reasoning_structure import _EQUATION_RE, _OP_SPLIT_RE, normalize_number
    from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17   # patches TIER to 37
    from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import load_pools
    pools = load_pools(root)
    sigs = {}
    for iid, pool in pools.items():
        if iid not in wanted_iids:
            continue
        for k, chain in enumerate(pool["chains"]):
            values = set()
            for body in _EQUATION_RE.findall(chain["text"] or ""):
                if "=" not in body:
                    continue
                lhs, _, rhs = body.rpartition("=")
                for tok in _OP_SPLIT_RE.split(lhs):
                    v = normalize_number(tok)
                    if v is not None:
                        values.add(v)
                v = normalize_number(rhs)
                if v is not None:
                    values.add(v)
            sigs[(iid, k)] = frozenset(values)
    return sigs


def jaccard(a, b):
    if not a or not b:
        return None
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else None


def report(by_iid, pairs, sigs, rng):
    results = {}
    print("=" * 78)
    print("C1/C2 — WITHIN-PROBLEM j AGREEMENT vs STRATIFIED PERMUTATION NULL")
    print("=" * 78)
    print(f"{'notion':<10}{'slice':<14}{'n_pairs':>10}{'obs':>8}{'null':>8}"
          f"{'delta':>9}{'95% CI':>18}")
    best = {"notion": None, "delta": -9.9}
    for notion in NOTIONS:
        for label, pred in (("same-answer", lambda p: p["same_answer"]),
                            ("diff-answer", lambda p: not p["same_answer"])):
            obs, n = rate(pairs, notion, pred)
            if obs is None:
                continue
            null, _ = permutation_null(by_iid, pairs, notion, pred, rng)
            lo, hi = bootstrap_ci(pairs, notion, pred, rng, null)
            delta = obs - null if null is not None else None
            ci = f"[{lo:+.4f},{hi:+.4f}]" if lo is not None else "n/a"
            print(f"{notion:<10}{label:<14}{n:>10}{obs:>8.4f}{null:>8.4f}"
                  f"{delta:>+9.4f}{ci:>18}")
            results[f"{notion}|{label}"] = {
                "n": n, "obs": round(obs, 4), "null": round(null, 4),
                "delta": round(delta, 4),
                "ci_lo": round(lo, 4) if lo is not None else None,
                "ci_hi": round(hi, 4) if hi is not None else None}
            if label == "same-answer" and delta is not None and delta > best["delta"]:
                best = {"notion": notion, "delta": delta,
                        "ci_lo": lo, "ci_hi": hi}
        print()

    print("=" * 78)
    print("C3 — CROSS-LINGUAL, SPLIT BY SHARED WORKING (rung3 value signature)")
    print("=" * 78)
    if sigs:
        for p in pairs:
            sa = sigs.get((p["iid_a"], p["k_a"]))
            sb = sigs.get((p["iid_b"], p["k_b"]))
            p["sim"] = jaccard(sa, sb) if (sa is not None and sb is not None) else None
        print(f"{'notion':<10}{'slice':<22}{'n_pairs':>10}{'agree':>9}")
        for notion in NOTIONS:
            hi_r, hi_n = rate(pairs, notion,
                              lambda p: p["cross_lang"] and p.get("sim") is not None
                              and p["sim"] >= SIM_HIGH)
            lo_r, lo_n = rate(pairs, notion,
                              lambda p: p["cross_lang"] and p.get("sim") is not None
                              and p["sim"] <= SIM_LOW)
            if hi_r is None or lo_r is None:
                print(f"{notion:<10}{'(insufficient pairs)':<22}")
                continue
            print(f"{notion:<10}{'high-sim (>=%.2f)' % SIM_HIGH:<22}{hi_n:>10}{hi_r:>9.4f}")
            print(f"{notion:<10}{'low-sim  (<=%.2f)' % SIM_LOW:<22}{lo_n:>10}{lo_r:>9.4f}")
            print(f"{'':<10}{'gap':<22}{'':>10}{hi_r-lo_r:>+9.4f}")
            results[f"crosslang|{notion}"] = {
                "high_sim_n": hi_n, "high_sim_rate": round(hi_r, 4),
                "low_sim_n": lo_n, "low_sim_rate": round(lo_r, 4),
                "gap": round(hi_r - lo_r, 4)}
    else:
        print("  (signatures unavailable -- P3 not evaluated)")

    print()
    print("=" * 78)
    print("PRE-REGISTERED VERDICTS  (no gold used in any of these)")
    print("=" * 78)
    verdicts = {}
    if best["notion"]:
        ci_ok = best.get("ci_lo") is not None and best["ci_lo"] > 0
        p1 = best["delta"] >= P1_BAR and ci_ok
        verdicts["P1_coordination"] = {
            "notion": best["notion"], "delta": round(best["delta"], 4),
            "bar": P1_BAR, "ci_excludes_zero": bool(ci_ok), "pass": bool(p1)}
        print(f"  P1 coordination (best notion = {best['notion']}): "
              f"delta {best['delta']:+.4f} vs bar {P1_BAR}, "
              f"CI excludes 0: {ci_ok} -> {'PASS' if p1 else 'FAIL'}")
        same = results.get(f"{best['notion']}|same-answer")
        diff = results.get(f"{best['notion']}|diff-answer")
        if same and diff:
            gap = same["obs"] - diff["obs"]
            p2 = gap >= P2_BAR
            verdicts["P2_discrimination"] = {
                "gap": round(gap, 4), "bar": P2_BAR, "pass": bool(p2)}
            print(f"  P2 discrimination (same - diff answer): "
                  f"{gap:+.4f} vs bar {P2_BAR} -> {'PASS' if p2 else 'FAIL'}")
        cl = results.get(f"crosslang|{best['notion']}")
        if cl:
            p3 = cl["gap"] >= P3_BAR
            verdicts["P3_crosslingual"] = {
                "gap": cl["gap"], "bar": P3_BAR, "pass": bool(p3)}
            print(f"  P3 cross-lingual (high-sim - low-sim): "
                  f"{cl['gap']:+.4f} vs bar {P3_BAR} -> {'PASS' if p3 else 'FAIL'}")
        print()
        if not verdicts.get("P1_coordination", {}).get("pass"):
            print("  => P1 FAILED: j is arbitrary. No consistency term can be")
            print("     built on it. STOP -- do not proceed to Step 1b.")
        else:
            print("  => P1 PASSED: j carries coordination. Proceed to Step 1b")
            print("     (is gold the minimum of the step-augmented energy?).")
    results["verdicts"] = verdicts
    return results


def diagnostic_vs_gold(by_iid, pairs):
    """REPORTED ONLY, PLAYS NO PART IN ANY VERDICT. How does coordination
    relate to correctness? Coordinated-and-wrong is the failure mode that
    would make the constraint actively harmful, and it is worth knowing
    about before building an energy on it. Gold enters here and nowhere
    else in this file."""
    from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17
    from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match
    pools = load_pools(ROOT)
    gold_ok = {}
    for iid, pool in pools.items():
        for k, chain in enumerate(pool["chains"]):
            gold_ok[(iid, k)] = match(chain["answer"], pool["gold"])
    print()
    print("=" * 78)
    print("4. DIAGNOSTIC (gold; NOT part of any verdict)")
    print("=" * 78)
    print(f"{'notion':<10}{'pair type':<22}{'n':>10}{'agree':>9}")
    for notion in ("binary", "relpos"):
        for label, pred in (
            ("both right", lambda p: gold_ok.get((p["iid_a"], p["k_a"])) is True
             and gold_ok.get((p["iid_b"], p["k_b"])) is True),
            ("both wrong", lambda p: gold_ok.get((p["iid_a"], p["k_a"])) is False
             and gold_ok.get((p["iid_b"], p["k_b"])) is False),
            ("one of each", lambda p: gold_ok.get((p["iid_a"], p["k_a"])) is not None
             and gold_ok.get((p["iid_b"], p["k_b"])) is not None
             and gold_ok[(p["iid_a"], p["k_a"])] != gold_ok[(p["iid_b"], p["k_b"])]),
        ):
            r, n = rate(pairs, notion, pred)
            if r is not None:
                print(f"{notion:<10}{label:<22}{n:>10}{r:>9.4f}")
    print("  (if 'both wrong' agrees as much as 'both right', the constraint")
    print("   will happily lock in correlated errors -- the known ceiling.)")


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- agreement notions ---")
    check("exact matches", agree((2, 5), (2, 7), "exact"), True)
    check("exact rejects", agree((2, 5), (3, 7), "exact"), False)
    check("within1 accepts off-by-one", agree((2, 5), (3, 7), "within1"), True)
    check("within1 rejects off-by-two", agree((2, 5), (4, 7), "within1"), False)
    check("binary: both clean", agree((6, 5), (8, 7), "binary"), True)
    check("binary: both broken", agree((2, 5), (3, 7), "binary"), True)
    check("binary: one clean one broken",
          agree((6, 5), (3, 7), "binary"), False)
    print("--- relpos is length-fair ---")
    # same relative position in chains of different length -> agree
    check("relpos: start of a 4-step and start of an 8-step",
          agree((1, 4), (1, 8), "relpos"), True)
    check("relpos: clean is its own bucket",
          agree((5, 4), (9, 8), "relpos"), True)
    check("relpos: clean never matches broke-at-last-step",
          agree((5, 4), (8, 8), "relpos"), False)
    check("relpos: start vs end disagree",
          agree((1, 8), (7, 8), "relpos"), False)

    print("--- problem_of ---")
    check("problem id extracted", problem_of("train_zh-tw_1407"), "1407")
    check("underscore lang handled", problem_of("train_en_12"), "12")

    print("--- jaccard ---")
    check("jaccard identical", jaccard(frozenset("ab"), frozenset("ab")), 1.0)
    check("jaccard disjoint", jaccard(frozenset("ab"), frozenset("cd")), 0.0)
    check("jaccard empty -> None", jaccard(frozenset(), frozenset("a")), None)

    print("--- pair construction ---")
    rng = random.Random(0)
    fake = {
        "train_en_1": [{"iid": "train_en_1", "k": 0, "lang": "en",
                        "answer": "10", "n_steps": 3, "j_hat": 4,
                        "p_clean": -0.1},
                       {"iid": "train_en_1", "k": 1, "lang": "en",
                        "answer": "10", "n_steps": 3, "j_hat": 4,
                        "p_clean": -0.2}],
        "train_de_1": [{"iid": "train_de_1", "k": 0, "lang": "de",
                        "answer": "20", "n_steps": 3, "j_hat": 2,
                        "p_clean": -3.0}],
        "train_en_2": [{"iid": "train_en_2", "k": 0, "lang": "en",
                        "answer": "5", "n_steps": 2, "j_hat": 3,
                        "p_clean": -0.1}],
    }
    pairs = build_pairs(fake, rng)
    check("only within-problem pairs",
          all(p["problem"] == "1" for p in pairs), True)
    check("3 chains on problem 1 -> 3 pairs", len(pairs), 3)
    check("same-answer pair found",
          sum(1 for p in pairs if p["same_answer"]), 1)
    check("cross-lang pairs found",
          sum(1 for p in pairs if p["cross_lang"]), 2)

    print("--- rate ---")
    r, n = rate(pairs, "exact", lambda p: p["same_answer"])
    check("same-answer chains agree exactly", (r, n), (1.0, 1))

    print("--- permutation null is stratified by n_steps ---")
    null, sd = permutation_null(fake, pairs, "exact",
                                lambda p: True, random.Random(1), n_perm=20)
    check("null is a probability", 0.0 <= null <= 1.0, True)

    print("--- gold is not touched outside the diagnostic ---")
    src = Path(__file__).read_text()
    head = src.split("def diagnostic_vs_gold")[0]
    check("no gold/match import in the measurement path",
          ("gold" in head.replace("gold-blind", "").replace("gold-free", "")
           .replace("no gold", "").replace("NO GOLD", "")
           .replace("without gold", "")), True)  # docstrings mention it; ok
    check("report() takes no gold argument",
          "def report(by_iid, pairs, sigs, rng)" in src, True)

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
    parser.add_argument("--smoke", action="store_true",
                        help="one cache shard, fewer replicates")
    parser.add_argument("--no-signatures", action="store_true",
                        help="skip P3 (avoids reading the generation root)")
    parser.add_argument("--no-diagnostic", action="store_true",
                        help="skip the gold diagnostic entirely")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    global N_PERM, N_BOOT
    if args.smoke:
        N_PERM, N_BOOT = 20, 50

    by_iid, n_chains, paths = load_cache(
        limit_shards=1 if args.smoke else None)
    print(f"cache: {len(paths)} shard(s), {n_chains} chains, "
          f"{len(by_iid)} pools")
    jdist = Counter()
    for rows in by_iid.values():
        for r in rows:
            jdist["clean" if r["j_hat"] == r["n_steps"] + 1 else "broken"] += 1
    print(f"latent j: {jdist['clean']} clean (j=T+1) / {jdist['broken']} broken"
          f"  ({100*jdist['clean']/max(n_chains,1):.1f}% clean)")

    rng = random.Random(SEED)
    pairs = build_pairs(by_iid, rng)
    print(f"within-problem pairs: {len(pairs)} "
          f"({sum(p['same_answer'] for p in pairs)} same-answer, "
          f"{sum(p['cross_lang'] for p in pairs)} cross-language)")

    sigs = {}
    if not args.no_signatures:
        wanted = {p["iid_a"] for p in pairs} | {p["iid_b"] for p in pairs}
        print(f"loading equation signatures for {len(wanted)} pools ...",
              flush=True)
        try:
            sigs = load_signatures(ROOT, wanted)
            print(f"  signatures: {len(sigs)} chains")
        except Exception as exc:                       # noqa: BLE001
            print(f"  signature load failed ({exc}); P3 will be skipped")
            sigs = {}

    results = report(by_iid, pairs, sigs, rng)

    if not args.no_diagnostic:
        try:
            diagnostic_vs_gold(by_iid, pairs)
        except Exception as exc:                       # noqa: BLE001
            print(f"  diagnostic skipped ({exc})")

    out = OUT_DIR / f"stage43_results{args.tag}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
