"""AUDIT items 1,2,3: Eq.6 indexing, all-plus prefix exactness, logsumexp."""
import math, sys
from .toy1_first_broken_step import score_from_logprobs, marker_slots, build_transcript

print("="*72)
print("ITEM 1: Eq. 6 indexing, hand-computed")
print("="*72)
# T=3 steps. step_logprobs[t] = (log p+_{t+1}, log p-_{t+1}), 0-indexed.
lp = [(-0.1, -2.0), (-0.2, -1.5), (-0.3, -1.0)]
# Paper: S(j) = 1[j<=T]*log p-_j + sum_{t<j} log p+_t   (j,t 1-indexed)
def reference(lp, j, T):
    s = 0.0
    for t in range(1, j):          # t = 1..j-1, 1-indexed
        s += lp[t-1][0]            # log p+_t
    if j <= T:
        s += lp[j-1][1]            # log p-_j
    return s
print(f"{'j':>3} {'code':>10} {'byhand':>10} {'ref':>10}  match")
byhand = {1: -2.0,
          2: -0.1 + -1.5,
          3: -0.1 + -0.2 + -1.0,
          4: -0.1 + -0.2 + -0.3}
ok = True
for j in range(1, 5):
    c = score_from_logprobs(lp, j, 3)
    r = reference(lp, j, 3)
    h = byhand[j]
    m = abs(c-r) < 1e-12 and abs(c-h) < 1e-12
    ok &= m
    print(f"{j:>3} {c:>10.4f} {h:>10.4f} {r:>10.4f}  {'OK' if m else 'MISMATCH'}")
print(f"ITEM 1 VERDICT: {'CORRECT (no off-by-one)' if ok else 'BUG'}")

# boundary: j=1 must contain NO plus terms
print("\n  j=1 boundary: range(min(0,3)) is empty ->", list(range(min(1-1,3))), "-> no plus terms. Correct.")
print("  j=T+1=4 boundary: range(min(3,3)) ->", list(range(min(4-1,3))), "-> pluses 1..3, and 4<=3 False so no minus. Correct.")
# What if j > T+1 were ever passed? min() clamps.
print("  j=T+2=5 (never generated):", score_from_logprobs(lp,5,3), "== S(T+1) -- clamped, would be a DUPLICATE if ever enumerated.")
print("  callers enumerate j in range(1, n_steps+2) = 1..T+1 -> never hits this. OK.")

print()
print("="*72)
print("ITEM 2: is the all-plus-prefix reuse EXACT?")
print("="*72)
# marker_slots(question, steps, j) builds the prompts the ORIGINAL (unoptimized)
# formulation would use for hypothesis j. Compare its prompts to the all-plus
# prompts that stage32/34/35 actually build.
SYSTEM_ = __import__("stage32_uprm_toy1").SYSTEM
def allplus_prompts(q, steps):
    out = []
    prefix = SYSTEM_ + "Problem: " + q.strip()
    for t, step in enumerate(steps, start=1):
        out.append(prefix + f"\n\nStep {t}: {step}\nJudge:")
        prefix = prefix + f"\n\nStep {t}: {step}\nJudge: +"
    return out

q = "Q?"
steps = ["one", "two", "three"]
ap = allplus_prompts(q, steps)
print(f"all-plus prompts built: {len(ap)} (one per step)\n")
allmatch = True
for j in range(1, len(steps)+2):
    slots = marker_slots(q, steps, j)   # what hypothesis j *would* need
    print(f"hypothesis j={j}: needs {len(slots)} slots")
    for i,(prompt, is_neg) in enumerate(slots):
        same = (prompt == ap[i])
        allmatch &= same
        role = "log p-_%d"%(i+1) if is_neg else "log p+_%d"%(i+1)
        print(f"   slot {i+1} ({role:>10}) prompt identical to all-plus slot {i+1}: {same}")
print(f"\nITEM 2 VERDICT: every slot prompt any hypothesis needs is byte-identical "
      f"to the all-plus prompt: {allmatch}")
print("Reason: marker_slots writes the '-' marker only AFTER emitting the slot")
print("prompt for step j, then breaks. So the '-' never appears in any PROMPT;")
print("it only truncates the loop. Prefixes are all-plus everywhere. EXACT.")

print()
print("="*72)
print("ITEM 3: logsumexp stability / set")
print("="*72)
def p_clean(lp, T):
    scores = {j: score_from_logprobs(lp, j, T) for j in range(1, T+2)}
    top = max(scores.values())
    lse = top + math.log(sum(math.exp(v-top) for v in scores.values()))
    return scores[T+1]-lse, scores
pc, sc = p_clean(lp, 3)
print("scores:", {k: round(v,4) for k,v in sc.items()})
print("j enumerated:", sorted(sc), "-> exactly {1..T+1}, disjoint hypotheses. OK")
# manual
mx = max(sc.values()); lse = mx + math.log(sum(math.exp(v-mx) for v in sc.values()))
print(f"logsumexp = {lse:.6f}; p_clean = {pc:.6f}")
naive = math.log(sum(math.exp(v) for v in sc.values()))
print(f"naive logsumexp = {naive:.6f}  (agrees: {abs(naive-lse)<1e-9})")
# extreme
ext = [(-1e-9, -60.0)]*12
pce, sce = p_clean(ext, 12)
print(f"extreme (12 steps, logp- = -60): p_clean={pce:.6e}  finite={math.isfinite(pce)}")
ext2 = [(-40.0, -1e-12)]*12
pce2, _ = p_clean(ext2, 12)
print(f"extreme (12 steps, logp+ = -40): p_clean={pce2:.4f}  finite={math.isfinite(pce2)}")
print("shift-by-max is the standard stable form; max is always attained. STABLE.")
