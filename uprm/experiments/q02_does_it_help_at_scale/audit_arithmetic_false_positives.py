"""AUDIT item 5: can first_bad_step_safe produce FALSE POSITIVES?"""
from ..q01_does_step_scoring_work.toy1_first_broken_step import eval_equation
from .error_types import first_bad_step_safe
from icm.experiments.q08_earlier_8b_study.rung3_reasoning_structure import _EQUATION_RE, _OP_SPLIT_RE

print("="*72); print("A. Synthetic probes for false positives"); print("="*72)
cases = [
 # (step text, is the arithmetic ACTUALLY fine?, note)
 ("He needs 10 / 3 = <<10/3=3>>3 boxes.", "ROUNDING (3.33->3)", None),
 ("Cost is 100 * 1.5 = <<100*1.5=150>>150.", "fine", None),
 ("So 2 + 3 * 4 = <<2+3*4=14>>14 (correct precedence).", "PRECEDENCE 14", None),
 ("Half of 7 is <<7/2=3>>3 people.", "ROUNDING", None),
 ("He earns 5 - 2 + 1 = <<5-2+1=4>>4.", "fine", None),
 ("Total <<-5+3=-2>>-2", "unary minus", None),
 ("<<1,200+300=1500>>1500", "comma thousands", None),
 ("<<10%*50=5>>5", "percent sign", None),
 ("<<3^2=9>>9", "exponent", None),
 ("<<2*3=6>>6 and <<6+1=8>>8", "second eq FALSE", None),
]
for text, note, _ in cases:
    idx = first_bad_step_safe([text])
    eqs = _EQUATION_RE.findall(text)
    print(f"  bad_step={str(idx):<5} {note:<22} eqs={eqs}  text={text[:50]!r}")

print()
print("  --- precedence check in detail ---")
print("   eval_equation('2+3*4','14') =", eval_equation("2+3*4","14"), " <- LEFT-TO-RIGHT gives 20, so 14 is called FALSE")
print("   eval_equation('2+3*4','20') =", eval_equation("2+3*4","20"), " <- code accepts the LEFT-TO-RIGHT value")
print("   Standard math: 2+3*4 = 14.  The evaluator IGNORES operator precedence.")
print()
print("  --- rounding / integer division ---")
print("   eval_equation('10/3','3') =", eval_equation("10/3","3"), "(true value 3.333; tol 0.5% -> FALSE)")
print("   eval_equation('7/2','3')  =", eval_equation("7/2","3"))

print()
print("="*72); print("B. Real data: how often does a mixed +/* or +/- expression occur?"); print("="*72)
import json, re
from collections import Counter
from icm.experiments.q04_full_scale_headline.full_scale_37x1500 import load
from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q01_does_step_scoring_work.toy1_first_broken_step import split_steps

scores=set()
for sh in range(4):
    for line in open(f"stage35_scores_{sh}of4.jsonl", encoding="utf-8"):
        r=json.loads(line); scores.add((r["iid"], r["k"]))
print("scored chains:", len(scores))
pools = load("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500")
print("pools loaded:", len(pools))

MIXED = re.compile(r"[*/×÷]")
ADD   = re.compile(r"(?<=[\d\)])\s*[+]|(?<=\d)\s*-\s*(?=\d)")
n_flag=0; n_prec=0; n_prec_would_pass=0; ex=[]
prec_ex=[]
for (iid,k) in scores:
    pool=pools.get(iid)
    if pool is None: continue
    steps=split_steps(pool["chains"][k]["text"])
    idx=first_bad_step_safe(steps)
    if idx is None: continue
    n_flag+=1
    # find the offending equation
    st=steps[idx-1]
    for body in _EQUATION_RE.findall(st):
        if "=" not in body: continue
        lhs,_,rhs=body.rpartition("=")
        lhs2=lhs.replace("×","*").replace("÷","/").replace(" ","")
        from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import norm as safe_norm
        right=safe_norm(rhs)
        if not lhs2 or right is None: continue
        try: v=eval_equation(lhs2,right)
        except Exception: continue
        if v is False:
            ops=_OP_SPLIT_RE.findall(lhs2)
            # mixed-precedence: contains at least one of */ AND at least one of +-
            has_md = any(o in "*/×÷" for o in ops)
            has_as = any(o in "+-" for o in ops)
            if has_md and has_as:
                n_prec+=1
                # would correct precedence make it TRUE?
                try:
                    val=eval(lhs2.replace("×","*").replace("÷","/"))
                    from decimal import Decimal
                    want=Decimal(right)
                    okp = abs(Decimal(str(val))-want)/max(abs(Decimal(str(val))),abs(want),Decimal(1))<=Decimal("0.005")
                except Exception:
                    okp=False
                if okp:
                    n_prec_would_pass+=1
                    if len(prec_ex)<12: prec_ex.append((iid,k,idx,body))
            break
print(f"\nchains flagged with a false equation: {n_flag}")
print(f"  of those, offending eq mixes */ with +- (precedence-sensitive): {n_prec}")
print(f"  of those, CORRECT precedence makes the equation TRUE (= FALSE POSITIVE): {n_prec_would_pass}")
if n_flag: print(f"  false-positive rate among flagged chains: {100*n_prec_would_pass/n_flag:.2f}%")
print("\n  examples (iid, k, step, equation):")
for e in prec_ex: print("   ", e)
