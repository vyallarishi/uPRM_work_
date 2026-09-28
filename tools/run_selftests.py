"""Run every experiment module's --self-test (CPU, no data, no GPU).

torch / vllm / transformers imports are blocked, so nothing heavy can load
(safe on a login node); a self-test that needs them fails loudly instead.

    python tools/run_selftests.py              # the copies in this repo
    python tools/run_selftests.py --baseline   # also run each ~/phase0 original
                                               # and require identical output
"""
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = Path.home() / "phase0"
prov_file = next(ROOT.glob("*/provenance.json"))
meta = json.loads(prov_file.read_text())

BLOCK = (
    "import sys, runpy\n"
    "class _Block:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name.split('.')[0] in ('torch', 'vllm', 'transformers'):\n"
    "            raise ImportError(f'blocked heavy import: {name}')\n"
    "sys.meta_path.insert(0, _Block())\n"
)


def has_self_test(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("add_argument") and n.args \
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--self-test":
            return True
    return False


def run(code, cwd, timeout):
    try:
        p = subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return "timeout", ""


ap = argparse.ArgumentParser()
ap.add_argument("--baseline", action="store_true")
ap.add_argument("--timeout", type=int, default=300)
ap.add_argument("--only", help="substring filter on the module path")
args = ap.parse_args()

rows, bad = [], 0
for new_rel, orig in sorted(meta["python"].items()):
    if args.only and args.only not in new_rel:
        continue
    if not has_self_test(ROOT / new_rel):
        continue
    mod = new_rel[:-3].replace("/", ".")
    rc, out = run(BLOCK + f"sys.argv=['{mod}','--self-test']\n"
                  f"runpy.run_module('{mod}', run_name='__main__', alter_sys=True)\n",
                  ROOT, args.timeout)
    status = "PASS" if rc == 0 else f"FAIL(rc={rc})"
    if args.baseline:
        brc, bout = run(BLOCK + f"sys.path.insert(0, '{PHASE0}')\n"
                        f"sys.argv=['{orig}','--self-test']\n"
                        f"runpy.run_path('{PHASE0 / orig}', run_name='__main__')\n",
                        PHASE0, args.timeout)
        if brc != rc:
            status += f" | baseline rc={brc} DIFFERS"
        elif bout != out:
            status += " | output DIFFERS from original"
        else:
            status += " | identical to original"
    if not status.startswith("PASS") or "DIFFERS" in status:
        bad += 1
    rows.append((mod, orig, status, out))
    print(f"{status:40s} {mod}  (was {orig})", flush=True)

print(f"\n{len(rows)} self-tests, {bad} problems")
for mod, orig, status, out in rows:
    if not status.startswith("PASS"):
        print(f"\n--- {mod} ---\n{out[-2000:]}")
sys.exit(1 if bad else 0)
