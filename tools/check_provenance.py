"""Verify every experiment module against its ~/phase0 original.

Each differing line must be one of the permitted edits (see README, "What
changed"); anything else is reported as a FAIL.

    python tools/check_provenance.py
"""
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = Path.home() / "phase0"
PHASE0_LIT = "/home/rvyalla/phase0"

prov_file = next(ROOT.glob("*/provenance.json"))
PKG = prov_file.parent.name
meta = json.loads(prov_file.read_text())
prov, siblings = meta["python"], list(meta.get("siblings", {}))
local = {p.stem for p in PHASE0.glob("*.py")}
MOD = r"(?:\.+[\w.]*|(?:" + "|".join([PKG] + siblings) + r")\.[\w.]+)"   # relative or sibling-absolute


def is_import_rewrite(old, new):
    o, n = old.strip(), new.strip()
    m = re.match(r"from\s+(\w+)\s+import\s+(.*)$", o)
    if m and m.group(1) in local:
        m2 = re.match(rf"from\s+{MOD}\s+import\s+(.*)$", n)
        return bool(m2) and m2.group(1) == m.group(2)
    m = re.match(r"import\s+(\w+)(?:\s+as\s+(\w+))?\s*(#.*)?$", o)
    if m and m.group(1) in local:
        alias = m.group(2) or m.group(1)
        m2 = re.match(rf"from\s+{MOD}\s+import\s+(\w+)\s+as\s+{alias}\s*(#.*)?$", n)
        return bool(m2) and (m2.group(2) or "") == (m.group(3) or "")
    m = re.match(r"import\s+(.*)$", o)
    if m and ";" in n:
        names = [x.strip() for x in m.group(1).split(",")]
        loc = [x for x in names if x.split()[0] in local]
        rest = ", ".join(x for x in names if x not in loc)
        if len(loc) == 1:
            alias = loc[0].split()[-1]
            return bool(re.match(rf"from\s+{MOD}\s+import\s+\w+\s+as\s+{alias};\s+import\s+{re.escape(rest)}$", n))
    return False


def classify(old, new):
    if old is None:
        return "added RUN_DIR import" if re.fullmatch(r"from \.+paths import RUN_DIR", new.strip()) else None
    if new is None:
        return None
    if is_import_rewrite(old, new):
        return "import"
    m = re.fullmatch(r'(\s*)(\w+) = (Path\(__file__\)\.parent|Path\("/home/rvyalla/phase0"\))\s*', old)
    if m and re.fullmatch(rf"{re.escape(m.group(1))}from \.+paths import RUN_DIR as {m.group(2)}"
                          rf"  # was: {re.escape(m.group(3))}", new):
        return "dir alias -> RUN_DIR"
    if "Path(__file__).parent" in old and old.replace("Path(__file__).parent", "RUN_DIR") == new:
        return "Path(__file__).parent -> RUN_DIR"
    if "Path(__file__).with_name(" in old and \
            re.sub(r'Path\(__file__\)\.with_name\(("[^"]*")\)', r"RUN_DIR / \1", old) == new:
        return "with_name -> RUN_DIR"
    if "sys.path" in old and PHASE0_LIT in old and "] removed" in new:
        return "sys.path removed"
    if PHASE0_LIT in old:
        back = new.replace('f"{RUN_DIR}/', 'f"' + PHASE0_LIT + "/")
        if old == back or re.sub(r'f"' + re.escape(PHASE0_LIT), '"' + PHASE0_LIT, back) == old:
            return "phase0 path -> RUN_DIR"
    return None


fails, counts = [], {}
for new_rel, orig in sorted(prov.items()):
    a = (PHASE0 / orig).read_text(encoding="utf-8").split("\n")
    b = (ROOT / new_rel).read_text(encoding="utf-8").split("\n")
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        olds, news = a[i1:i2], b[j1:j2]
        if tag == "replace" and len(olds) != len(news):
            ins = [k for k, l in enumerate(news) if re.fullmatch(r"from \.+paths import RUN_DIR", l.strip())]
            if len(ins) == 1 and len(news) == len(olds) + 1:
                counts["added RUN_DIR import"] = counts.get("added RUN_DIR import", 0) + 1
                news = news[: ins[0]] + news[ins[0] + 1:]
            else:
                fails.append((new_rel, i1 + 1, "\n".join(olds), "\n".join(news)))
                continue
        pairs = list(zip(olds, news)) if tag == "replace" else \
            [(None, n) for n in news] if tag == "insert" else [(o, None) for o in olds]
        for o, n in pairs:
            kind = classify(o, n)
            if kind is None:
                fails.append((new_rel, i1 + 1, o, n))
            else:
                counts[kind] = counts.get(kind, 0) + 1

print(f"experiment modules checked: {len(prov)}")
for k, v in sorted(counts.items()):
    print(f"  {v:4d} x {k}")
if fails:
    print(f"\nFAIL: {len(fails)} unexplained line changes")
    for rel, ln, o, n in fails[:40]:
        print(f"  {rel}:{ln}\n    - {o!r}\n    + {n!r}")
    sys.exit(1)
print("\nPASS: every change is one of the permitted edit types")
