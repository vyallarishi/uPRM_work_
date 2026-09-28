"""Static import check (no code is executed).

For every relative import, and every absolute import of a sibling repository
(icm.…, uprm.…): the target module must exist and every imported name must be
bound at its top level. Fails if any module still imports a ~/phase0 name
(stageNN_*, rung*, …) directly, or uses RUN_DIR before the import that binds it.

    python tools/check_imports.py
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
prov_file = next(ROOT.glob("*/provenance.json"))
PKG_DIR = prov_file.parent
meta = json.loads(prov_file.read_text())
old_names = {Path(v).stem for v in meta["python"].values()}
for sib in meta.get("siblings", {}).values():
    for p in Path(sib).glob("*/provenance.json"):
        old_names |= {Path(v).stem for v in json.loads(p.read_text())["python"].values()}
PKG_ROOTS = {PKG_DIR.name: ROOT, **{name: Path(path) for name, path in meta.get("siblings", {}).items()}}


def top_level_names(tree):
    names = set()
    for node in tree.body:
        for n in ast.walk(node) if isinstance(node, (ast.If, ast.Try)) else [node]:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            names.add(x.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    names.add((a.asname or a.name).split(".")[0])
    return names


def check_target(rel, node, target):
    if target.with_suffix(".py").is_file():
        names = top_level_names(ast.parse(target.with_suffix(".py").read_text(encoding="utf-8")))
        for a in node.names:
            if a.name != "*" and a.name not in names:
                errors.append(f"{rel}:{node.lineno} {a.name!r} not defined in {target.with_suffix('.py')}")
    elif (target / "__init__.py").is_file():
        for a in node.names:
            if not (target / f"{a.name}.py").is_file():
                errors.append(f"{rel}:{node.lineno} no module {a.name} in {target}")
    else:
        errors.append(f"{rel}:{node.lineno} unresolved import target {target}")


errors, n_rel, n_abs = [], 0, 0
modules = sorted(PKG_DIR.rglob("*.py"))
for path in modules:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rel = path.relative_to(ROOT)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in old_names:
                    errors.append(f"{rel}:{node.lineno} direct import of phase0 module {a.name}")
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            top = (node.module or "").split(".")[0]
            if top in old_names:
                errors.append(f"{rel}:{node.lineno} direct import of phase0 module {node.module}")
            elif top in PKG_ROOTS:
                n_abs += 1
                check_target(rel, node, PKG_ROOTS[top].joinpath(*node.module.split(".")))
            continue
        n_rel += 1
        base = path.parent
        for _ in range(node.level - 1):
            base = base.parent
        check_target(rel, node, base.joinpath(*(node.module.split(".") if node.module else [])))

for path in modules:
    lines = path.read_text(encoding="utf-8").split("\n")
    bind = [i for i, l in enumerate(lines) if "import RUN_DIR" in l]
    uses = [i for i, l in enumerate(lines) if "RUN_DIR" in l and i not in bind
            and not l.lstrip().startswith("#")]
    if bind and uses and min(uses) < min(bind):
        errors.append(f"{path.relative_to(ROOT)}:{min(uses)+1} uses RUN_DIR before its import "
                      f"at line {min(bind)+1}")

print(f"modules: {len(modules)}, relative imports: {n_rel}, sibling-repo imports: {n_abs}")
if errors:
    print(f"FAIL: {len(errors)}")
    for e in errors:
        print("  " + e)
    sys.exit(1)
print("PASS: every import resolves and every imported name exists")
