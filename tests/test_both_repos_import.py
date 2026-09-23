"""Every module in BOTH repos still imports.

WHY. tools/check_repo_sync.py lists three files that are MEANT to differ
between solar-map and solar-wellington, and stops comparing them. That is
right -- Wellington is a single-region deploy and its patch_buildings.py
genuinely does something else -- but it also means a shared function can have
its signature changed in one repo while the allowed-divergent file in the
other still calls the old one. That happened on 22 September: dropping the
unused `dem` and `dem_inv` parameters from gate_area() left Wellington's
patch_buildings.py passing four arguments to a function that now takes two.
The sync check passed. Nothing else would have caught it until a Wellington
rebuild crashed.

A byte-compile would not catch it -- the call is syntactically fine. Importing
does not catch it either, but it catches the much larger class of "a module in
the other repo now references something that no longer exists", and it is the
cheapest check that looks at the sibling repo's code at all.

The signature case specifically is caught by the arity check below, which
reads both repos' call sites for the functions they share.
"""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIBLINGS = [ROOT, *(p for p in (ROOT.parent / "solar-wellington",) if p.is_dir())]


def _modules(repo):
    return sorted(p for p in (repo / "src").glob("*.py") if p.name != "__init__.py")


# A module that reads a data file or parses argv AT IMPORT fails here for
# reasons that have nothing to do with code drift, and those failures are not
# what this test is for. Only errors that mean "a name is wrong" count.
SYMBOL_ERRORS = ("ImportError", "ModuleNotFoundError", "AttributeError",
                 "NameError", "TypeError", "SyntaxError", "IndentationError")


def test_every_module_imports_in_every_repo():
    bad = []
    for repo in SIBLINGS:
        py = repo / ".venv" / "bin" / "python"
        py = py if py.exists() else Path(sys.executable)
        for mod in _modules(repo):
            r = subprocess.run(
                [str(py), "-c", f"import sys; sys.path.insert(0, '.'); "
                                f"import src.{mod.stem}"],
                cwd=repo, capture_output=True, text=True, timeout=180)
            if r.returncode == 0:
                continue
            last = [l for l in r.stderr.strip().splitlines() if l.strip()]
            msg = last[-1] if last else "failed"
            if msg.split(":")[0].split(".")[-1] in SYMBOL_ERRORS:
                bad.append(f"{repo.name}/src/{mod.name}: {msg}")
    assert not bad, "modules that do not import:\n  " + "\n  ".join(bad)


def _defs(repo):
    """name -> (min_args, max_args), for names with ONE unambiguous signature.

    Every def, nested ones included: a call inside a function usually resolves
    to the nested def that shadows the module-level one, and comparing it
    against the outer signature invents failures. Where a name is defined more
    than once with different arities there is no single right answer, so it is
    dropped rather than guessed at.
    """
    seen = {}
    for mod in _modules(repo):
        try:
            tree = ast.parse(mod.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n = len(node.args.args)
                sig = (n - len(node.args.defaults), n,
                       node.args.vararg is not None)
                seen.setdefault(node.name, set()).add(sig)
    return {k: v.pop() for k, v in seen.items() if len(v) == 1}


def test_call_sites_match_the_definition_in_both_repos():
    """A positional call with the wrong number of arguments, anywhere."""
    bad = []
    for repo in SIBLINGS:
        defs = _defs(repo)
        for path in sorted((repo / "src").glob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id in defs):
                    continue
                if any(isinstance(a, ast.Starred) for a in node.args):
                    continue
                lo, hi, star = defs[node.func.id]
                given = len(node.args)
                if star:
                    continue
                if given > hi or given + len(node.keywords) < lo:
                    bad.append(f"{repo.name}/src/{path.name}:{node.lineno} "
                               f"{node.func.id}() takes {lo}-{hi} positional "
                               f"args, called with {given}")
    assert not bad, "call sites that do not match:\n  " + "\n  ".join(bad)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  pass  {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL  {name}: {e}")
    print(f"\n{'all passed' if not fails else str(fails) + ' failed'}")
    sys.exit(1 if fails else 0)
