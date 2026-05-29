"""_hs_bootstrap.py -- import bootstrap for the `hyperscalees` (EGGROLL) package.

Problem this solves
-------------------
The deliverable modules (unimol_jax.py / esol_pipeline.py / train_eggroll.py)
run as top-level modules, not inside the `hyperscalees` package, so relative
imports into it (`from .base_model import ...`) are impossible. Importing
`hyperscalees` the normal way also fails:

  * `hyperscalees` is usually cloned as a *src-layout repo*, so the folder
    literally named `hyperscalees/` is the REPOSITORY, and the importable
    package lives one level deeper at `<repo>/src/hyperscalees/`. The repo
    root has no `__init__.py`, so a bare `import hyperscalees` resolves the
    repo root as an empty namespace package (`__file__` is None, and it has
    no `models` submodule) -- exactly the original
    `ModuleNotFoundError: No module named 'hyperscalees.models'` /
    `TypeError: ... not NoneType` crash.

  * Even when the real package is on the path, `hyperscalees/models/__init__.py`
    does `from . import rl, llm`, pulling heavy optional deps (gymnax / distrax)
    that are absent on a CPU-only box.

Fix
---
1. LOCATE the real package directory (the one that actually contains
   `models/` and `noiser/`), descending into `src/` layouts and, if needed,
   doing a bounded recursive search of the project tree.
2. Register lightweight *package shims* for `hyperscalees`,
   `hyperscalees.models` and `hyperscalees.noiser` directly in `sys.modules`,
   BEFORE anything triggers a real import. A shim is a bare module object
   whose `__path__` points at the real package directory. With the shims in
   place, `importlib.import_module` loads only the leaf modules actually
   needed -- `models/base_model.py`, `models/common.py`, `noiser/eggroll.py`
   and their light siblings -- directly from disk, WITHOUT ever executing the
   heavy `__init__.py` files.

This is layout-agnostic and never touches `hyperscalees.__file__`.

Public API
----------
    bootstrap_hyperscalees(hs_dir=None, search_from=None) -> str
        Idempotent. Register the shims; return the package directory.
        `hs_dir` may be either the package directory itself OR a repo root
        that contains it -- it is resolved automatically. If omitted, the
        package is located by searching sys.path and the project tree
        around `search_from`.

    import_hs(*names)
        Convenience: bootstrap (idempotent) then import and return the named
        leaf modules.
"""
import importlib
import importlib.machinery
import os
import sys
import types

# sub-packages we shim so their (heavy) __init__.py is never executed
_SHIM_SUBPACKAGES = ("models", "noiser")

# directories never worth descending into during the recursive search
_SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules",
              ".venv", "venv", "env", ".env", "site-packages",
              ".ipynb_checkpoints", "build", "dist", ".mypy_cache",
              ".pytest_cache", ".tox", ".idea", ".vscode", "tok_files"}

_HS_DIR = None          # resolved hyperscalees package directory (cached)


def _looks_like_hyperscalees(path):
    """True if `path` is the hyperscalees PACKAGE dir (has the leaves we need).

    The repo root does NOT satisfy this (it has src/, tests/, llm_experiments/
    but no models/ or noiser/ directly inside) -- only the real package does.
    """
    return (os.path.isdir(os.path.join(path, "models"))
            and os.path.isdir(os.path.join(path, "noiser"))
            and os.path.isfile(os.path.join(path, "models", "common.py"))
            and os.path.isfile(os.path.join(path, "models", "base_model.py")))


def _resolve_to_package(path):
    """Given a directory that is either the package itself or a repo/parent
    containing it, return the actual package directory, or None."""
    if not os.path.isdir(path):
        return None
    if _looks_like_hyperscalees(path):
        return path
    for rel in (os.path.join("src", "hyperscalees"),
                "hyperscalees",
                os.path.join("hyperscalees", "src", "hyperscalees"),
                os.path.join("hyperscalees", "hyperscalees")):
        cand = os.path.join(path, rel)
        if os.path.isdir(cand) and _looks_like_hyperscalees(cand):
            return cand
    return None


def _walk_for_package(root, max_depth):
    """Depth-limited walk under `root` for the hyperscalees package directory.

    Returns the shallowest match (the intended package, not a stray copy), or
    None. Skips obvious junk directories.
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return None
    matches = []
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        if os.path.basename(d) == "hyperscalees" and _looks_like_hyperscalees(d):
            matches.append(d)
            continue                       # found a package; don't descend in
        if depth >= max_depth:
            continue
        try:
            entries = sorted(os.listdir(d))
        except OSError:
            continue
        for name in entries:
            if name in _SKIP_DIRS or name.startswith("."):
                continue
            sub = os.path.join(d, name)
            if os.path.isdir(sub):
                stack.append((sub, depth + 1))
    if not matches:
        return None
    matches.sort(key=lambda p: (p.count(os.sep), len(p)))
    return matches[0]


def _find_hyperscalees(search_from):
    """Locate the `hyperscalees` PACKAGE directory.

    Order: (1) exact candidates off sys.path and `search_from`/parents,
    including src-layout nestings; (2) a bounded recursive walk of the
    project tree. Returns the directory, or raises ImportError.
    """
    search_from = os.path.abspath(search_from)

    bases = []
    for p in list(sys.path):
        if p:
            bases.append(os.path.abspath(p))
    d = search_from
    for _ in range(6):
        bases.append(d)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent

    seen = set()
    for base in bases:
        for rel in ("hyperscalees",
                    os.path.join("hyperscalees", "src", "hyperscalees"),
                    os.path.join("hyperscalees", "hyperscalees"),
                    os.path.join("src", "hyperscalees")):
            cand = os.path.normpath(os.path.join(base, rel))
            if cand in seen:
                continue
            seen.add(cand)
            if os.path.isdir(cand) and _looks_like_hyperscalees(cand):
                return cand

    d = search_from
    for _ in range(4):
        hit = _walk_for_package(d, max_depth=6)
        if hit is not None:
            return hit
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent

    raise ImportError(
        "_hs_bootstrap: could not locate the 'hyperscalees' PACKAGE.\n"
        "Searched sys.path and the project tree at/above:\n    "
        + search_from
        + "\nThe package directory is the one that directly contains "
          "'models/' and 'noiser/' (for a src-layout clone that is "
          "<repo>/src/hyperscalees/). Pass it explicitly via "
          "bootstrap_hyperscalees(hs_dir=...) or run.py's HYPERSCALEES_DIR.")


def _make_pkg(name, directory):
    """Create (or reuse) a lightweight package shim module for `name`."""
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__hs_shim__", False):
        return existing
    mod = types.ModuleType(name)
    mod.__path__ = [directory]                       # marks the module as a package
    mod.__package__ = name
    mod.__hs_shim__ = True                           # our marker (idempotency)
    init_py = os.path.join(directory, "__init__.py")
    mod.__file__ = init_py if os.path.isfile(init_py) else None
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [directory]
    mod.__spec__ = spec
    sys.modules[name] = mod
    return mod


def bootstrap_hyperscalees(hs_dir=None, search_from=None):
    """Register lightweight `hyperscalees` package shims. Idempotent.

    `hs_dir` may be the package directory itself or any parent/repo that
    contains it (src-layout is resolved automatically). If omitted, the
    package is auto-located. Returns the resolved package directory.
    """
    global _HS_DIR
    if _HS_DIR is not None and hs_dir is None:
        needed = ("hyperscalees",) + tuple(f"hyperscalees.{s}"
                                           for s in _SHIM_SUBPACKAGES)
        if all(n in sys.modules for n in needed):
            return _HS_DIR

    if hs_dir is not None:
        resolved = _resolve_to_package(os.path.abspath(hs_dir))
        if resolved is None:
            raise ImportError(
                f"_hs_bootstrap: {hs_dir!r} is not (and does not contain) "
                f"the 'hyperscalees' package -- no directory with models/ + "
                f"noiser/ found there.")
        hs_dir = resolved
    else:
        if search_from is None:
            search_from = os.path.dirname(os.path.abspath(__file__))
        hs_dir = _find_hyperscalees(search_from)

    _make_pkg("hyperscalees", hs_dir)                # top-level package shim
    for sub in _SHIM_SUBPACKAGES:                    # sub-package shims
        subdir = os.path.join(hs_dir, sub)
        if os.path.isdir(subdir):
            _make_pkg(f"hyperscalees.{sub}", subdir)

    _HS_DIR = hs_dir
    return hs_dir


def import_hs(*names):
    """Bootstrap (idempotent), then import and return the named leaf modules."""
    bootstrap_hyperscalees()
    mods = tuple(importlib.import_module(n) for n in names)
    return mods[0] if len(mods) == 1 else mods