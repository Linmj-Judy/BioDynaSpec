#!/usr/bin/env python
"""Static import-graph check for BioDynaSpec.

Two jobs:
  1. scan      - report every unresolved local import in the tree
  2. reachable - from the real entry points, walk the local import closure and
                 report any unresolved import that is actually reachable

Resolution mirrors the launch scripts: PYTHONPATH is project root + src/,
so both `from main_model import X` and `from src.utils.file_io import Y` work.
Relative imports are resolved against the owning package.

Usage:
  python check_imports.py            # whole-tree scan
  python check_imports.py --chain    # entry-point reachability
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SEARCH = [ROOT, os.path.join(ROOT, "src")]
SKIP_DIRS = {"__pycache__", ".git", "openfold_local"}

# The 5 real entry modules invoked by scripts/*.sh (see AUDIT.md)
ENTRIES = [
    "trainer_far",                                             # pretrain.sh
    "inferencer_far",                                          # inference.sh
    "runner/data_prepare/replace_positions_full_fft_atlas",    # generate_data.sh
    "runner/data_prepare/compute_fft_window_scaling",          # generate_data.sh
    "runner/data_prepare/extract_frames/save_frames_to_npy_atlas",  # extract_frames.sh
]

STDLIB = set(sys.stdlib_module_names)

# Third-party packages we cannot resolve locally and must not flag.
THIRDPARTY_TOPS = {
    "torch", "numpy", "scipy", "pandas", "matplotlib", "tqdm", "seaborn",
    "pytorch_lightning", "lightning", "torchmetrics", "ray", "wandb", "yaml",
    "esm", "MDAnalysis", "Bio", "biotite", "rdkit", "torchdrug", "flash_attn",
    "deepspeed", "einops", "omegaconf", "pdbx", "gemmi", "pdbeccdutils",
    "networkx", "sklearn", "joblib", "requests", "numba", "protenix",
    "openfold", "ml_collections", "absl", "attr", "attrs", "urllib3",
    "bio2token", "megatron", "alphafold3", "haiku", "jax", "pytest", "optree",
}


def resolve(mod):
    """Return the .py path for a dotted module, or None."""
    parts = mod.split(".")
    for base in SEARCH:
        pkg = os.path.join(base, *parts)
        if os.path.isfile(os.path.join(pkg, "__init__.py")):
            return os.path.join(pkg, "__init__.py")
        for cand in (pkg + ".py", pkg + ".so", pkg + ".pyd"):
            if os.path.isfile(cand):
                return cand
        if os.path.isdir(pkg):  # namespace package
            return os.path.join(pkg, "__init__.py")
    return None


def path_to_module(path):
    rel = os.path.relpath(path, ROOT)
    if rel.endswith(".py"):
        rel = rel[:-3]
    if rel.endswith("/__init__"):
        rel = rel[: -len("/__init__")]
    return rel.replace(os.sep, ".")


def read(path):
    try:
        return open(path, encoding="utf-8").read()
    except UnicodeDecodeError:
        try:
            return open(path, encoding="latin-1").read()
        except Exception:
            return ""


def imports_of(path):
    """Yield (module_name, lineno) for each import in the file."""
    src = read(path)
    if not src or "\x00" in src:
        return
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    pkg = path_to_module(path).rsplit(".", 1)[0] if path.endswith(".py") else ""
    # package of this file (for relative imports)
    filedir = os.path.dirname(os.path.relpath(path, ROOT))
    filepkg = filedir.replace(os.sep, ".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = filepkg
                for _ in range(node.level - 1):
                    base = base.rsplit(".", 1)[0] if "." in base else ""
                mod = f"{base}.{node.module}" if node.module and base else (node.module or base)
            else:
                mod = node.module or ""
            if mod:
                yield mod, node.lineno


def classify(path, mod, lineno):
    """-> ('ok'|'external'|'broken', detail)"""
    if resolve(mod):
        return "ok", resolve(mod)
    top = mod.split(".")[0]
    if top in STDLIB or top in THIRDPARTY_TOPS:
        return "external", None
    # bare sibling import (module run as a script from its own dir)
    if resolve(os.path.basename(mod)) or os.path.isfile(
        os.path.join(os.path.dirname(path), mod.split(".")[0] + ".py")
    ):
        return "ok", None
    return "broken", None


def scan_all():
    broken = []
    n = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            n += 1
            rel = os.path.relpath(p, ROOT)
            src = read(p)
            if "\x00" in src:
                broken.append((rel, 0, "binary/AppleDouble file"))
                continue
            try:
                ast.parse(src)
            except SyntaxError as e:
                broken.append((rel, 0, f"SYNTAX ERROR: {e}"))
                continue
            for mod, ln in imports_of(p):
                kind, _ = classify(p, mod, ln)
                if kind == "broken":
                    broken.append((rel, ln, mod))
    return n, broken


def reachable():
    """BFS the local import closure from ENTRIES; report broken imports hit."""
    seen, broken, queue = set(), [], []
    for e in ENTRIES:
        p = resolve(e)
        if not p or not os.path.isfile(p):
            broken.append((e, 0, "ENTRY POINT NOT FOUND"))
            continue
        queue.append(p)
    while queue:
        p = queue.pop()
        rp = os.path.abspath(p)
        if rp in seen:
            continue
        seen.add(rp)
        rel = os.path.relpath(rp, ROOT)
        for mod, ln in imports_of(rp):
            kind, target = classify(rp, mod, ln)
            if kind == "broken":
                broken.append((rel, ln, mod))
            elif kind == "ok" and target and os.path.isfile(target):
                if os.path.abspath(target) not in seen:
                    queue.append(target)
    return seen, broken


def main():
    if "--chain" in sys.argv:
        seen, broken = reachable()
        print(f"=== entry-point closure: {len(seen)} local modules reachable ===")
        for p in sorted(seen):
            print("  " + os.path.relpath(p, ROOT))
        print(f"\n=== BROKEN imports reachable from entry points: {len(broken)} ===")
        for rel, ln, m in broken:
            print(f"  {rel}:{ln}  ->  {m}")
        return 1 if broken else 0

    n, broken = scan_all()
    print(f"scanned {n} files")
    print(f"\n=== unresolved local imports: {len(broken)} ===")
    for rel, ln, m in broken:
        print(f"  {rel}:{ln}  ->  {m}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
