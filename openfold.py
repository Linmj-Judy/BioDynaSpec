"""Compatibility shim: make the top-level ``openfold`` package resolve to the
vendored copy at ``src/openfold_local``.

Why this exists
---------------
``fair-esm``'s ``esm.esmfold.v1`` modules import from the *top-level* package
name, e.g.::

    from openfold.model.triangular_attention import TriangleAttention
    from openfold.np.protein import Protein as OFProtein

``src/featureextractor.py`` uses ``esm.esmfold.v1.tri_self_attn_block``
(``TriangularSelfAttentionBlock``), so those imports run at import time.

This repository does not install upstream OpenFold; it vendors it under
``src/openfold_local`` (see ``src/openfold_local/README.md``). Installing a
second, separately-compiled OpenFold just to satisfy the package name would be
wasteful and could shadow the vendored, locally-patched copy.

How it works
------------
``src.openfold_local`` stays the single canonical implementation (so module
identity is shared — no duplicate classes, no isinstance mismatches).
``sys.modules['openfold']`` is rebound to that package object, and a meta-path
finder lazily aliases ``openfold.X`` -> ``src.openfold_local.X`` on demand.

Lazy matters: ``openfold_local.data`` pulls in mmcif/jackhmmer paths that cost
minutes to import and are never used by BioDynaSpec's training path.

Note the assignment below is *unconditional*: while this module is executing,
``sys.modules['openfold']`` already holds the partially-initialised shim, so
guarding on its absence would silently skip the rebind.

If you later install a real ``openfold`` distribution, remove this file.
"""
import importlib
import importlib.abc
import importlib.util
import sys

import src.openfold_local as _local

_PREFIX = "openfold"
_TARGET = "src.openfold_local"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str):
        self._target = target

    def create_module(self, spec):
        mod = importlib.import_module(self._target)
        sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module):
        # Already executed by import_module above.
        return None


class _OpenfoldAliasFinder(importlib.abc.MetaPathFinder):
    """Resolve ``openfold.X`` to ``src.openfold_local.X`` lazily."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX + "."):
            return None
        suffix = fullname[len(_PREFIX) + 1:]
        target_name = f"{_TARGET}.{suffix}"
        try:
            real = importlib.util.find_spec(target_name)
        except (ImportError, AttributeError, ValueError):
            return None
        if real is None:
            return None
        is_pkg = real.submodule_search_locations is not None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(target_name), is_package=is_pkg
        )


# Rebind ``openfold`` to the vendored package (must be unconditional — see
# module docstring) and make submodule imports resolve through the finder.
sys.modules[_PREFIX] = _local

if not any(isinstance(f, _OpenfoldAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _OpenfoldAliasFinder())
