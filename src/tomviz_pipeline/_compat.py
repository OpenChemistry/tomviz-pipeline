###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Compatibility shims for operator scripts written against the old
`tomviz` import package.

Operator scripts embedded in state files contain `import tomviz.operators`
/ `import tomviz.nodes` / `import tomviz.utils` (and sometimes reference
`tomviz.utils.*` without importing it). install_script_module_aliases()
registers sys.modules aliases mapping those names onto this package so
the scripts run unchanged — unless a real `tomviz` package is installed,
in which case it wins and no aliases are registered."""

import importlib
import importlib.machinery
import importlib.util
import sys
import types

# Alias name (as operator scripts import it) -> module in this package.
_ALIASES = {
    'tomviz.operators': 'tomviz_pipeline.operators',
    'tomviz.nodes': 'tomviz_pipeline._legacy_nodes',
    'tomviz.dataset': 'tomviz_pipeline.dataset',
    'tomviz.external_dataset': 'tomviz_pipeline.dataset',
    'tomviz.utils': 'tomviz_pipeline.utils',
}

_decided = False


def install_script_module_aliases():
    """Register `tomviz.*` sys.modules aliases for operator scripts.

    Idempotent, and deliberately conservative: the decision is made once
    per process using importlib.util.find_spec('tomviz') — if a real
    tomviz package is importable (e.g. this library runs inside an
    environment that also has the full tomviz Python package), nothing
    is aliased and the real package resolves as usual."""
    global _decided
    if _decided:
        return
    _decided = True

    try:
        spec = importlib.util.find_spec('tomviz')
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        # A real tomviz install wins; never shadow it.
        return

    # Build the `tomviz` shim package. Mark it as a (namespace-like)
    # package with a proper spec so later find_spec('tomviz') calls see
    # a well-formed module instead of raising.
    shim = types.ModuleType('tomviz')
    shim.__doc__ = ('Compatibility shim mapping tomviz.* onto '
                    'tomviz_pipeline (see tomviz_pipeline._compat).')
    shim.__path__ = []
    shim.__spec__ = importlib.machinery.ModuleSpec(
        'tomviz', None, is_package=True)
    shim.__spec__.submodule_search_locations = shim.__path__
    sys.modules['tomviz'] = shim

    for alias, target in _ALIASES.items():
        module = importlib.import_module(target)
        sys.modules[alias] = module
        # `from tomviz import operators` and attribute access like
        # `tomviz.utils.foo` resolve through the parent module's
        # attribute, not sys.modules — set it too.
        setattr(shim, alias.rsplit('.', 1)[1], module)


def _augment_with_legacy(bases: list, module_name: str,
                         class_names: 'tuple[str, ...]') -> tuple:
    """Extend `bases` with the same-named classes from a real `tomviz`
    install, when one is importable and distinct from ours. Needed in
    environments where the old tomviz Python package coexists with this
    one: operator scripts there subclass the *old* base classes, and
    class discovery must recognize both."""
    try:
        legacy = importlib.import_module(module_name)
    except ImportError:
        return tuple(bases)
    for name in class_names:
        cls = getattr(legacy, name, None)
        if cls is not None and cls not in bases:
            bases.append(cls)
    return tuple(bases)


def operator_base_classes() -> tuple:
    """Base classes a v1 operator script's class may derive from:
    tomviz_pipeline.operators.Operator plus, if a real tomviz package is
    installed, its tomviz.operators.Operator."""
    from tomviz_pipeline import operators
    return _augment_with_legacy([operators.Operator],
                                'tomviz.operators', ('Operator',))


def kernel_base_classes(kind: str) -> tuple:
    """Base classes a schema-v2 script's kernel class may derive from
    ('source' or 'transform'): tomviz_pipeline.kernels.SourceKernel /
    TransformKernel plus, if a real tomviz package is installed, its
    legacy tomviz.nodes.SourceNode / TransformNode. (When no real
    tomviz exists, tomviz.nodes is our alias and contributes the same
    class objects — deduped.)"""
    from tomviz_pipeline import kernels
    if kind == 'source':
        ours, name = kernels.SourceKernel, 'SourceNode'
    else:
        ours, name = kernels.TransformKernel, 'TransformNode'
    return _augment_with_legacy([ours], 'tomviz.nodes', (name,))
