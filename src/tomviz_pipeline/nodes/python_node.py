###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""PythonNode: the user-facing constructor for schema-v2 python nodes.

A python node is built from exactly two artifacts:

- ``definition`` — the *interface*: ports, parameters, label. This is
  the operator JSON description, verbatim (the same vocabulary as the
  ``.json`` sidecar files and the ``description`` field of state
  entries). Accepted as a dict, a JSON string, or a path to a ``.json``
  file.
- ``kernel`` — the *compute*: a ``tomviz_pipeline.kernels`` class you
  hold, the source text of an operator script, or a path to a ``.py``
  file. Strings are always content; paths are always ``pathlib.Path``
  (or any ``os.PathLike``).

Both are required — a node that could never execute cannot be built.
The definition decides the host type: no ``inputs`` means a
``source.python`` node (``PythonSource``), otherwise a
``transform.python`` node (``PythonTransform``); ``PythonNode(...)``
returns the right one and ``isinstance(node, PythonNode)`` holds for
both.

    from tomviz_pipeline import Pipeline, PythonNode
    from tomviz_pipeline.kernels import TransformKernel

    class Multiply(TransformKernel):
        def transform(self, inputs, factor=2.0):
            ds = inputs['volume']
            return {'volume': ds.apply_to_each_scalar_array(
                lambda a: a * factor)}

    MULTIPLY = {
        'name': 'Multiply',
        'inputs':  [{'name': 'volume', 'type': 'ImageData'}],
        'outputs': [{'name': 'volume', 'type': 'ImageData'}],
        'parameters': [{'name': 'factor', 'type': 'double',
                        'default': 2.0}],
    }

    node = PythonNode(MULTIPLY, kernel=Multiply)
    node.set_parameters(factor=3.0)

A class-bound kernel executes in-process directly. When the node must
be serialized — saving a state file, or running under an
ExternalNodeExecutor, whose child process only sees the serialized form
— the class is re-expressed as a script by capturing its source
(``inspect.getsource`` plus an import prologue). That requires the
class body to be self-contained apart from numpy / tomviz_pipeline
imports; kernels needing module-level helpers should be given as script
text or a ``.py`` path instead.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import textwrap
from pathlib import Path

logger = logging.getLogger('tomviz_pipeline')

# Imports prepended to a captured class body so the generated script is
# self-contained under both the canonical and the legacy spellings.
_SCRIPT_PROLOGUE = '''\
import numpy
import numpy as np

import tomviz_pipeline
import tomviz_pipeline.dataset
import tomviz_pipeline.kernels
from tomviz_pipeline.dataset import Dataset
from tomviz_pipeline.kernels import Kernel, SourceKernel, TransformKernel


'''


class PythonNode:
    """Facade constructor and common base of PythonSource /
    PythonTransform. See the module docstring for usage."""

    def __new__(cls, definition=None, kernel=None):
        if cls is not PythonNode:
            # Concrete host construction (NodeFactory / deserialize
            # path): plain instance, the subclass __init__ runs next.
            return super().__new__(cls)

        if definition is None:
            raise TypeError(
                "PythonNode() missing required argument: 'definition'")
        if kernel is None:
            raise TypeError(
                "PythonNode() missing required argument: 'kernel'")

        defn = _normalize_definition(definition)
        kernel_class, script = _normalize_kernel(kernel)

        if defn.get('inputs'):
            from tomviz_pipeline.nodes.transforms.python_transform import (
                PythonTransform,
            )
            host = PythonTransform()
            kind = 'transform'
        else:
            from tomviz_pipeline.nodes.sources.python_source import (
                PythonSource,
            )
            host = PythonSource()
            kind = 'source'

        if kernel_class is not None:
            from tomviz_pipeline._compat import kernel_base_classes
            if not issubclass(kernel_class, kernel_base_classes(kind)):
                expected = ('SourceKernel' if kind == 'source'
                            else 'TransformKernel')
                raise TypeError(
                    f'{kernel_class.__name__} does not subclass '
                    f'{expected}, which the definition requires '
                    f'({kind} shape)')

        host.set_json_description(json.dumps(defn))
        if not host.label:
            host.label = defn.get('name', '')
        if script is not None:
            host._backend.set_script(script)
        host._backend.kernel_class = kernel_class
        host._facade_constructed = True
        return host

    def __init__(self, *args, **kwargs):
        # Reached only through the concrete hosts' super().__init__()
        # chain (bare construction); the facade path guards in the
        # hosts' own __init__. Continue the normal chain
        # (SourceNode / TransformNode / Node).
        super().__init__()


def script_from_kernel(kernel_class) -> str:
    """Best-effort re-expression of a kernel class as a self-contained
    operator script. Returns '' when the source is not retrievable
    (interactive definitions)."""
    try:
        body = textwrap.dedent(inspect.getsource(kernel_class))
    except (OSError, TypeError):
        return ''
    return _SCRIPT_PROLOGUE + body


def _normalize_definition(definition) -> dict:
    if isinstance(definition, (Path, os.PathLike)):
        text = Path(definition).read_text()
        source = f'definition file {definition}'
    elif isinstance(definition, str):
        text = definition
        source = 'definition string'
    elif isinstance(definition, dict):
        return definition
    else:
        raise TypeError(
            'definition must be a dict, a JSON string, or a path to a '
            f'.json file — got {type(definition).__name__}')
    try:
        defn = json.loads(text)
    except ValueError as exc:
        raise ValueError(f'{source} is not valid JSON: {exc}') from exc
    if not isinstance(defn, dict):
        raise ValueError(f'{source} must contain a JSON object')
    return defn


def _normalize_kernel(kernel):
    """Returns (kernel_class, script) — exactly one is set."""
    if inspect.isclass(kernel):
        return kernel, None
    if isinstance(kernel, (Path, os.PathLike)):
        return None, Path(kernel).read_text()
    if isinstance(kernel, str):
        return None, kernel
    raise TypeError(
        'kernel must be a kernel class, a script string, or a path to '
        f'a .py file — got {type(kernel).__name__}')
