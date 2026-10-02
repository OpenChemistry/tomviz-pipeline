###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""ScriptableNode: a node whose behavior is supplied as a definition
plus a script, rather than built into the library.

A scriptable node is built from exactly two artifacts:

- ``definition`` — the *interface*: ports, parameters, label. This is
  the JSON definition, verbatim (the same vocabulary as the ``.json``
  sidecar files and the ``description`` field of state entries).
  Accepted as a dict, a JSON string, or a path to a ``.json`` file.
- ``kernel`` — the *compute*: a ``tomviz_pipeline.kernels`` class you
  hold, the source text of a script, or a path to a ``.py`` file.
  Strings are always content; paths are always ``pathlib.Path`` (or
  any ``os.PathLike``).

Both are required — a node that could never execute cannot be built.
An optional ``parameters`` mapping sets initial values in place of the
definition's defaults; it may only name parameters the definition
declares. The definition decides the node class (see
``definition_schema``):

- schema v2 with ``inputs``: ``ScriptableTransformNode``
  (``transform.python``);
- schema v2 without ``inputs``: ``ScriptableSourceNode``
  (``source.python``);
- schema v1, a ``transform(dataset, **params)`` script:
  ``LegacyScriptableTransformNode`` (``transform.legacyPython``). Its
  kernel must be a script; a class is rejected.

``ScriptableNode(...)`` returns the right one, and
``isinstance(node, ScriptableNode)`` holds for all three.

    from tomviz_pipeline import Pipeline, ScriptableNode
    from tomviz_pipeline.kernels import TransformKernel

    class Multiply(TransformKernel):
        def transform(self, inputs, factor=2.0):
            ds = inputs['volume']
            return {'volume': ds.apply_to_each_scalar_array(
                lambda a: a * factor)}

    MULTIPLY = {
        'schemaVersion': 2,
        'name': 'Multiply',
        'inputs':  [{'name': 'volume', 'type': 'ImageData'}],
        'outputs': [{'name': 'volume', 'type': 'ImageData'}],
        'parameters': [{'name': 'factor', 'type': 'double',
                        'default': 2.0}],
    }

    node = ScriptableNode(MULTIPLY, kernel=Multiply,
                          parameters={'factor': 3.0})

A class-bound kernel executes in-process directly. When the node must
be serialized — saving a state file, or running under an
ExternalNodeExecutor, whose child process only sees the serialized form
— the class is re-expressed as a script by capturing its source
(``inspect.getsource`` plus an import prologue). That requires the
class body to be self-contained apart from numpy / tomviz_pipeline
imports; kernels needing module-level helpers should be given as script
text or a ``.py`` path instead.

An existing node's definition and script are edited through
``reconfigure_description()`` and the ``script`` property; both mark
the node stale.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

from .kernel_backends import (
    SCHEMA_V1,
    BaseKernelBackend,
    definition_schema,
    script_from_kernel,
)
from .kernel_backends.base import _definition_schema

__all__ = ['ScriptableNode', 'definition_schema', 'script_from_kernel']


class ScriptableNode:
    """Factory and common base of the scriptable node classes. See the
    module docstring for usage.

    Subclasses combine it with a core node class and name the backend
    that runs their kernel."""

    #: The kernel backend class a subclass runs.
    _backend_class: type[BaseKernelBackend] = BaseKernelBackend

    def __new__(cls, definition=None, kernel=None, parameters=None):
        if cls is not ScriptableNode:
            # Concrete construction (NodeFactory / deserialize path):
            # a plain instance, __init__ runs next.
            return super().__new__(cls)

        if definition is None:
            raise TypeError(
                "ScriptableNode() missing required argument: 'definition'")
        if kernel is None:
            raise TypeError(
                "ScriptableNode() missing required argument: 'kernel'")

        defn = _normalize_definition(definition)
        kernel_class, script = _normalize_kernel(kernel)

        # Warns at the caller of ScriptableNode(...).
        if _definition_schema(defn, stacklevel=3) == SCHEMA_V1:
            if kernel_class is not None:
                raise TypeError(
                    f'{kernel_class.__name__} cannot run a schema v1 '
                    'definition: pass its script text or .py path')
            from tomviz_pipeline.nodes.transforms.legacy_scriptable import (
                LegacyScriptableTransformNode,
            )
            host = LegacyScriptableTransformNode()
        elif defn.get('inputs'):
            from tomviz_pipeline.nodes.transforms.scriptable import (
                ScriptableTransformNode,
            )
            host = ScriptableTransformNode()
        else:
            from tomviz_pipeline.nodes.sources.scriptable import (
                ScriptableSourceNode,
            )
            host = ScriptableSourceNode()

        if kernel_class is not None:
            from tomviz_pipeline._compat import kernel_base_classes
            kind = 'transform' if defn.get('inputs') else 'source'
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
        else:
            host._backend.kernel_class = kernel_class
        if parameters:
            unknown = sorted(set(parameters)
                             - set(host._backend.declared_parameters()))
            if unknown:
                raise ValueError(
                    'the definition declares no parameter '
                    + ', '.join(repr(name) for name in unknown))
            # The node is in no graph yet: nothing to mark stale.
            host._backend.parameters.update(parameters)
        # Python calls __init__ again on the returned instance, with the
        # factory arguments; this tells it the node is already built.
        host._factory_built = True
        return host

    def __init__(self, *args, **kwargs):
        if self.__dict__.pop('_factory_built', False):
            return
        # Continue the chain: SourceNode / TransformNode / Node.
        super().__init__()
        self._backend = self._backend_class()
        self._backend.init_ports(self)

    # ---- definition / script -------------------------------------------

    @property
    def json_description(self) -> str:
        """The node's JSON definition, as text."""
        return self._backend.json_description

    def set_json_description(self, json_str: str):
        """Install the definition of a node being built: parameters take
        their declared defaults, the definition's label (if any) becomes
        the node's, and the ports it declares are created. To change
        the definition of a node in use, call
        ``reconfigure_description()``."""
        self._backend.set_json_description(json_str)
        if self._backend.default_label:
            self.label = self._backend.default_label
        self._backend.apply_ports(self)

    def reconfigure_description(self, json_str: str) -> list[str]:
        """Replace the definition of an existing node, as a definition
        editor does. Parameter values carry over where the new
        definition still declares them with the same type (and, for an
        enumeration, still offers them); the others go back to their
        defaults and their names are returned. Ports, the label and the
        executor are left alone, and the node is marked stale.

        Raises ``ValueError``, leaving the node untouched, when the
        definition is not a JSON object or would change the node's
        schema or ports: those are fixed once a node exists."""
        reset = self._backend.reconfigure(json_str)
        self.mark_stale()
        return reset

    @property
    def script(self) -> str:
        """The node's script. Setting it replaces a class-bound kernel
        and marks the node stale."""
        return self._backend.script

    @script.setter
    def script(self, value: str):
        self._backend.set_script(value)
        self.mark_stale()

    def _parameter_store(self) -> dict:
        # Parameters live on the backend (serialized as `arguments`);
        # the base Node parameters API operates on that store.
        return self._backend.parameters

    # ---- serialize / deserialize ---------------------------------------

    def serialize(self) -> dict:
        return self._backend.serialize_into(super().serialize())

    def deserialize(self, data: dict) -> bool:
        # The definition goes first: it creates the ports the base class
        # restores per-port state onto.
        self._backend.load(data)
        if self._backend.default_label:
            self.label = self._backend.default_label
        self._backend.apply_ports(self)
        if not super().deserialize(data):
            return False
        # Honor a `tomviz_pipeline_env` key in the definition through
        # the per-node executor mechanism.
        from tomviz_pipeline.external import promote_description_env_path
        promote_description_env_path(
            self, data, self._backend.external_python_env_path)
        return True

    # ---- execution -----------------------------------------------------

    def query_should_auto_execute(self) -> bool:
        return self._backend.run_should_auto_execute(self)


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
