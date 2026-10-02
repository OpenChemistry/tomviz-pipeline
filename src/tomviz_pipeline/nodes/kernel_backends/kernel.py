###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""The schema-v2 kernel backend. Mirrors the C++
``tomviz::pipeline::PythonNodeBackend``: the definition declares the
ports, and the script defines one ``tomviz_pipeline.kernels``
``SourceKernel`` / ``TransformKernel`` subclass (or its legacy
``tomviz.nodes`` spelling), instantiated fresh for every run. The kernel
may also be bound as a class, which runs in-process directly and is
captured back into a script when the node is serialized."""

from __future__ import annotations

import copy
import inspect
import logging
import textwrap

from tomviz_pipeline._internal import ExecutionContext, attach_execution_context
from tomviz_pipeline.core import PortData, TransformNode

from .base import SCHEMA_V2, BaseKernelBackend


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


def script_from_kernel(kernel_class) -> str:
    """Best-effort re-expression of a kernel class as a self-contained
    script. Returns '' when the source is not retrievable (interactive
    definitions)."""
    try:
        body = textwrap.dedent(inspect.getsource(kernel_class))
    except (OSError, TypeError):
        return ''
    return _SCRIPT_PROLOGUE + body


def _find_kernel_class(module, base_classes):
    """Locate a single kernel subclass of any of ``base_classes`` (a
    class or tuple — our kernels plus a real tomviz install's legacy
    bases, see ``_compat.kernel_base_classes``) in ``module``. Returns
    ``None`` when no subclass is present; raises ``ValueError`` when
    more than one is found (mirrors the C++
    ``PythonNodeUtils::findNodeClass`` contract)."""
    if not isinstance(base_classes, tuple):
        base_classes = (base_classes,)
    found = None
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls in base_classes:
            continue
        if not issubclass(cls, base_classes):
            continue
        if found is not None:
            raise ValueError(
                'Multiple kernel classes defined in module — only one '
                'kernel class can be defined per script.')
        found = cls
    return found


class KernelBackend(BaseKernelBackend):
    """Runs a schema-v2 kernel for a source or transform node."""

    schema_version = SCHEMA_V2

    def __init__(self):
        super().__init__()
        # When set (ScriptableNode(definition, kernel=SomeClass)),
        # execution uses this class directly instead of loading the
        # script. The script and definition still round-trip through
        # state files.
        self.kernel_class: type | None = None
        self.external_only: bool = False
        # (name, port_type) per declared input.
        self._inputs: list[tuple[str, str]] = []
        # (name, port_type, persistent) per declared output; persistent
        # is None when the definition leaves it to the node class.
        self._outputs: list[tuple[str, str, bool | None]] = []

    # ---- definition ----------------------------------------------------

    def _parse_definition(self, obj: dict):
        self.supports_cancel = bool(obj.get('supportsCancel', False))
        self.supports_complete = bool(obj.get('supportsComplete', False))
        self.external_only = bool(obj.get('externalOnly', False))
        self._inputs = []
        for entry in obj.get('inputs', []) or []:
            name = entry.get('name')
            port_type = entry.get('type')
            if name and port_type:
                self._inputs.append((name, port_type))
        self._outputs = []
        for entry in obj.get('outputs', []) or []:
            name = entry.get('name')
            port_type = entry.get('type')
            # None = not specified in the definition: the host node
            # class's default persistence applies (C++ parity — an
            # explicit key overrides only the persistent bool).
            persistent = entry.get('persistent')
            if persistent is not None:
                persistent = bool(persistent)
            if name and port_type:
                self._outputs.append((name, port_type, persistent))

    def _port_fields(self, obj: dict) -> dict:
        return {'inputs': obj.get('inputs'), 'outputs': obj.get('outputs')}

    def is_transform_shape(self) -> bool:
        return bool(self._inputs)

    # ---- script --------------------------------------------------------

    def set_script(self, script: str):
        # A script replaces a bound kernel class: it is authoritative.
        self.kernel_class = None
        super().set_script(script)

    def load(self, data: dict):
        # Saved content replaces a bound kernel class: the script is
        # authoritative again.
        self.kernel_class = None
        super().load(data)

    def _serialized_script(self) -> str:
        if self.script or self.kernel_class is None:
            return self.script
        # A class-bound kernel must be re-expressed as a script to
        # survive serialization (state files, external execution).
        script = script_from_kernel(self.kernel_class)
        if not script:
            logger.warning(
                "Kernel class '%s' has no retrievable source; the "
                'serialized node will not be executable outside this '
                'process.', self.kernel_class.__name__)
        return script

    # ---- ports ---------------------------------------------------------

    def apply_ports(self, host):
        # A source host takes no inputs. A definition that declares
        # some anyway is ignored here; the factory routes it to a
        # transform.
        if isinstance(host, TransformNode):
            for name, port_type in self._inputs:
                if host.input_port(name) is None:
                    host.add_input(name, port_type)
        for name, port_type, persistent in self._outputs:
            if host.output_port(name) is None:
                host.add_output(name, port_type, persistent=persistent)

    def primary_output_name(self) -> str:
        return self._outputs[0][0] if self._outputs else ''

    # ---- execution -----------------------------------------------------

    def run_transform(self, host, inputs: dict[str, PortData]
                      ) -> dict[str, PortData]:
        return self._run(host, inputs, is_source=False)

    def run_source(self, host) -> dict[str, PortData]:
        return self._run(host, {}, is_source=True)

    def _resolve_kernel_class(self, is_source: bool):
        """The user's kernel class: the bound class of a
        programmatically-hosted kernel, else the single subclass found
        in the script. Returns ``None`` (after logging) when there is no
        usable class."""
        if self.kernel_class is not None:
            return self.kernel_class

        kind = 'SourceKernel' if is_source else 'TransformKernel'
        from tomviz_pipeline._compat import kernel_base_classes
        base_classes = kernel_base_classes('source' if is_source
                                           else 'transform')

        module = self._load_script_module()
        if module is None:
            return None

        try:
            user_class = _find_kernel_class(module, base_classes)
        except ValueError:
            logger.exception(
                'Multiple %s subclasses found in the script of %s',
                kind, self.kernel_name)
            return None
        if user_class is None:
            logger.error(
                'No %s subclass (or legacy tomviz.nodes equivalent) '
                'found in the script of %s', kind, self.kernel_name)
        return user_class

    def _instantiate(self, host, user_class):
        # Kernel.__new__ pre-attaches a Progress(self); _attach_runtime
        # swaps in the runtime's progress reporter when there is one.
        instance = user_class()
        self._attach_runtime(host, instance)
        self._inject_state(host, instance)
        self._inject_parameter_api(instance)
        return instance

    def _run(self, host, inputs: dict[str, PortData],
             is_source: bool) -> dict[str, PortData]:
        user_class = self._resolve_kernel_class(is_source)
        if user_class is None:
            return {}

        instance = self._instantiate(host, user_class)
        kwargs = dict(self.parameters)

        try:
            if is_source:
                result = instance.produce(**kwargs)
            else:
                # Deep-copy each input payload so user mutations don't
                # leak back into upstream port state. Mirrors the C++
                # backend's portDataToPython(deep-copy) behavior.
                inputs_dict = {
                    name: copy.deepcopy(pd.payload)
                    for name, pd in inputs.items()
                }
                result = instance.transform(inputs_dict, **kwargs)
        except Exception:
            logger.exception("Kernel '%s' raised", self.kernel_name)
            return {}
        finally:
            self._harvest_state(host, instance)
            self._harvest_parameter_updates(host, instance)

        # None is the documented signal for "cancel or error" per
        # tomviz_pipeline.kernels — the user's transform/produce
        # returned without producing outputs. Any other non-dict is
        # treated the same way.
        if not isinstance(result, dict):
            return {}

        outputs: dict[str, PortData] = {}
        for name, port_type, _ in self._outputs:
            if name in result:
                # The effective type when the host's port inferred one.
                port = host.output_port(name) if host is not None else None
                outputs[name] = PortData(
                    result[name],
                    port.port_type if port is not None else port_type)
        return outputs

    def run_should_auto_execute(self, host) -> bool:
        """Run the user's ``should_auto_execute`` hook and return its
        answer. Instantiates the kernel as a run does but calls the hook
        instead of ``produce`` / ``transform``. Any error answers
        ``False``; state mutations made by the hook are harvested
        either way."""
        user_class = self._resolve_kernel_class(
            is_source=not self.is_transform_shape())
        if user_class is None:
            return False

        instance = user_class()
        attach_execution_context(instance, ExecutionContext(None, node=host))
        self._inject_state(host, instance)
        self._inject_parameter_api(instance)

        method = getattr(instance, 'should_auto_execute', None)
        if not callable(method):
            # Script written against a kernel base predating the hook.
            return False
        try:
            result = method(**dict(self.parameters))
        except Exception:
            logger.exception("should_auto_execute for '%s' raised",
                             self.kernel_name)
            return False
        finally:
            self._harvest_state(host, instance)
            self._harvest_parameter_updates(host, instance)
        return bool(result)

    @staticmethod
    def _inject_state(host, instance):
        """Hand the host node's persistent state bag to the user
        instance as ``self.state``. The dict object itself is shared,
        so in-place mutations land on the host immediately;
        ``_harvest_state`` covers rebinding (``self.state = {...}``)."""
        state = getattr(host, 'user_state', None)
        if not isinstance(state, dict):
            state = {}
            try:
                host.user_state = state
            except Exception:
                pass
        instance.state = state

    @staticmethod
    def _harvest_state(host, instance):
        """Copy ``self.state`` back onto the host node. A rebound
        non-dict state is rejected with a warning rather than
        clobbering the existing bag."""
        state = getattr(instance, 'state', None)
        if isinstance(state, dict):
            try:
                host.user_state = state
            except Exception:
                pass
        elif state is not None:
            logger.warning(
                'self.state must be a dict; ignoring the %s it was '
                'rebound to', type(state).__name__)

    def _inject_parameter_api(self, instance):
        """Give the user instance what ``self.set_parameter`` /
        ``self.parameter`` need: the declared parameter specs (for
        validation and coercion) and the current values. Nothing is
        shared with the backend — updates are collected afterwards by
        ``_harvest_parameter_updates``."""
        instance._parameter_spec = dict(self._parameter_specs)
        instance._parameter_values = dict(self.parameters)
        instance._parameter_updates = {}

    @staticmethod
    def _harvest_parameter_updates(host, instance):
        """Install the parameter values the kernel changed through
        ``self.set_parameter`` on the host node — quietly (no staleness,
        no ``parameters_applied``), see ``Node.apply_parameter_updates``.
        Runs even when the user method raised, mirroring
        ``_harvest_state``."""
        updates = getattr(instance, '_parameter_updates', None)
        if not isinstance(updates, dict) or not updates:
            return
        apply = getattr(host, 'apply_parameter_updates', None)
        if callable(apply):
            apply(updates)
