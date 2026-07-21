###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Shared backend for schema-v2 Python source / transform nodes. Mirrors
the C++ ``tomviz::pipeline::PythonNodeBackend``: owns the JSON
description, the script source, and the current parameter values; runs
the user's kernel class — a ``tomviz_pipeline.kernels.SourceKernel`` /
``TransformKernel`` subclass (or its legacy ``tomviz.nodes`` spelling) —
when the host's shell calls ``run_source`` / ``run_transform``.

The shells (``PythonSource`` in ``sources/python_source.py`` and
``PythonTransform`` in ``transforms/python_transform.py``) are thin —
they own one of these and forward."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import logging
import os
import tempfile
from typing import Any, Callable

from tomviz_pipeline._compat import install_script_module_aliases
from tomviz_pipeline._internal import OperatorWrapper
from tomviz_pipeline.core import PortData


logger = logging.getLogger('tomviz_pipeline')


def _coerce_param_default(param_type: str, default):
    if isinstance(default, list):
        return list(default)
    if param_type == 'double':
        return float(default) if default is not None else 0.0
    if param_type in ('int', 'integer', 'enumeration'):
        return int(default) if default is not None else 0
    if param_type in ('bool', 'boolean'):
        return bool(default)
    if param_type in ('string', 'file', 'save_file', 'directory'):
        return str(default) if default is not None else ''
    return default


def _resolve_enum_default(param: dict):
    """For an enumeration parameter, return the value of the option at
    the index given by ``default``, or ``None`` if the description is
    malformed."""
    options = param.get('options') or []
    default = param.get('default')
    if not isinstance(default, int) or not (0 <= default < len(options)):
        return None
    opt = options[default]
    if not isinstance(opt, dict) or not opt:
        return None
    return next(iter(opt.values()))


def _find_node_class(module, base_classes):
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
                'Multiple node classes defined in module — only one '
                'node class can be defined per script.')
        found = cls
    return found


class PythonNodeBackend:
    """Owns the schema-v2 Python state and runs the user's class."""

    def __init__(self):
        self.json_description: str = ''
        self.script: str = ''
        # When set (see nodes.hosting.node_from_kernel), execution uses
        # this class directly instead of loading the script — the
        # Python-only fast path for programmatically-hosted kernels.
        # The script/description still round-trip through state files.
        self.kernel_class: type = None
        self.operator_name: str = ''
        self.default_label: str = ''
        self.description_text: str = ''
        self.help_text: str = ''
        self.custom_widget_id: str = ''
        self.supports_cancel: bool = False
        self.supports_complete: bool = False
        self.external_python_env_path: str = ''
        self.external_only: bool = False
        # List of (name, port_type) for inputs.
        self._inputs: list[tuple[str, str]] = []
        # List of (name, port_type, persistent) for outputs.
        self._outputs: list[tuple[str, str, bool]] = []
        self.parameters: dict[str, Any] = {}
        self._parameter_types: dict[str, str] = {}
        self._enum_options: dict[str, list] = {}

    # ---- accessors ----------------------------------------------------

    def primary_output_name(self) -> str:
        return self._outputs[0][0] if self._outputs else ''

    def is_transform_shape(self) -> bool:
        return bool(self._inputs)

    # ---- description / script -----------------------------------------

    def set_json_description(self, json_str: str):
        self.json_description = json_str
        self._parse_description()

    def set_script(self, script: str):
        self.script = script

    def _parse_description(self):
        self.operator_name = ''
        self.default_label = ''
        self.description_text = ''
        self.help_text = ''
        self.custom_widget_id = ''
        self.supports_cancel = False
        self.supports_complete = False
        self.external_python_env_path = ''
        self.external_only = False
        self._inputs = []
        self._outputs = []
        self.parameters = {}
        self._parameter_types = {}
        self._enum_options = {}

        if not self.json_description:
            return
        try:
            obj = json.loads(self.json_description)
        except json.JSONDecodeError:
            logger.exception('Failed to parse operator JSON description')
            return
        if not isinstance(obj, dict):
            return

        self.operator_name = obj.get('name', '')
        self.default_label = obj.get('label', '')
        self.description_text = obj.get('description', '')
        self.help_text = obj.get('help', '')
        self.custom_widget_id = obj.get('widget', '')
        self.supports_cancel = bool(obj.get('supportsCancel', False))
        self.supports_complete = bool(obj.get('supportsComplete', False))
        self.external_python_env_path = obj.get('tomviz_pipeline_env', '')
        self.external_only = bool(obj.get('externalOnly', False))

        for entry in obj.get('inputs', []) or []:
            name = entry.get('name')
            ptype = entry.get('type')
            if name and ptype:
                self._inputs.append((name, ptype))
        for entry in obj.get('outputs', []) or []:
            name = entry.get('name')
            ptype = entry.get('type')
            # None = not specified in the description: the host node
            # class's default persistence applies (C++ parity — an
            # explicit key overrides only the persistent bool).
            persistent = entry.get('persistent')
            if persistent is not None:
                persistent = bool(persistent)
            if name and ptype:
                self._outputs.append((name, ptype, persistent))

        for param in obj.get('parameters', []) or []:
            name = param.get('name')
            ptype = param.get('type', '')
            if not name:
                continue
            self._parameter_types[name] = ptype
            if ptype == 'enumeration':
                options = param.get('options') or []
                self._enum_options[name] = options
                resolved = _resolve_enum_default(param)
                if resolved is not None:
                    self.parameters[name] = resolved
                    continue
            if 'default' not in param:
                continue
            self.parameters[name] = _coerce_param_default(
                ptype, param['default'])

    # ---- port creation -------------------------------------------------

    def apply_description(
            self,
            add_input: Callable[[str, str], None] | None,
            add_output: Callable[[str, str, 'bool | None'], None] | None):
        if add_input:
            for name, ptype in self._inputs:
                add_input(name, ptype)
        if add_output:
            for name, ptype, persistent in self._outputs:
                add_output(name, ptype, persistent)

    # ---- serialize / deserialize --------------------------------------

    def serialize_into(self, base: dict) -> dict:
        base['description'] = self.json_description
        script = self.script
        if not script and self.kernel_class is not None:
            # A class-bound kernel must be re-expressed as a script to
            # survive serialization (state files, external execution).
            from tomviz_pipeline.nodes.python_node import script_from_kernel
            script = script_from_kernel(self.kernel_class)
            if not script:
                logger.warning(
                    "Kernel class '%s' has no retrievable source; the "
                    'serialized node will not be executable outside '
                    'this process.', self.kernel_class.__name__)
        base['script'] = script
        if self.parameters:
            base['arguments'] = dict(self.parameters)
        return base

    def apply_serialized_fields(
            self, data: dict,
            add_input: Callable[[str, str], None] | None,
            add_output: Callable[[str, str, bool], None] | None):
        # Deserialized content replaces any programmatically-hosted
        # kernel class; the script is authoritative again.
        self.kernel_class = None
        if 'description' in data:
            self.set_json_description(data['description'])
        if 'script' in data:
            self.script = data.get('script', '')
        self.apply_description(add_input, add_output)
        for key, value in (data.get('arguments') or {}).items():
            self.parameters[key] = value

    # ---- execution ----------------------------------------------------

    def run_transform(self, host, inputs: dict[str, PortData]
                      ) -> dict[str, PortData]:
        return self._run_impl(host, inputs, is_source=False)

    def run_source(self, host) -> dict[str, PortData]:
        return self._run_impl(host, {}, is_source=True)

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
            user_class = _find_node_class(module, base_classes)
        except ValueError:
            logger.exception(
                'Multiple %s subclasses found in script for %s',
                kind, self.operator_name)
            return None
        if user_class is None:
            logger.error(
                'No %s subclass (or legacy tomviz.nodes equivalent) '
                'found in script for %s', kind, self.operator_name)
        return user_class

    def _run_impl(self, host, inputs: dict[str, PortData],
                  is_source: bool) -> dict[str, PortData]:
        user_class = self._resolve_kernel_class(is_source)
        if user_class is None:
            return {}

        # Instantiate; tomviz_pipeline.kernels.Kernel.__new__ pre-attaches
        # a tomviz_pipeline.operators.Progress(self), which we then
        # overwrite with the runtime's progress object so
        # `self.progress.value = X` lands in the parent-bound progress
        # channel directly (bypassing the _operator_wrapper indirection
        # that's unused on this side). The wrapper still backs
        # `self.canceled` / `self.completed`: it ORs the parent-process
        # control channel with the host node's in-process flags.
        instance = user_class()
        progress = getattr(host, 'progress', None)
        if progress is not None:
            instance.progress = progress
            channel = (progress.control_channel()
                       if hasattr(progress, 'control_channel') else None)
            instance._operator_wrapper = OperatorWrapper(channel, node=host)
            if hasattr(progress, 'set_primary_port'):
                progress.set_primary_port(self.primary_output_name())
        else:
            instance._operator_wrapper = OperatorWrapper(None, node=host)

        self._inject_state(host, instance)

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
            logger.exception("Operator '%s' raised", self.operator_name)
            return {}
        finally:
            self._harvest_state(host, instance)

        # None is the documented signal for "cancel or error" per
        # tomviz_pipeline.kernels — the user's transform/produce
        # returned without producing outputs. Any other non-dict is
        # treated the same way.
        if not isinstance(result, dict):
            return {}

        outputs: dict[str, PortData] = {}
        for name, ptype, _ in self._outputs:
            if name in result:
                outputs[name] = PortData(result[name], ptype)
        return outputs

    def run_should_auto_execute(self, host) -> bool:
        """Run the user's ``should_auto_execute`` hook and return its
        answer. Mirrors ``_run_impl``'s instantiation (wrapper +
        ``self.state`` injection) but calls the hook instead of
        ``produce``/``transform``. Any error answers ``False``; state
        mutations made by the hook are harvested either way."""
        user_class = self._resolve_kernel_class(
            is_source=not self.is_transform_shape())
        if user_class is None:
            return False

        instance = user_class()
        instance._operator_wrapper = OperatorWrapper(None, node=host)
        self._inject_state(host, instance)

        method = getattr(instance, 'should_auto_execute', None)
        if not callable(method):
            # Script written against a kernel base predating the hook.
            return False
        try:
            result = method(**dict(self.parameters))
        except Exception:
            logger.exception("should_auto_execute for '%s' raised",
                             self.operator_name)
            return False
        finally:
            self._harvest_state(host, instance)
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

    def _load_script_module(self):
        """Materialize self.script as a temp .py and import it. Returns
        the module, or ``None`` on failure."""
        # The script may contain `import tomviz.nodes` / reference
        # `tomviz.utils.*` — make sure the aliases resolve before exec.
        install_script_module_aliases()
        fd = None
        path = None
        try:
            fd, path = tempfile.mkstemp(suffix='.py', text=True)
            os.write(fd, self.script.encode())
            os.close(fd)
            fd = None
            label = (self.operator_name
                     or self.default_label
                     or 'NodeModule')
            spec = importlib.util.spec_from_file_location(label, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception:
            logger.exception(
                'Failed to load script for %s', self.operator_name)
            return None
        finally:
            if fd is not None:
                os.close(fd)
            if path is not None:
                try:
                    os.unlink(path)
                except OSError:
                    pass
