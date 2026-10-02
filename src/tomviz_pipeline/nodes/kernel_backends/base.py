###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""What every kernel backend shares: the JSON definition and its common
fields, the script, the parameter values (declared defaults, coercion,
carrying values across a definition edit), serialization, loading the
script as a module, and the progress / cancel wiring of a running
kernel instance.

A backend holds no ports. The scriptable node that owns it calls
``init_ports`` / ``apply_ports`` with itself, and the flavor decides
which ports its definition implies."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import tempfile
import warnings
from typing import Any

from tomviz_pipeline._compat import install_script_module_aliases
from tomviz_pipeline._internal import ExecutionContext, attach_execution_context
from tomviz_pipeline.core import PortData


logger = logging.getLogger('tomviz_pipeline')

SCHEMA_V1 = 1
SCHEMA_V2 = 2


def definition_schema(definition: dict) -> int:
    """The schema a definition is written against: 2 for a kernel class
    script with declared ``inputs`` / ``outputs``, 1 for a legacy
    ``transform(dataset, **params)`` script.

    ``schemaVersion`` decides. A definition without it is v1, except
    that one declaring ``inputs`` or ``outputs`` still reads as v2 with
    a ``DeprecationWarning``: definitions written for ``PythonNode``,
    which only built v2 nodes, did not need the field. From 4.0 a
    definition without ``schemaVersion`` is always v1."""
    return _definition_schema(definition, stacklevel=3)


def _definition_schema(definition: dict, stacklevel: int) -> int:
    """definition_schema(), with the ``warnings.warn`` stacklevel (1 is
    this function) that reaches the code handing over the definition."""
    version = definition.get('schemaVersion')
    if version is not None:
        if version not in (SCHEMA_V1, SCHEMA_V2):
            raise ValueError(f'unsupported schemaVersion {version!r}')
        return version
    if 'inputs' in definition or 'outputs' in definition:
        warnings.warn(
            'A definition that declares inputs or outputs without '
            '"schemaVersion" is read as schema 2; add "schemaVersion": 2. '
            'From tomviz-pipeline 4.0 a definition without '
            '"schemaVersion" is schema 1.',
            DeprecationWarning, stacklevel=stacklevel)
        return SCHEMA_V2
    return SCHEMA_V1


def parse_definition(json_str: str, strict: bool = False) -> dict:
    """The definition object in ``json_str``. An empty string is an
    empty definition. Text that is not a JSON object raises
    ``ValueError`` when ``strict``, and is otherwise logged and read as
    empty."""
    if not json_str:
        return {}
    try:
        obj = json.loads(json_str)
    except ValueError as exc:
        if strict:
            raise ValueError(f'the definition is not valid JSON: {exc}') \
                from exc
        logger.exception('Failed to parse the node definition')
        return {}
    if not isinstance(obj, dict):
        if strict:
            raise ValueError('the definition must be a JSON object')
        logger.error('The node definition is not a JSON object')
        return {}
    return obj


def coerce_parameter_default(param_type: str, default):
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


def enum_option_value(option):
    """The value an enumeration option passes to the kernel: each entry
    in ``options`` is a single-key dict mapping the UI label to it.
    Returns ``None`` for a malformed entry."""
    if not isinstance(option, dict) or not option:
        return None
    return next(iter(option.values()))


def resolve_enum_default(param: dict):
    """For an enumeration parameter, return the value of the option at
    the index given by ``default``, or ``None`` if the definition is
    malformed."""
    options = param.get('options') or []
    default = param.get('default')
    if not isinstance(default, int) or not (0 <= default < len(options)):
        return None
    return enum_option_value(options[default])


def merge_parameter_values(previous_values: dict, previous_types: dict,
                           new_defaults: dict, new_types: dict,
                           new_enum_options: dict,
                           reset_names: list | None = None) -> dict:
    """Carry previously set parameter values onto a freshly parsed
    default set. A value survives only when its name is still declared
    and its declared type is unchanged; an enumeration value must also
    still be one of the declared options. Everything else keeps the new
    default and its name is appended to ``reset_names``. Mirrors the
    C++ ``mergeParameterValues``."""
    merged = dict(new_defaults)
    for name, param_type in new_types.items():
        if name not in previous_values:
            continue  # A new parameter: the declared default stands.
        previous = previous_values[name]
        if previous_types.get(name) != param_type:
            if reset_names is not None:
                reset_names.append(name)
            continue
        if param_type == 'enumeration':
            offered = (enum_option_value(option)
                       for option in new_enum_options.get(name, []))
            if previous not in offered:
                if reset_names is not None:
                    reset_names.append(name)
                continue
        merged[name] = previous
    return merged


class BaseKernelBackend:
    """Owns a scriptable node's definition, script and parameter values,
    and runs its kernel. Subclasses add the schema-specific parts."""

    #: The definition schema this backend runs (see definition_schema).
    schema_version: int = 0

    def __init__(self):
        self.json_description: str = ''
        self.script: str = ''
        self.kernel_name: str = ''
        self.default_label: str = ''
        self.description_text: str = ''
        self.help_text: str = ''
        self.custom_widget_id: str = ''
        self.supports_cancel: bool = False
        self.supports_complete: bool = False
        self.external_python_env_path: str = ''
        self.parameters: dict[str, Any] = {}
        self._parameter_types: dict[str, str] = {}
        self._enum_options: dict[str, list] = {}
        # name -> the definition's parameter entry, verbatim. Handed to
        # v2 kernels so self.set_parameter() can validate against it.
        self._parameter_specs: dict[str, dict] = {}

    # ---- definition ----------------------------------------------------

    def set_json_description(self, json_str: str):
        """Install a definition; every parameter takes its declared
        default."""
        self._apply_description(json_str, parse_definition(json_str),
                                keep_values=False)

    def reconfigure(self, json_str: str) -> list[str]:
        """Replace the definition of an existing node. Parameter values
        carry over where the definition still allows them (see
        merge_parameter_values); the names that went back to their
        defaults are returned. Raises ``ValueError`` when the
        definition is not a JSON object, empties a definition, changes
        the schema, or changes anything that implies the node's ports,
        which are fixed once the node exists."""
        candidate = parse_definition(json_str, strict=True)
        current = parse_definition(self.json_description)
        if not candidate and current:
            raise ValueError('the definition cannot be emptied')
        # Warns at the caller of ScriptableNode.reconfigure_description.
        if _definition_schema(candidate, stacklevel=4) != \
                self.schema_version:
            raise ValueError(
                'the schema cannot change on an existing node: it picks '
                'the node class when the node is created')
        frozen = self._port_fields(current)
        changed = [key for key, value in self._port_fields(candidate).items()
                   if frozen.get(key) != value]
        if changed:
            raise ValueError(
                f'{", ".join(changed)} cannot change on an existing node: '
                "a node's ports are fixed once it exists")
        return self._apply_description(json_str, candidate, keep_values=True)

    def _apply_description(self, json_str: str, obj: dict,
                           keep_values: bool) -> list[str]:
        self.json_description = json_str
        self.kernel_name = obj.get('name', '')
        self.default_label = obj.get('label', '')
        self.description_text = obj.get('description', '')
        self.help_text = obj.get('help', '')
        self.custom_widget_id = obj.get('widget', '')
        self.external_python_env_path = obj.get('tomviz_pipeline_env', '')
        self._parse_definition(obj)

        defaults, types, specs, enum_options = {}, {}, {}, {}
        for param in obj.get('parameters', []) or []:
            name = param.get('name')
            param_type = param.get('type', '')
            if not name or self._claims_parameter(name, param_type):
                continue
            types[name] = param_type
            specs[name] = dict(param)
            if param_type == 'enumeration':
                enum_options[name] = param.get('options') or []
                resolved = resolve_enum_default(param)
                if resolved is not None:
                    defaults[name] = resolved
                    continue
            if 'default' not in param:
                # No default: the kernel's own Python default applies.
                continue
            defaults[name] = coerce_parameter_default(
                param_type, param['default'])

        reset: list[str] = []
        if keep_values:
            defaults = merge_parameter_values(
                self.parameters, self._parameter_types, defaults, types,
                enum_options, reset)
        self._parameter_types = types
        self._parameter_specs = specs
        self._enum_options = enum_options
        # One assignment, so a run snapshotting the values on a worker
        # thread never sees a half-built set.
        self.parameters = defaults
        return reset

    def declared_parameters(self) -> dict[str, str]:
        """Name -> declared type of every parameter the definition
        declares as a value (not the ones it turns into ports)."""
        return dict(self._parameter_types)

    def _parse_definition(self, obj: dict):
        """Read the schema-specific fields of a definition. Runs before
        the parameters are parsed."""

    def _claims_parameter(self, name: str, param_type: str) -> bool:
        """True when a declared parameter is not a value (v1 turns
        ``dataset`` parameters into input ports)."""
        return False

    def _port_fields(self, obj: dict) -> dict:
        """The definition fields that decide the node's ports."""
        return {}

    # ---- script --------------------------------------------------------

    def set_script(self, script: str):
        self.script = script

    def _serialized_script(self) -> str:
        return self.script

    # ---- ports ---------------------------------------------------------

    def init_ports(self, host):
        """Create the ports every node of this flavor has before it
        reads a definition."""

    def apply_ports(self, host):
        """Create the ports the definition implies. Ports that already
        exist are left alone, so this is safe to repeat."""

    def primary_output_name(self) -> str:
        return ''

    # ---- serialize / deserialize --------------------------------------

    def serialize_into(self, base: dict) -> dict:
        base['description'] = self.json_description
        base['script'] = self._serialized_script()
        if self.parameters:
            base['arguments'] = dict(self.parameters)
        return base

    def load(self, data: dict):
        """Read the definition, script and parameter values of a saved
        node entry. The values go in last: the definition resets them
        to their defaults."""
        if 'description' in data:
            self.set_json_description(data['description'])
        if 'script' in data:
            self.set_script(data.get('script') or '')
        for key, value in (data.get('arguments') or {}).items():
            self.parameters[key] = value

    # ---- execution -----------------------------------------------------

    def run_transform(self, host, inputs: dict[str, PortData]
                      ) -> dict[str, PortData]:
        raise NotImplementedError

    def run_should_auto_execute(self, host) -> bool:
        return False

    def _attach_runtime(self, host, instance):
        """Point a kernel instance's progress, cancel and complete
        plumbing at its host node. With a runtime progress reporter
        installed, ``self.progress.value = X`` lands in it directly;
        the execution context backs ``self.canceled`` /
        ``self.completed``, OR-ing the parent-process control channel
        with the node's in-process flags."""
        progress = getattr(host, 'progress', None)
        channel = None
        if progress is not None:
            instance.progress = progress
            channel = (progress.control_channel()
                       if hasattr(progress, 'control_channel') else None)
        attach_execution_context(instance,
                                 ExecutionContext(channel, node=host))
        if hasattr(progress, 'set_primary_port'):
            progress.set_primary_port(self.primary_output_name())

    def _load_script_module(self):
        """Materialize the script as a temp .py file and import it, so
        relative imports and decorators behave as in a file. Returns
        the module, or ``None`` on failure."""
        # Scripts import `tomviz.operators` / `tomviz.nodes` and may
        # reference `tomviz.utils.*` without importing it; make sure the
        # aliases resolve before exec.
        install_script_module_aliases()
        fd = None
        path = None
        try:
            fd, path = tempfile.mkstemp(suffix='.py', text=True)
            os.write(fd, self.script.encode())
            os.close(fd)
            fd = None
            module_name = (self.kernel_name or self.default_label
                           or 'ScriptModule')
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception:
            logger.exception('Failed to load the script of %s',
                             self.kernel_name)
            return None
        finally:
            if fd is not None:
                os.close(fd)
            if path is not None:
                try:
                    os.unlink(path)
                except OSError:
                    pass
