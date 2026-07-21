###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""LegacyPythonTransform — runs a JSON-described Python operator (the
~170 .py/.json operator pairs in the tomviz repo) inside the graph
runtime.

Mirrors the C++ LegacyPythonTransform's serialized form: the saved JSON
carries `description` (the operator's JSON schema as a string), `script`
(the operator's Python source), and `arguments` (current parameter
values). Output ports are derived from the operator description's
`results` and `children` entries the same way the C++ side does it."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import tempfile

from tomviz_pipeline._compat import install_script_module_aliases
from tomviz_pipeline._internal import (
    OperatorWrapper,
    add_transform_decorators,
    find_transform_function,
)
from tomviz_pipeline.core import PortData, TransformNode
from tomviz_pipeline.dataset import Dataset, LegacyDataset


def _upgrade_to_legacy(payload):
    """Upgrade a base Dataset payload to a LegacyDataset so v1
    operators get the create_child_dataset API. Non-Dataset payloads
    pass through unchanged."""
    if isinstance(payload, Dataset) and not isinstance(payload,
                                                       LegacyDataset):
        return LegacyDataset.from_dataset(payload)
    return payload


logger = logging.getLogger('tomviz_pipeline')


# Operator-description parameter types that materialize as additional
# input ports rather than scalar parameters (the linked dataset gets
# substituted into the kwargs at execute time).
_DATASET_PARAM_TYPES = ('dataset',)


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
    malformed. Each entry in ``options`` is a single-key dict where the
    key is the UI label and the value is what gets passed to the
    operator."""
    options = param.get('options') or []
    default = param.get('default')
    if not isinstance(default, int) or not (0 <= default < len(options)):
        return None
    opt = options[default]
    if not isinstance(opt, dict) or not opt:
        return None
    return next(iter(opt.values()))


class LegacyPythonTransform(TransformNode):
    type_name = 'transform.legacyPython'

    def __init__(self):
        super().__init__()
        self.add_input('volume', 'ImageData')
        self.add_output('volume', 'ImageData', persistent=True)
        self.label = 'Python Operator'

        self._json_description: str = ''
        self._script: str = ''
        self._operator_name: str = ''
        self._external_env_path: str = ''
        self._parameters: dict = {}
        self._dataset_input_names: list[str] = []
        self._result_names: list[str] = []
        self._result_types: list[str] = []
        self._primary_output_name: str = 'volume'
        self._child_name: str = ''

    # ---- serialization -------------------------------------------------

    def serialize(self) -> dict:
        # Mirrors C++ LegacyPythonTransform::serialize: the operator's
        # JSON description (as a string), its Python source, and the
        # current parameter values under `arguments`.
        data = super().serialize()
        if self._json_description:
            data['description'] = self._json_description
        if self._script:
            data['script'] = self._script
        if self._parameters:
            data['arguments'] = dict(self._parameters)
        return data

    def deserialize(self, data: dict) -> bool:
        # Order matters: parse the description first (it adds ports),
        # then the script, then defer to the base class so it can apply
        # outputPort metadata onto the ports we just created.
        if 'description' in data:
            self._set_json_description(data['description'])
        if 'script' in data:
            self._script = data.get('script', '')
        if not super().deserialize(data):
            return False
        for key, value in (data.get('arguments') or {}).items():
            self._parameters[key] = value
        # Pre-extraction operator JSONs could request a specific Python
        # environment via `tomviz_pipeline_env`; honor it through the
        # modern per-node executor mechanism.
        from tomviz_pipeline.external import promote_description_env_path
        promote_description_env_path(self, data, self._external_env_path)
        return True

    def _set_json_description(self, json_desc: str):
        self._json_description = json_desc
        try:
            obj = json.loads(json_desc) if json_desc else {}
        except json.JSONDecodeError:
            logger.exception('Failed to parse operator JSON description')
            obj = {}
        self._operator_name = obj.get('name', '')
        self._external_env_path = obj.get('tomviz_pipeline_env', '')
        if 'label' in obj:
            self.label = obj['label']

        self._apply_parameters(obj.get('parameters', []))
        self._apply_results(obj.get('results', []))
        self._apply_children(obj.get('children', []))
        self._apply_type_overrides(obj)

    def _apply_parameters(self, params):
        self._parameters = {}
        self._dataset_input_names = []
        for param in params:
            name = param.get('name')
            ptype = param.get('type', '')
            if not name:
                continue
            if ptype in _DATASET_PARAM_TYPES:
                if self.input_port(name) is None:
                    self.add_input(name, 'ImageData')
                self._dataset_input_names.append(name)
                continue
            if 'default' not in param:
                # Unset complex types — operator's own Python default wins.
                continue
            if ptype == 'enumeration':
                # The description's `default` is an index into `options`;
                # the operator expects the option's value. Resolve here
                # so _parameters always holds the value, not the index.
                resolved = _resolve_enum_default(param)
                if resolved is not None:
                    self._parameters[name] = resolved
                    continue
            self._parameters[name] = _coerce_param_default(
                ptype, param['default'])

    def _apply_results(self, results):
        self._result_names = []
        self._result_types = []
        for result in results:
            rname = result.get('name')
            rtype = result.get('type', '')
            if not rname:
                continue
            self._result_names.append(rname)
            self._result_types.append(rtype)
            if rtype in ('table', 'molecule') and \
                    self.output_port(rname) is None:
                port_type = 'Table' if rtype == 'table' else 'Molecule'
                self.add_output(rname, port_type, persistent=True)

    def _apply_children(self, children):
        if not children:
            return
        child_name = children[0].get('name', '')
        if not child_name:
            return
        self._child_name = child_name
        primary = self.output_port(self._primary_output_name)
        if primary is not None:
            primary.name = child_name
            primary.persistent = True
            self._primary_output_name = child_name

    def _apply_type_overrides(self, obj):
        if 'inputType' in obj:
            in_port = self.input_port('volume')
            if in_port is not None:
                in_port.accepted_types = [obj['inputType']]
                in_port.port_type = obj['inputType']
        if 'outputType' in obj:
            out_port = self.output_port(self._primary_output_name)
            if out_port is not None:
                out_port.port_type = obj['outputType']

    # ---- execution -----------------------------------------------------

    def transform(self, inputs: dict[str, PortData]) -> dict[str, PortData]:
        primary = inputs.get('volume')
        if primary is None:
            return {}

        import copy as _copy
        # Operators mutate the dataset they're handed — give them a deep
        # copy so upstream ports keep their original payload. Upgrade
        # to LegacyDataset so v1 operators have create_child_dataset.
        dataset = _upgrade_to_legacy(_copy.deepcopy(primary.payload))

        transform_fn = self._resolve_transform_fn()
        if transform_fn is None:
            return {}

        kwargs = self._build_kwargs(inputs)
        try:
            result = transform_fn(dataset, **kwargs)
        except Exception:
            logger.exception("Operator '%s' raised", self._operator_name)
            return {}

        return self._collect_outputs(dataset, result)

    def _resolve_transform_fn(self):
        operator_module = self._load_operator_module()
        try:
            transform_fn = find_transform_function(operator_module)
        except Exception:
            logger.exception(
                'Could not locate transform function in operator %s',
                self._operator_name)
            return None

        operator_dict = {'description': self._json_description}
        if self.label:
            operator_dict['label'] = self.label
        if self._script:
            operator_dict['script'] = self._script
        if self._parameters:
            operator_dict['arguments'] = self._parameters

        transform_fn = add_transform_decorators(transform_fn, operator_dict)

        # Wire progress + cancel/complete onto the operator instance.
        # `self.progress.value = ...` routes to the runtime's progress
        # reporter; `self.canceled` / `self.completed` lazily check
        # the parent → child control channel exposed by the same
        # progress object (None for tqdm; socket/files otherwise) OR'd
        # with this node's in-process cancel / complete flags so
        # ThreadedExecutor.cancel() reaches polling operators too.
        if hasattr(transform_fn, '__self__'):
            transform_fn.__self__.progress = self.progress
            channel = (self.progress.control_channel()
                       if self.progress is not None else None)
            transform_fn.__self__._operator_wrapper = OperatorWrapper(
                channel, node=self)
            # Tell the progress reporter which port a bare `progress.data
            # = X` should populate. Multi-port operators that pass a
            # dict bypass this.
            if hasattr(self.progress, 'set_primary_port'):
                self.progress.set_primary_port(self._primary_output_name)

        return transform_fn

    def _build_kwargs(self, inputs):
        kwargs = dict(self._parameters)
        for dsname in self._dataset_input_names:
            ds_input = inputs.get(dsname)
            if ds_input is not None:
                # v1 ops may call create_child_dataset on these too,
                # so the same upgrade applies.
                kwargs[dsname] = _upgrade_to_legacy(ds_input.payload)
        return kwargs

    def _collect_outputs(self, dataset, result):
        outputs: dict[str, PortData] = {}

        # If a child was declared and the operator returned a dict
        # containing it, that becomes the primary output. Otherwise the
        # mutated `dataset` itself is the primary output.
        primary_payload = dataset
        is_dict_result = isinstance(result, dict)
        if (self._child_name and is_dict_result and
                self._child_name in result):
            primary_payload = result[self._child_name]

        primary_port = self.output_port(self._primary_output_name)
        primary_port_type = (primary_port.port_type
                             if primary_port is not None else 'ImageData')
        outputs[self._primary_output_name] = PortData(
            primary_payload, primary_port_type)

        if is_dict_result:
            for rname, _ in zip(self._result_names, self._result_types):
                if rname == self._child_name or rname not in result:
                    continue
                port = self.output_port(rname)
                ptype = port.port_type if port is not None else 'ImageData'
                outputs[rname] = PortData(result[rname], ptype)

        return outputs

    def _load_operator_module(self):
        # The operator is described entirely by `script`. Materialize the
        # source as a temp .py file and import it the way the legacy CLI
        # did so the operator's relative imports / decorators continue
        # to work.
        # Operator scripts do `import tomviz.operators` (and may
        # reference tomviz.utils.* without importing it) — make sure the
        # aliases resolve before exec.
        install_script_module_aliases()
        fd = None
        path = None
        try:
            fd, path = tempfile.mkstemp(suffix='.py', text=True)
            os.write(fd, self._script.encode())
            os.close(fd)
            fd = None
            label = self._operator_name or self.label or 'OperatorModule'
            spec = importlib.util.spec_from_file_location(label, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        finally:
            if fd is not None:
                os.close(fd)
            if path is not None:
                try:
                    os.unlink(path)
                except OSError:
                    pass
