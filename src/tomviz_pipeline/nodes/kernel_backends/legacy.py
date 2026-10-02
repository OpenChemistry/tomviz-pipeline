###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""The legacy (v1) kernel backend: runs the JSON-described
``transform(dataset, **params)`` scripts of the tomviz kernel catalog,
written either as a module-level function or as a ``tomviz.operators``
``Operator`` subclass. Mirrors the C++ ``LegacyPythonTransform``.

A legacy node always has a ``volume`` input and output. The definition
adds the rest: an input port per ``dataset`` parameter, a Table or
Molecule output per such ``results`` entry, a ``children`` entry renames
the primary output, and ``inputType`` / ``outputType`` override the
``volume`` port types."""

from __future__ import annotations

import copy
import logging

from tomviz_pipeline._internal import (
    add_transform_decorators,
    find_transform_function,
)
from tomviz_pipeline.core import PortData
from tomviz_pipeline.dataset import Dataset, LegacyDataset

from .base import SCHEMA_V1, BaseKernelBackend


logger = logging.getLogger('tomviz_pipeline')

# Parameter types that become input ports rather than values (the
# linked dataset is substituted into the kwargs at execute time).
_DATASET_PARAM_TYPES = ('dataset',)

_RESULT_PORT_TYPES = {'table': 'Table', 'molecule': 'Molecule'}


def _upgrade_to_legacy(payload):
    """Upgrade a base Dataset payload to a LegacyDataset so v1 kernels
    get the create_child_dataset API. Non-Dataset payloads pass through
    unchanged."""
    if isinstance(payload, Dataset) and not isinstance(payload,
                                                       LegacyDataset):
        return LegacyDataset.from_dataset(payload)
    return payload


class LegacyKernelBackend(BaseKernelBackend):
    """Runs a v1 ``transform(dataset, **params)`` script."""

    schema_version = SCHEMA_V1

    def __init__(self):
        super().__init__()
        self._dataset_input_names: list[str] = []
        self._result_names: list[str] = []
        self._result_types: list[str] = []
        self._child_name: str = ''
        self._input_type: str = ''
        self._output_type: str = ''
        # Tracks the host's primary output port, renamed by `children`.
        self._primary_output_name: str = 'volume'

    # ---- definition ----------------------------------------------------

    def _parse_definition(self, obj: dict):
        self._dataset_input_names = []
        self._result_names = []
        self._result_types = []
        for result in obj.get('results', []) or []:
            name = result.get('name')
            if not name:
                continue
            self._result_names.append(name)
            self._result_types.append(result.get('type', ''))
        children = obj.get('children') or []
        self._child_name = children[0].get('name', '') if children else ''
        self._input_type = obj.get('inputType', '')
        self._output_type = obj.get('outputType', '')

    def _claims_parameter(self, name: str, param_type: str) -> bool:
        if param_type in _DATASET_PARAM_TYPES:
            self._dataset_input_names.append(name)
            return True
        return False

    def _port_fields(self, obj: dict) -> dict:
        datasets = [param.get('name')
                    for param in obj.get('parameters', []) or []
                    if param.get('type') in _DATASET_PARAM_TYPES]
        fields = {key: obj.get(key) for key in
                  ('results', 'children', 'inputType', 'outputType')}
        fields['dataset parameters'] = datasets
        return fields

    # ---- script --------------------------------------------------------

    def set_script(self, script: str):
        super().set_script(script)
        # Cancel / complete support follows the base class the script
        # derives from (C++ parity).
        self.supports_complete = 'CompletableOperator' in script
        self.supports_cancel = (self.supports_complete
                                or 'CancelableOperator' in script)

    # ---- ports ---------------------------------------------------------

    def init_ports(self, host):
        host.add_input('volume', 'ImageData')
        host.add_output('volume', 'ImageData', persistent=True)

    def apply_ports(self, host):
        for name in self._dataset_input_names:
            if host.input_port(name) is None:
                host.add_input(name, 'ImageData')
        for name, result_type in zip(self._result_names,
                                     self._result_types):
            port_type = _RESULT_PORT_TYPES.get(result_type)
            if port_type and host.output_port(name) is None:
                host.add_output(name, port_type, persistent=True)
        if self._child_name:
            primary = host.output_port(self._primary_output_name)
            if primary is not None:
                primary.name = self._child_name
                primary.persistent = True
                self._primary_output_name = self._child_name
        if self._input_type:
            in_port = host.input_port('volume')
            if in_port is not None:
                in_port.accepted_types = [self._input_type]
                in_port.port_type = self._input_type
        if self._output_type:
            out_port = host.output_port(self._primary_output_name)
            if out_port is not None:
                out_port.port_type = self._output_type

    def primary_output_name(self) -> str:
        return self._primary_output_name

    # ---- execution -----------------------------------------------------

    def run_transform(self, host, inputs: dict[str, PortData]
                      ) -> dict[str, PortData]:
        primary = inputs.get('volume')
        if primary is None:
            return {}

        # Kernels mutate the dataset they're handed — give them a deep
        # copy so upstream ports keep their original payload. Upgrade
        # to LegacyDataset so v1 kernels have create_child_dataset.
        dataset = _upgrade_to_legacy(copy.deepcopy(primary.payload))

        transform_fn = self._resolve_transform_function(host)
        if transform_fn is None:
            return {}

        kwargs = dict(self.parameters)
        for name in self._dataset_input_names:
            port_data = inputs.get(name)
            if port_data is not None:
                # v1 kernels may call create_child_dataset on these too,
                # so the same upgrade applies.
                kwargs[name] = _upgrade_to_legacy(port_data.payload)

        try:
            result = transform_fn(dataset, **kwargs)
        except Exception:
            logger.exception("Kernel '%s' raised", self.kernel_name)
            return {}

        return self._collect_outputs(host, dataset, result)

    def _resolve_transform_function(self, host):
        module = self._load_script_module()
        if module is None:
            return None
        try:
            transform_fn = find_transform_function(module)
        except Exception:
            logger.exception(
                'Could not locate the transform function of %s',
                self.kernel_name)
            return None

        # The decorators read the node entry as a state file holds it.
        entry = {'description': self.json_description}
        if host.label:
            entry['label'] = host.label
        if self.script:
            entry['script'] = self.script
        if self.parameters:
            entry['arguments'] = self.parameters
        transform_fn = add_transform_decorators(transform_fn, entry)

        # An Operator subclass runs as a bound method: wire its progress,
        # cancel and complete plumbing to the host. Module-level
        # functions have no instance to wire.
        instance = getattr(transform_fn, '__self__', None)
        if instance is not None:
            self._attach_runtime(host, instance)
        return transform_fn

    def _collect_outputs(self, host, dataset, result):
        outputs: dict[str, PortData] = {}

        # If a child was declared and the kernel returned a dict
        # containing it, that becomes the primary output. Otherwise the
        # mutated `dataset` itself is the primary output.
        primary_payload = dataset
        is_dict_result = isinstance(result, dict)
        if (self._child_name and is_dict_result
                and self._child_name in result):
            primary_payload = result[self._child_name]

        primary_port = host.output_port(self._primary_output_name)
        primary_port_type = (primary_port.port_type
                             if primary_port is not None else 'ImageData')
        outputs[self._primary_output_name] = PortData(
            primary_payload, primary_port_type)

        if is_dict_result:
            for name in self._result_names:
                if name == self._child_name or name not in result:
                    continue
                port = host.output_port(name)
                port_type = (port.port_type if port is not None
                             else 'ImageData')
                outputs[name] = PortData(result[name], port_type)

        return outputs
