###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""PythonTransform — schema-v2 Python transform node for the headless
runtime.

Mirrors the C++ ``tomviz::pipeline::PythonTransform``: a thin shell that
owns a :class:`PythonNodeBackend` and delegates all parsing /
serialization / execution to it. The user-facing class is a subclass of
:class:`tomviz_pipeline.kernels.TransformKernel` (or its legacy
``tomviz.nodes.TransformNode`` spelling) defined in the operator
script."""

from __future__ import annotations

from tomviz_pipeline.core import PortData, TransformNode
from tomviz_pipeline.nodes.python_backend import PythonNodeBackend
from tomviz_pipeline.nodes.python_node import PythonNode


class PythonTransform(PythonNode, TransformNode):
    type_name = 'transform.python'

    def __init__(self, *args, **kwargs):
        # Facade path: PythonNode.__new__ already built and configured
        # this instance; Python re-invokes __init__ with the facade
        # arguments — nothing left to do.
        if getattr(self, '_facade_constructed', False):
            return
        super().__init__()
        self._backend = PythonNodeBackend()

    # ---- description / script -----------------------------------------

    def set_json_description(self, json_str: str):
        self._backend.set_json_description(json_str)
        if self._backend.default_label:
            self.label = self._backend.default_label
        self._backend.apply_description(
            lambda name, ptype: self.add_input(name, ptype),
            lambda name, ptype, persistent: self.add_output(
                name, ptype, persistent=persistent))

    @property
    def json_description(self) -> str:
        return self._backend.json_description

    @property
    def script(self) -> str:
        return self._backend.script

    @script.setter
    def script(self, value: str):
        self._backend.set_script(value)

    def _parameter_store(self) -> dict:
        # Parameters live on the backend (serialized as `arguments`);
        # the base Node parameters API operates on that store.
        return self._backend.parameters

    # ---- serialize / deserialize --------------------------------------

    def serialize(self) -> dict:
        return self._backend.serialize_into(super().serialize())

    def deserialize(self, data: dict) -> bool:
        # Apply description first so add_input / add_output run before
        # the base class restores per-port state.
        self._backend.apply_serialized_fields(
            data,
            lambda name, ptype: (
                self.add_input(name, ptype)
                if self.input_port(name) is None else None),
            lambda name, ptype, persistent: (
                self.add_output(name, ptype, persistent=persistent)
                if self.output_port(name) is None else None))
        if not super().deserialize(data):
            return False
        # Honor a `tomviz_pipeline_env` key in the description through
        # the modern per-node executor mechanism.
        from tomviz_pipeline.external import promote_description_env_path
        promote_description_env_path(
            self, data, self._backend.external_python_env_path)
        return True

    # ---- execution ----------------------------------------------------

    def query_should_auto_execute(self) -> bool:
        return self._backend.run_should_auto_execute(self)

    def transform(self, inputs: dict[str, PortData]) -> dict[str, PortData]:
        return self._backend.run_transform(self, inputs)
