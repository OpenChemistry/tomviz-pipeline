###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""PythonSource — schema-v2 Python source node for the headless runtime.

Mirrors the C++ ``tomviz::pipeline::PythonSource``. The user-facing class
is a subclass of :class:`tomviz_pipeline.kernels.SourceKernel` (or its legacy
``tomviz.nodes.SourceNode`` spelling) whose
``produce`` method returns the outputs dict."""

from tomviz_pipeline.core import SourceNode
from tomviz_pipeline.nodes.python_backend import PythonNodeBackend
from tomviz_pipeline.nodes.python_node import PythonNode


class PythonSource(PythonNode, SourceNode):
    type_name = 'source.python'

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
        # Sources have no inputs — pass a null AddInputFn. A v2 source
        # description that erroneously declares inputs is silently
        # dropped here; the menu-routing reaction in the app rejects
        # the same case at construction time.
        self._backend.apply_description(
            None,
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
        self._backend.apply_serialized_fields(
            data,
            None,
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

    def execute(self) -> bool:
        outputs = self._backend.run_source(self)
        if not outputs and self.output_ports():
            return False
        for name, port_data in outputs.items():
            port = self.output_port(name)
            if port is not None:
                port.set_data(port_data)
        return True
