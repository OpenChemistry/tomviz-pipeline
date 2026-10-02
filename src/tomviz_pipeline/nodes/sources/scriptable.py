###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""ScriptableSourceNode — a source whose outputs come from a schema-v2
kernel: a :class:`tomviz_pipeline.kernels.SourceKernel` subclass (or its
legacy ``tomviz.nodes.SourceNode`` spelling) whose ``produce`` method
returns the outputs dict. Mirrors the C++
``tomviz::pipeline::PythonSource``."""

from tomviz_pipeline.core import SourceNode
from tomviz_pipeline.nodes.kernel_backends import KernelBackend
from tomviz_pipeline.nodes.scriptable import ScriptableNode


class ScriptableSourceNode(ScriptableNode, SourceNode):
    type_name = 'source.python'
    _backend_class = KernelBackend

    def execute(self) -> bool:
        outputs = self._backend.run_source(self)
        if not outputs and self.output_ports():
            return False
        for name, port_data in outputs.items():
            port = self.output_port(name)
            if port is not None:
                port.set_data(port_data)
        return True
