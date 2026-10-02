###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""ScriptableTransformNode — a transform run by a schema-v2 kernel: a
:class:`tomviz_pipeline.kernels.TransformKernel` subclass (or its legacy
``tomviz.nodes.TransformNode`` spelling) whose ``transform`` method maps
the inputs dict to the outputs dict. Mirrors the C++
``tomviz::pipeline::PythonTransform``."""

from tomviz_pipeline.core import PortData, TransformNode
from tomviz_pipeline.nodes.kernel_backends import KernelBackend
from tomviz_pipeline.nodes.scriptable import ScriptableNode


class ScriptableTransformNode(ScriptableNode, TransformNode):
    type_name = 'transform.python'
    _backend_class = KernelBackend

    def transform(self, inputs: dict[str, PortData]) -> dict[str, PortData]:
        return self._backend.run_transform(self, inputs)
