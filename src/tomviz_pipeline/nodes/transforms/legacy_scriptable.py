###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""LegacyScriptableTransformNode — a transform run by a v1 script: a
JSON-described ``transform(dataset, **params)`` function or
``tomviz.operators`` class, as in the tomviz kernel catalog. Mirrors the
C++ ``LegacyPythonTransform``; the serialized form carries
``description`` (the definition as a string), ``script`` and
``arguments`` (the parameter values)."""

from tomviz_pipeline.core import PortData, TransformNode
from tomviz_pipeline.nodes.kernel_backends import LegacyKernelBackend
from tomviz_pipeline.nodes.scriptable import ScriptableNode


class LegacyScriptableTransformNode(ScriptableNode, TransformNode):
    type_name = 'transform.legacyPython'
    _backend_class = LegacyKernelBackend

    def transform(self, inputs: dict[str, PortData]) -> dict[str, PortData]:
        return self._backend.run_transform(self, inputs)
