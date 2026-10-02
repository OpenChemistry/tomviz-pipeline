###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Former home of ``PythonTransform``, kept so existing imports work.
It is now ``ScriptableTransformNode`` in
``tomviz_pipeline.nodes.transforms.scriptable``."""

from tomviz_pipeline.nodes.transforms.scriptable import (  # noqa: F401
    ScriptableTransformNode as PythonTransform,
)

__all__ = ['PythonTransform']
