###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Former home of ``LegacyPythonTransform``, kept so existing imports
work. It is now ``LegacyScriptableTransformNode`` in
``tomviz_pipeline.nodes.transforms.legacy_scriptable``."""

from tomviz_pipeline.nodes.transforms.legacy_scriptable import (  # noqa: F401
    LegacyScriptableTransformNode as LegacyPythonTransform,
)

__all__ = ['LegacyPythonTransform']
