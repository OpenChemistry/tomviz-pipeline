###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Former home of ``PythonNode``, kept so existing imports work.
``PythonNode`` is now ``tomviz_pipeline.nodes.scriptable.ScriptableNode``,
which also builds v1 (legacy) nodes."""

from tomviz_pipeline.nodes.scriptable import (  # noqa: F401
    ScriptableNode as PythonNode,
    script_from_kernel,
)

__all__ = ['PythonNode', 'script_from_kernel']
