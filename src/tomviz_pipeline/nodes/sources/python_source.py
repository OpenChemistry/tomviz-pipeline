###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Former home of ``PythonSource``, kept so existing imports work. It
is now ``ScriptableSourceNode`` in
``tomviz_pipeline.nodes.sources.scriptable``."""

from tomviz_pipeline.nodes.sources.scriptable import (  # noqa: F401
    ScriptableSourceNode as PythonSource,
)

__all__ = ['PythonSource']
