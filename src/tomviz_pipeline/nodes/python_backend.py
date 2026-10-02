###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Former home of ``PythonNodeBackend``, kept so existing imports work.
It is now ``tomviz_pipeline.nodes.kernel_backends.KernelBackend``."""

from tomviz_pipeline.nodes.kernel_backends import (  # noqa: F401
    KernelBackend as PythonNodeBackend,
)

__all__ = ['PythonNodeBackend']
