###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Legacy names for schema-v2 operator scripts.

Older scripts import ``tomviz.nodes`` and subclass ``SourceNode`` /
``TransformNode``; the ``tomviz.nodes`` alias installed by
``tomviz_pipeline._compat`` points here, so those names resolve to the
same class objects as :mod:`tomviz_pipeline.kernels`. New scripts should
import ``tomviz_pipeline.kernels`` directly."""

from tomviz_pipeline.kernels import (
    Kernel as Node,
    SourceKernel as SourceNode,
    TransformKernel as TransformNode,
)

__all__ = ['Node', 'SourceNode', 'TransformNode']
