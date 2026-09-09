###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Built-in tomviz node types layered on the generic core engine.
register_builtins() populates tomviz_pipeline.core.NodeFactory with every
type string a schema-v2 state file can contain."""

from tomviz_pipeline.core import NodeFactory, SinkGroupNode, SourceNode

__all__ = ['register_builtins']

_registered = False


def register_builtins():
    """Register all built-in node types. Idempotent."""
    global _registered
    if _registered:
        return
    _registered = True

    # Source types
    NodeFactory.register('source.generic', SourceNode)
    from tomviz_pipeline.nodes.sources.reader import ReaderSourceNode
    NodeFactory.register('source.reader', ReaderSourceNode)
    from tomviz_pipeline.nodes.sources.python_source import PythonSource
    NodeFactory.register('source.python', PythonSource)

    # Transform types
    from tomviz_pipeline.nodes.transforms.legacy_python import (
        LegacyPythonTransform,
    )
    from tomviz_pipeline.nodes.transforms.python_transform import (
        PythonTransform,
    )
    from tomviz_pipeline.nodes.transforms.convert_to_volume import (
        ConvertToVolumeTransform,
    )
    from tomviz_pipeline.nodes.transforms.set_tilt_angles import (
        SetTiltAnglesTransform,
    )
    from tomviz_pipeline.nodes.transforms.convert_to_float import (
        ConvertToFloatTransform,
    )
    from tomviz_pipeline.nodes.transforms.transpose import (
        TransposeDataTransform,
    )
    from tomviz_pipeline.nodes.transforms.crop import CropTransform
    from tomviz_pipeline.nodes.transforms.cylindrical_crop import (
        CylindricalCropTransform,
    )
    from tomviz_pipeline.nodes.transforms.threshold import ThresholdTransform

    NodeFactory.register('transform.legacyPython', LegacyPythonTransform)
    NodeFactory.register('transform.python', PythonTransform)
    NodeFactory.register('transform.convertToVolume',
                         ConvertToVolumeTransform)
    NodeFactory.register('transform.setTiltAngles', SetTiltAnglesTransform)
    NodeFactory.register('transform.convertToFloat', ConvertToFloatTransform)
    NodeFactory.register('transform.transposeData', TransposeDataTransform)
    NodeFactory.register('transform.crop', CropTransform)
    NodeFactory.register('transform.cylindricalCrop',
                         CylindricalCropTransform)
    NodeFactory.register('transform.threshold', ThresholdTransform)

    # Sinks: all collapse to the inert SinkNode placeholder. Each defines
    # its own input port set lazily when deserialize() inspects the JSON.
    from tomviz_pipeline.nodes.sinks import _SINK_TYPES, _make_inert_sink
    for t in _SINK_TYPES:
        NodeFactory.register(t, _make_inert_sink)
    # Sink groups are real passthrough containers, not placeholders.
    NodeFactory.register(SinkGroupNode.type_name, SinkGroupNode)

    # Per-node executors: 'external' round-trips the {type, envPath}
    # executor block on nodes configured to run in another Python env.
    from tomviz_pipeline.external import register_external_executor
    register_external_executor()
