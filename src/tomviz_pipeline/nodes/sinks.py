###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Inert sink placeholder. All sink type strings collapse to the same
inert class — the headless runtime never executes sinks. A placeholder
keeps the sink's own settings so saving the pipeline writes the sink
back as it was loaded."""

import copy

from tomviz_pipeline.core import SinkNode

# All sink type strings collapse to the same inert class. We don't need
# per-sink behavior in the headless runtime; deserialize() on the base
# SinkNode is enough to load and ignore them.
_SINK_TYPES = (
    'sink.clip',
    'sink.contour',
    'sink.molecule',
    'sink.outline',
    'sink.plot',
    'sink.ruler',
    'sink.scaleCube',
    'sink.segment',
    'sink.slice',
    'sink.threshold',
    'sink.volume',
    'sink.volumeStats',
)

# The keys of a node entry that belong to the graph: the ones
# Node.serialize() writes, plus the id and type the pipeline adds.
# Everything else in a sink entry is the sink's own.
_NODE_KEYS = frozenset((
    'id',
    'type',
    'label',
    'state',
    'breakpoint',
    'properties',
    'typeInferenceSources',
    'executor',
    'autoExecute',
    'outputPorts',
    'inputPorts',
))


class _InertSink(SinkNode):
    """SinkNode that adopts whatever input ports the saved state
    declares. The rest of a sink entry (the desktop's ``viewId``,
    ``visible`` and per-type settings) goes to ``settings``, which the
    runtime does not interpret and serialize() writes back verbatim;
    the label, state and ports are the node's, so graph edits (a new
    label, a sink relinked) are what gets saved. The `sinkGroup`
    container that fans one output out to several sinks is
    core.SinkGroupNode, registered separately."""

    def __init__(self):
        super().__init__()
        self.settings: dict = {}

    def serialize(self) -> dict:
        data = copy.deepcopy(self.settings)
        data.update(super().serialize())
        return data

    def deserialize(self, data: dict) -> bool:
        for name, entry in (data.get('inputPorts') or {}).items():
            if self.input_port(name) is None:
                accepted = entry.get('type', ['ImageData'])
                self.add_input(name, accepted)
        self.settings = {key: copy.deepcopy(value)
                         for key, value in data.items()
                         if key not in _NODE_KEYS}
        return super().deserialize(data)


def _make_inert_sink() -> SinkNode:
    return _InertSink()
