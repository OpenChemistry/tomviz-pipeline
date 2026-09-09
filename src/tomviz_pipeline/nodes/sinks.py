###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Inert sink placeholder. All sink type strings collapse to the same
inert class — the headless runtime never executes sinks; deserialize()
on the base SinkNode is enough to load and ignore them."""

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


class _InertSink(SinkNode):
    """SinkNode that adopts whatever input ports the saved state
    declares. Real sinks (sink.outline, sink.slice, …) only declare
    inputPorts in the JSON. The `sinkGroup` container that fans one
    output out to several sinks is core.SinkGroupNode, registered
    separately."""

    def deserialize(self, data: dict) -> bool:
        for name, entry in (data.get('inputPorts') or {}).items():
            if self.input_port(name) is None:
                accepted = entry.get('type', ['ImageData'])
                self.add_input(name, accepted)
        return super().deserialize(data)


def _make_inert_sink() -> SinkNode:
    return _InertSink()
