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
    """SinkNode that adopts whatever input AND output ports the saved
    state declares. The output-port branch is for `sinkGroup` nodes,
    which on the C++ side are passthrough containers; for the headless
    runtime we never execute them but downstream links still need their
    output ports to exist so loading resolves correctly. Real sinks
    (sink.outline, sink.slice, …) only declare inputPorts in the JSON,
    so the output branch is a no-op for them."""

    def deserialize(self, data: dict) -> bool:
        for name, entry in (data.get('inputPorts') or {}).items():
            if self.input_port(name) is None:
                accepted = entry.get('type', ['ImageData'])
                self.add_input(name, accepted)
        for name, entry in (data.get('outputPorts') or {}).items():
            if self.output_port(name) is None:
                self.add_output(
                    name, entry.get('type', 'ImageData'),
                    persistent=bool(entry.get('persistent', True)))
        return super().deserialize(data)


def _make_inert_sink() -> SinkNode:
    return _InertSink()
