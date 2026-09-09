###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""SinkGroupNode and PassthroughOutputPort: a transparent container for
the sinks that read one output port. Mirrors the C++ SinkGroupNode and
PassthroughOutputPort.

A group sits between a data node's output and its sinks. Each
passthrough pair is an input port plus a PassthroughOutputPort that
forwards whatever feeds the input, with zero copies, so the sinks see
the upstream data and moving them all to another upstream port (when a
transform is inserted, say) is one link change on the group.
Applications draw the group as one compact card instead of one card per
sink."""

from __future__ import annotations

from typing import Optional

from .events import Connection
from .node import (
    InputPort,
    Node,
    OutputPort,
    PortData,
    PortDataHandle,
    SinkNode,
)
from .persistence import DataLocation

DEFAULT_GROUP_LABEL = 'Visualizations'


class PassthroughOutputPort(OutputPort):
    """An OutputPort proxy that delegates data access and signal
    forwarding to a source OutputPort, so grouped sinks read the upstream
    data without any copying.

    A passthrough owns no data: it always reports non-persistent (state
    writers skip it, and an old state file's persistent flag cannot flip
    it) and set_data() with a payload is refused. Its effective type
    follows the source's through the group's type inference (the
    passthrough is declared 'ImageData' with itself as inference source).
    Only sinks may link to it."""

    def __init__(self, name: str, port_type: str):
        self._source: Optional[OutputPort] = None
        self._source_connections: list[Connection] = []
        super().__init__(name, port_type, persistent=False)

    # ---- source ----------------------------------------------------------

    def set_source(self, source: Optional[OutputPort]):
        """Set (or change) the upstream port this proxy delegates to;
        None disconnects. The source's data_changed and
        data_location_changed are re-emitted as this port's own."""
        if source is self._source:
            return
        for connection in self._source_connections:
            connection.disconnect()
        self._source_connections = []
        self._source = source
        if source is not None:
            self._source_connections = [
                source.data_changed.connect(self._forward_data_changed),
                source.data_location_changed.connect(
                    self._forward_data_location_changed),
            ]

    @property
    def source(self) -> Optional[OutputPort]:
        return self._source

    def _forward_data_changed(self, _port):
        self.data_changed.emit(self)

    def _forward_data_location_changed(self, _port, location):
        self.data_location_changed.emit(self, location)

    # ---- persistence -----------------------------------------------------

    @property
    def persistent(self) -> bool:
        return False

    @persistent.setter
    def persistent(self, value: bool):
        # Nothing to persist: the source port owns the payload.
        pass

    # ---- data ------------------------------------------------------------

    def data_location(self) -> DataLocation:
        if self._source is None:
            return DataLocation.Nowhere
        return self._source.data_location()

    def has_data(self) -> bool:
        return self._source is not None and self._source.has_data()

    def data(self) -> Optional[PortData]:
        if self._source is None:
            return None
        return self._source.data()

    def materialize(self) -> Optional[PortDataHandle]:
        if self._source is None:
            return None
        return self._source.materialize()

    def take(self) -> Optional[PortDataHandle]:
        if self._source is None:
            return None
        return self._source.take()

    def set_data(self, data: Optional[PortData]):
        if data is None:
            return  # clearing a port that holds nothing
        raise TypeError(
            f"Passthrough port '{self.name}' forwards upstream data and "
            'cannot hold its own')

    def clear_data(self):
        pass

    def can_accept_link(self, to_port: InputPort) -> bool:
        """Only sinks may join a group."""
        return isinstance(to_port.node, SinkNode)


class SinkGroupNode(Node):
    """A pipeline node that acts as a transparent grouping container for
    sinks.

    Inherits directly from Node, not TransformNode or SinkNode: it has
    output ports (unlike SinkNode) but performs no data transformation
    (unlike TransformNode). add_passthrough() creates an input port and
    a PassthroughOutputPort pair; the output delegates to whatever port
    feeds the input, tracked through the input's connection_changed
    signal, so re-pointing the group's single input link moves every
    grouped sink at once.

    Executes like any node: it succeeds when every input is linked and
    has data (the executor then marks it Current), so sinks behind it
    are planned and re-run exactly like sinks linked to the data."""

    type_name = 'sinkGroup'

    def __init__(self):
        super().__init__()
        self.label = DEFAULT_GROUP_LABEL

    def add_passthrough(self, name: str,
                        port_type: str) -> PassthroughOutputPort:
        """Add a passthrough port pair: an input accepting ``port_type``
        and a PassthroughOutputPort of the same name whose effective type
        follows the input's source (type inference is configured
        accordingly). Returns the output port."""
        input_port = self.add_input(name, port_type)
        output = PassthroughOutputPort(name, port_type)
        output.node = self
        self._output_ports.append(output)
        self.type_inference_sources[name] = name

        def on_connection_changed(port: InputPort, output=output):
            link = port.link
            output.set_source(None if link is None else link.from_port)

        input_port.connection_changed.connect(on_connection_changed)
        return output

    def passthrough_output(self, input_port: InputPort
                           ) -> Optional[PassthroughOutputPort]:
        """The output port forwarding ``input_port``, or None."""
        output = self.output_port(input_port.name)
        if isinstance(output, PassthroughOutputPort):
            return output
        return None

    def sinks(self) -> list[SinkNode]:
        """Every sink linked to an output of this group, once each, in
        link order."""
        result: list[SinkNode] = []
        for port in self._output_ports:
            for link in port.outgoing_links:
                node = link.to_port.node
                if (isinstance(node, SinkNode)
                        and not any(n is node for n in result)):
                    result.append(node)
        return result

    def execute(self) -> bool:
        return all(port.link is not None and port.has_data()
                   for port in self._input_ports)

    def deserialize(self, data: dict) -> bool:
        # Recreate the passthrough pairs from the saved outputPorts map
        # before Node.deserialize applies the label, properties and
        # per-port state.
        for name, entry in (data.get('outputPorts') or {}).items():
            if self.output_port(name) is None:
                self.add_passthrough(name, entry.get('type', 'ImageData'))
        return super().deserialize(data)
