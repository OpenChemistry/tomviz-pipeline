###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Leaf-detection + per-leaf writing logic shared by the CLI and the
batch (`run_many`) interface. Sinks are ignored at execution time and
sink groups merely forward data to them, so the leaves of the graph for
output purposes are the other nodes whose remaining outgoing links all
land on sinks or sink groups."""

import logging
import re
from pathlib import Path

from tomviz_pipeline.core import SinkGroupNode, SinkNode
from tomviz_pipeline.writers import writer_for


logger = logging.getLogger('tomviz_pipeline')

# Nodes that produce no output of their own.
_SINK_LIKE = (SinkNode, SinkGroupNode)


def sanitize(name: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_')
    return safe or 'node'


def is_data_leaf(node) -> bool:
    """A node is a data leaf if every one of its outgoing links lands on
    a sink (or a sink group). A node with zero outgoing links also
    qualifies. Sinks and groups themselves are never leaves."""
    if isinstance(node, _SINK_LIKE):
        return False
    for port in node.output_ports():
        for link in port.outgoing_links:
            consumer = link.to_port.node
            if consumer is None:
                continue
            if not isinstance(consumer, _SINK_LIKE):
                return False
    return True


def is_unconsumed_port(port) -> bool:
    """A port should be written to disk if no non-sink node consumes it."""
    for link in port.outgoing_links:
        consumer = link.to_port.node
        if consumer is None:
            continue
        if not isinstance(consumer, _SINK_LIKE):
            return False
    return True


def write_leaf_outputs(pipeline, output_dir: Path) -> int:
    """Walk the pipeline, write every unconsumed output port of every
    non-sink leaf node into ``output_dir`` using the writer registered
    for that port's type. Returns the number of files written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for node in pipeline.nodes:
        if isinstance(node, _SINK_LIKE):
            continue
        if not is_data_leaf(node):
            continue
        for port in node.output_ports():
            if not is_unconsumed_port(port):
                continue
            # materialize() so a persistent-OnDisk leaf whose payload
            # was evicted to its cache file is still written; the
            # handle keeps it alive until the writer is done.
            handle = port.materialize()
            data = handle.data if handle is not None else None
            if data is None or data.payload is None:
                logger.warning("No data on leaf port %s.%s — skipping",
                               node.label or type(node).__name__,
                               port.name)
                continue
            entry = writer_for(port.port_type)
            if entry is None:
                logger.warning(
                    "No writer registered for port type %r on %s.%s — "
                    "skipping", port.port_type,
                    node.label or type(node).__name__, port.name)
                continue
            extension, writer = entry
            label = sanitize(node.label or type(node).__name__)
            filename = (f'{node.id}_{label}__{sanitize(port.name)}'
                        f'.{extension}')
            target = output_dir / filename
            logger.info("Writing %s", target)
            try:
                writer(data.payload, target)
            except Exception:
                logger.exception("Failed to write %s", target)
                continue
            written += 1
    return written
