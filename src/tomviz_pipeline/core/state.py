###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Generic schema-v2 (de)serialization of the node graph — plain dicts
only, no file formats. Mirrors the C++ PipelineStateIO save/load of the
`pipeline` section:

    {
      "schemaVersion": 2,
      "pipeline": {
        "nextNodeId": <int>,
        "nodes": [ {"id", "type", "label", "state"?, ...}, ... ],
        "links": [ {"from": {"node", "port"}, "to": {"node", "port"}} ]
      }
    }

Container formats (.tvsm files, .tvh5 HDF5 bundles with embedded data
payloads) live in tomviz_pipeline.state, layered on top of this module.
"""

from __future__ import annotations

import logging
from typing import Optional

from .factory import NodeFactory
from .node import NodeState
from .node_executor import NodeExecutorFactory
from .pipeline import Pipeline

logger = logging.getLogger('tomviz_pipeline')

SCHEMA_VERSION = 2


def pipeline_to_state_dict(pipeline: Pipeline) -> dict:
    """Serialize a Pipeline to a full schema-v2 state dict."""
    nodes = []
    for node in pipeline.nodes:
        entry = node.serialize()
        entry['id'] = node.id
        entry['type'] = node.type_name
        nodes.append(entry)

    links = []
    for node in pipeline.nodes:
        for port in node.input_ports():
            link = port.link
            if link is None or link.from_port.node is None:
                continue
            links.append({
                'from': {'node': link.from_port.node.id,
                         'port': link.from_port.name},
                'to': {'node': node.id, 'port': port.name},
            })

    return {
        'schemaVersion': SCHEMA_VERSION,
        'pipeline': {
            'nextNodeId': pipeline._next_node_id,
            'nodes': nodes,
            'links': links,
        },
    }


def pipeline_from_state_dict(state: dict, state_dir=None,
                             reset_to_new: bool = False) -> Pipeline:
    """Build a Pipeline from a schema-v2 state dict.

    Node types resolve through NodeFactory — register your types first
    (for the tomviz built-ins call
    tomviz_pipeline.nodes.register_builtins()). Unknown node types are
    skipped with a warning, as in the existing Python runtime.

    Saved node states are re-applied after link resolution (link creation
    cascades staleness, which must not clobber what the file says), then
    Current nodes whose consumed outputs carry no data are downgraded to
    Stale — matching the C++ PipelineStateIO::load reconciliation passes.
    Callers that always want a full re-run (the CLI) pass
    reset_to_new=True instead.

    state_dir, when given, is recorded on the pipeline and stamped on
    every node as _state_dir before deserialize() so source nodes can
    resolve relative file paths.
    """
    schema_version = state.get('schemaVersion')
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f'Unsupported schemaVersion {schema_version!r}; '
            'only schema-v2 state files are supported.')

    pipeline_obj = state.get('pipeline')
    if not pipeline_obj:
        raise ValueError("State file is missing the 'pipeline' section")

    pipeline = Pipeline()
    pipeline.state_dir = state_dir

    id_to_node = {}
    entries_by_id = {}
    for entry in pipeline_obj.get('nodes', []):
        type_name = entry.get('type')
        node_id = entry.get('id', -1)
        if not type_name or node_id < 0:
            logger.warning('Skipping node with missing id/type: %r', entry)
            continue

        node = NodeFactory.create(type_name)
        if node is None:
            logger.warning("Unknown node type '%s' (id=%d) — skipping",
                           type_name, node_id)
            continue

        # Stamp the state dir before deserialize so source nodes can
        # resolve relative file paths.
        node._state_dir = state_dir

        if not node.deserialize(entry):
            logger.warning("Deserialize failed for node '%s' (id=%d)",
                           type_name, node_id)
            continue

        _restore_node_executor(node, entry)

        pipeline.add_node(node)
        pipeline.set_node_id(node, node_id)
        id_to_node[node_id] = node
        entries_by_id[node_id] = entry

    if 'nextNodeId' in pipeline_obj:
        pipeline._next_node_id = max(pipeline._next_node_id,
                                     int(pipeline_obj['nextNodeId']))

    for entry in pipeline_obj.get('links', []):
        from_obj = entry.get('from', {})
        to_obj = entry.get('to', {})
        from_id = from_obj.get('node', -1)
        to_id = to_obj.get('node', -1)
        from_node = id_to_node.get(from_id)
        to_node = id_to_node.get(to_id)
        if from_node is None or to_node is None:
            logger.warning('Link references unknown node: %d -> %d',
                           from_id, to_id)
            continue
        from_port = from_node.output_port(from_obj.get('port'))
        to_port = to_node.input_port(to_obj.get('port'))
        if from_port is None or to_port is None:
            logger.warning('Link references unknown port: %s -> %s',
                           from_obj.get('port'), to_obj.get('port'))
            continue
        pipeline.create_link(from_port, to_port)

    # Re-apply saved states without cascading: create_link marked
    # downstream nodes stale while wiring the graph.
    for node_id, node in id_to_node.items():
        saved = entries_by_id[node_id].get('state')
        if reset_to_new:
            node._state = NodeState.New
        elif saved == 'Current':
            node._state = NodeState.Current
        elif saved == 'Stale':
            node._state = NodeState.Stale
        else:
            node._state = NodeState.New

    if not reset_to_new:
        _downgrade_dataless_current_nodes(pipeline)

    return pipeline


def _restore_node_executor(node, entry: dict):
    block = entry.get('executor')
    if not block:
        return
    type_name = block.get('type', '')
    if not type_name:
        return
    executor = NodeExecutorFactory.create(type_name)
    if executor is None:
        logger.warning(
            "Unknown node executor type '%s' on node '%s' — running "
            'in-process', type_name, entry.get('label', ''))
        return
    if executor.deserialize(block):
        node.node_executor = executor


def _downgrade_dataless_current_nodes(pipeline: Pipeline):
    """A node saved as Current whose needed outputs carry no data can't
    be trusted — downgrade it to Stale so it re-runs (cascading). A port
    "needs data" when it is persistent or feeds a link; ports get data
    either from a container format (tvh5 payload population) or not at
    all (.tvsm). Mirrors C++ PipelineStateIO::load pass 3; note that a
    persistent-OnDisk port whose payload sits in its cache file reports
    has_data() == True and keeps its node Current."""
    for node in pipeline.nodes:
        if node.state != NodeState.Current:
            continue
        for port in node.output_ports():
            needs_data = port.persistent or port.outgoing_links
            if needs_data and not port.has_data():
                node.mark_stale()
                break


def find_node_state(state: dict, node_id: int) -> Optional[dict]:
    """Convenience: the JSON entry for one node id, or None."""
    for entry in state.get('pipeline', {}).get('nodes', []):
        if entry.get('id') == node_id:
            return entry
    return None
