###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Tests for the pure-Python pipeline graph runtime: topo sort, leaf
detection (with sinks ignored), state cascade on failure, schema-v2
state-file loading."""

import json

import numpy as np
import pytest

from tomviz_pipeline import (
    Dataset,
    DefaultExecutor,
    NodeFactory,
    NodeState,
    Pipeline,
    PortData,
    SinkNode,
    SourceNode,
    TransformNode,
    load_state,
    register_builtins,
)


@pytest.fixture(autouse=True)
def _ensure_builtins_registered():
    register_builtins()


def _make_dataset(value=1.0, shape=(2, 2, 3)):
    arr = np.full(shape, value, dtype=np.float32, order='F')
    ds = Dataset({'Scalars': arr}, 'Scalars')
    ds.spacing = (1.0, 1.0, 1.0)
    return ds


class _MemorySource(SourceNode):
    type_name = 'test.memorySource'

    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset
        self.add_output('volume', 'ImageData')

    def execute(self):
        port = self.output_port('volume')
        port.set_data(PortData(self._dataset, 'ImageData'))
        return True


class _ScaleTransform(TransformNode):
    type_name = 'test.scale'

    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        self.add_input('volume', 'ImageData')
        self.add_output('output', 'ImageData')

    def transform(self, inputs):
        ds = inputs['volume'].payload
        out = Dataset({n: a * self.factor for n, a in ds.arrays.items()},
                      ds.active_name)
        return {'output': PortData(out, 'ImageData')}


class _AlwaysFails(TransformNode):
    type_name = 'test.fails'

    def __init__(self):
        super().__init__()
        self.add_input('volume', 'ImageData')
        self.add_output('output', 'ImageData')

    def execute(self):
        return False


# ---- topology -----------------------------------------------------------


def test_execution_order_is_topological():
    p = Pipeline()
    src = _MemorySource(_make_dataset())
    a = _ScaleTransform(2.0)
    b = _ScaleTransform(3.0)
    p.add_node(src)
    p.add_node(b)  # added in reverse to confirm sort
    p.add_node(a)
    p.create_link(src.output_port('volume'), a.input_port('volume'))
    p.create_link(a.output_port('output'), b.input_port('volume'))
    order = p.execution_order()
    assert order.index(src) < order.index(a) < order.index(b)


def test_execute_chain_propagates_payload():
    p = Pipeline()
    src = _MemorySource(_make_dataset(value=2.0))
    scale = _ScaleTransform(5.0)
    p.add_node(src)
    p.add_node(scale)
    p.create_link(src.output_port('volume'), scale.input_port('volume'))
    DefaultExecutor(p).execute()
    out = scale.output_port('output').data().payload
    assert np.all(out.active_scalars == 10.0)
    assert scale.state == NodeState.Current


def test_failure_marks_downstream_stale():
    p = Pipeline()
    src = _MemorySource(_make_dataset())
    failing = _AlwaysFails()
    downstream = _ScaleTransform(1.0)
    for n in (src, failing, downstream):
        p.add_node(n)
    p.create_link(src.output_port('volume'), failing.input_port('volume'))
    p.create_link(failing.output_port('output'),
                  downstream.input_port('volume'))
    ok = DefaultExecutor(p).execute()
    assert ok is False
    assert failing.state == NodeState.Stale
    assert downstream.state == NodeState.Stale


def test_sinks_are_skipped():
    """A SinkNode never runs and never blocks topo sort."""
    p = Pipeline()
    src = _MemorySource(_make_dataset())
    sink = SinkNode()
    sink.add_input('volume', 'ImageData')
    p.add_node(src)
    p.add_node(sink)
    p.create_link(src.output_port('volume'), sink.input_port('volume'))
    DefaultExecutor(p).execute()
    # Sink stays New (executor short-circuits sinks before any mark).
    assert sink.state == NodeState.New
    assert src.state == NodeState.Current


def test_cycle_detection_raises():
    """create_link rejects a cycle up front with ValueError (the old
    runtime allowed the link and only had execution_order() raise
    RuntimeError later). Since a cycle can no longer be built through
    the public API, the old execution_order() RuntimeError assertion
    is dropped."""
    p = Pipeline()
    a = _ScaleTransform(1.0)
    b = _ScaleTransform(1.0)
    p.add_node(a)
    p.add_node(b)
    p.create_link(a.output_port('output'), b.input_port('volume'))
    with pytest.raises(ValueError):
        p.create_link(b.output_port('output'), a.input_port('volume'))
    # The offending link was not added; the graph is still sortable.
    assert len(p.links) == 1
    assert p.execution_order()


# ---- state-file loading -------------------------------------------------


def test_load_state_unknown_node_skipped(tmp_path):
    state = {
        'schemaVersion': 2,
        'pipeline': {
            'nodes': [
                {'id': 1, 'type': 'transform.crop',
                 'label': 'Crop',
                 'bounds': [0, 1, 0, 1, 0, 1]},
                {'id': 2, 'type': 'transform.bogus',
                 'label': 'Bogus'},
            ],
            'links': [],
        },
    }
    f = tmp_path / 'state.tvsm'
    f.write_text(json.dumps(state))
    p = load_state(f)
    assert len(p.nodes) == 1
    assert p.nodes[0].id == 1


def test_load_state_resolves_links(tmp_path):
    state = {
        'schemaVersion': 2,
        'pipeline': {
            'nodes': [
                {'id': 1, 'type': 'source.reader', 'label': 'Reader',
                 'fileNames': ['nonexistent.emd']},
                {'id': 2, 'type': 'transform.convertToFloat',
                 'label': 'CvtFloat'},
            ],
            'links': [
                {'from': {'node': 1, 'port': 'volume'},
                 'to':   {'node': 2, 'port': 'volume'}},
            ],
        },
    }
    f = tmp_path / 'state.tvsm'
    f.write_text(json.dumps(state))
    p = load_state(f)
    assert len(p.nodes) == 2
    assert len(p.links) == 1
    src_node = p.node_by_id(1)
    cvt_node = p.node_by_id(2)
    assert src_node.output_port('volume').outgoing_links
    assert (cvt_node.input_port('volume').link.from_port ==
            src_node.output_port('volume'))


def test_load_state_rejects_wrong_schema(tmp_path):
    state = {'schemaVersion': 1,
             'pipeline': {'nodes': [], 'links': []}}
    f = tmp_path / 'state.tvsm'
    f.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        load_state(f)


# ---- factory ------------------------------------------------------------


def test_load_state_resolves_sinkgroup_passthrough_links(tmp_path):
    """Regression: a sinkGroup node has both inputPorts and outputPorts
    declared in the saved JSON. The loader must adopt both so links
    landing on / leaving its 'volume' port resolve. With this in place,
    a transform feeding only into a sinkGroup (which fans out to real
    sinks) is correctly identified as a leaf."""
    import json as _json

    state = {
        'schemaVersion': 2,
        'pipeline': {
            'nodes': [
                # Stand-in for a transform node whose only consumer is
                # the sinkGroup. Use ConvertToFloat (always present).
                {'id': 1, 'type': 'transform.convertToFloat',
                 'label': 'Cvt'},
                {'id': 2, 'type': 'sinkGroup', 'label': 'Modules',
                 'inputPorts': {'volume': {'type': ['ImageData']}},
                 'outputPorts': {'volume': {'type': 'ImageData',
                                            'persistent': False}}},
                {'id': 3, 'type': 'sink.outline', 'label': 'Outline',
                 'inputPorts': {'volume': {'type': ['ImageData']}}},
            ],
            'links': [
                {'from': {'node': 1, 'port': 'output'},
                 'to':   {'node': 2, 'port': 'volume'}},
                {'from': {'node': 2, 'port': 'volume'},
                 'to':   {'node': 3, 'port': 'volume'}},
            ],
        },
    }
    f = tmp_path / 'state.tvsm'
    f.write_text(_json.dumps(state))
    p = load_state(f)
    # Three nodes load; both links resolve.
    assert len(p.nodes) == 3
    assert len(p.links) == 2
    cvt = p.node_by_id(1)
    sg = p.node_by_id(2)
    assert cvt.output_port('output').outgoing_links
    # The sinkGroup adopted the declared output port and the link
    # leaving it resolved.
    assert sg.output_port('volume') is not None
    assert sg.output_port('volume').outgoing_links


def test_load_state_overrides_saved_current_state(tmp_path):
    """Regression: a state file saved after a successful in-app run
    stamps every node `state: "Current"`. The CLI must NOT trust that
    — otherwise nothing actually runs. Saved state should be reset to
    New on load so the executor re-runs the graph end-to-end."""
    import json as _json

    state = {
        'schemaVersion': 2,
        'pipeline': {
            'nodes': [
                {'id': 1, 'type': 'transform.convertToFloat',
                 'label': 'Cvt', 'state': 'Current'},
                {'id': 2, 'type': 'transform.crop', 'label': 'Crop',
                 'state': 'Current'},
            ],
            'links': [
                {'from': {'node': 1, 'port': 'output'},
                 'to':   {'node': 2, 'port': 'volume'}},
            ],
        },
    }
    f = tmp_path / 'state.tvsm'
    f.write_text(_json.dumps(state))
    p = load_state(f)
    for node in p.nodes:
        assert node.state == NodeState.New


def test_factory_creates_known_types():
    for t in (
        'source.reader', 'transform.crop', 'transform.threshold',
        'transform.convertToFloat', 'transform.convertToVolume',
        'transform.transposeData', 'transform.setTiltAngles',
        'transform.legacyPython', 'sink.volume',
    ):
        assert NodeFactory.create(t) is not None


def test_factory_unknown_returns_none():
    assert NodeFactory.create('something.bogus') is None
