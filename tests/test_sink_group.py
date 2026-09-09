###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""SinkGroupNode / PassthroughOutputPort: data forwarding, execution
through a group, the sinks-only link rule, re-pointing every grouped
sink with one link change, state round trips, and the writers that
must not mistake a group for a data leaf."""

import pytest

from tomviz_pipeline import (
    DataLocation,
    NodeExecState,
    NodeFactory,
    NodeState,
    PassthroughOutputPort,
    Pipeline,
    PortData,
    SinkGroupNode,
    SinkNode,
    SourceNode,
    TransformNode,
    pipeline_from_state_dict,
    pipeline_to_state_dict,
    register_builtins,
)
from tomviz_pipeline.leaf_writer import is_data_leaf, is_unconsumed_port


class _Const(SourceNode):
    type_name = 'sgtest.const'

    def __init__(self, value=1, port_type='ImageData'):
        super().__init__()
        self.value = value
        self.add_output('out', port_type)

    def execute(self):
        port = self.output_port('out')
        port.set_data(PortData(self.value, port.port_type))
        return True


class _Double(TransformNode):
    type_name = 'sgtest.double'

    def __init__(self):
        super().__init__()
        self.add_input('in', 'ImageData')
        self.add_output('out', 'ImageData')

    def transform(self, inputs):
        return {'out': PortData(inputs['in'].payload * 2, 'ImageData')}


class _Collect(SinkNode):
    type_name = 'sgtest.collect'
    executable = True

    def __init__(self):
        super().__init__()
        self.add_input('in', 'ImageData')
        self.seen = []

    def consume(self, inputs):
        self.seen.append(inputs['in'].payload)
        return True


@pytest.fixture(autouse=True)
def _registry():
    register_builtins()
    for cls in (_Const, _Double, _Collect):
        NodeFactory.register(cls.type_name, cls)


def _grouped(n_sinks=2, value=1, port_type='ImageData'):
    """source -> group -> n sinks."""
    p = Pipeline()
    src = p.add_node(_Const(value, port_type))
    group = p.add_node(SinkGroupNode())
    group.add_passthrough('out', 'ImageData')
    p.create_link(src.output_port('out'), group.input_port('out'))
    sinks = []
    for _ in range(n_sinks):
        sink = p.add_node(_Collect())
        p.create_link(group.output_port('out'), sink.input_port('in'))
        sinks.append(sink)
    return p, src, group, sinks


# ---- InputPort.connection_changed -------------------------------------------

def test_input_port_reports_link_changes():
    p = Pipeline()
    src = p.add_node(_Const())
    dbl = p.add_node(_Double())
    seen = []
    dbl.input_port('in').connection_changed.connect(seen.append)

    link = p.create_link(src.output_port('out'), dbl.input_port('in'))
    assert seen == [dbl.input_port('in')]
    p.remove_link(link)
    assert len(seen) == 2
    assert dbl.input_port('in').link is None


# ---- passthrough -------------------------------------------------------------

def test_passthrough_follows_the_input_link():
    p, src, group, _sinks = _grouped(n_sinks=0, port_type='TiltSeries')
    out = group.output_port('out')
    assert isinstance(out, PassthroughOutputPort)
    assert out.source is src.output_port('out')
    assert out.node is group
    # Effective type follows the source; the declared type is what was
    # asked for.
    assert out.port_type == 'TiltSeries'
    assert out.declared_type == 'ImageData'
    assert group.passthrough_output(group.input_port('out')) is out

    p.remove_link(group.input_port('out').link)
    assert out.source is None
    assert out.port_type == 'ImageData'
    assert not out.has_data()
    assert out.data_location() == DataLocation.Nowhere


def test_passthrough_forwards_data_and_signals():
    p, src, group, _sinks = _grouped(n_sinks=0)
    out = group.output_port('out')
    changed = []
    located = []
    out.data_changed.connect(changed.append)
    out.data_location_changed.connect(lambda port, loc: located.append(
        (port, loc)))

    assert out.data() is None
    src.execute()
    assert out.has_data()
    assert out.data().payload == 1
    assert out.materialize().data.payload == 1
    assert out.data_location() == DataLocation.InMemory
    assert changed == [out]
    assert located == [(out, DataLocation.InMemory)]

    # Re-pointing the input drops the old source's signals.
    other = p.add_node(_Const(5))
    p.create_link(other.output_port('out'), group.input_port('out'))
    src.execute()
    assert changed == [out]
    assert out.data() is None
    other.execute()
    assert out.data().payload == 5


def test_passthrough_owns_no_data():
    _p, _src, group, _sinks = _grouped(n_sinks=0)
    out = group.output_port('out')
    assert out.persistent is False
    out.persistent = True  # ignored: the source owns the payload
    assert out.persistent is False
    out.set_data(None)  # clearing nothing is fine
    out.clear_data()
    with pytest.raises(TypeError):
        out.set_data(PortData(1, 'ImageData'))


def test_only_sinks_may_link_to_a_passthrough():
    p, _src, group, _sinks = _grouped(n_sinks=0)
    dbl = p.add_node(_Double())
    with pytest.raises(ValueError):
        p.create_link(group.output_port('out'), dbl.input_port('in'))
    assert dbl.input_port('in').link is None
    assert group.output_port('out').outgoing_links == []


# ---- execution ---------------------------------------------------------------

def test_grouped_sinks_receive_the_upstream_data():
    p, src, group, sinks = _grouped(n_sinks=3, value=7)
    assert p.execute().succeeded()
    assert [s.seen for s in sinks] == [[7], [7], [7]]
    assert group.state == NodeState.Current
    assert group.exec_state == NodeExecState.Idle
    assert all(s.state == NodeState.Current for s in sinks)


def test_group_without_input_fails():
    p = Pipeline()
    group = p.add_node(SinkGroupNode())
    group.add_passthrough('out', 'ImageData')
    sink = p.add_node(_Collect())
    p.create_link(group.output_port('out'), sink.input_port('in'))

    assert not p.execute().succeeded()
    assert group.exec_state == NodeExecState.Failed
    assert sink.seen == []


def test_repointing_the_group_moves_every_sink():
    p, src, group, sinks = _grouped(n_sinks=2, value=3)
    p.execute()

    dbl = p.add_node(_Double())
    p.create_link(src.output_port('out'), dbl.input_port('in'))
    # One link change: the group (and so every sink) now reads dbl.
    p.create_link(dbl.output_port('out'), group.input_port('out'))
    assert group.output_port('out').source is dbl.output_port('out')
    assert group.state == NodeState.Stale
    assert all(s.state == NodeState.Stale for s in sinks)

    assert p.execute().succeeded()
    assert [s.seen for s in sinks] == [[3, 6], [3, 6]]
    # The source was Current with data: it did not re-run.
    assert src.output_port('out').data().payload == 3


def test_transient_upstream_flows_through_the_group():
    p, src, group, sinks = _grouped(n_sinks=2, value=4)
    src.output_port('out').persistent = False

    assert p.execute().succeeded()
    assert [s.seen for s in sinks] == [[4], [4]]
    # The payload was taken through the group for the plan and released
    # with it: nothing pins it afterwards.
    assert not src.output_port('out').has_data()
    assert not group.output_port('out').has_data()


def test_sinks_lists_each_member_once():
    p, _src, group, sinks = _grouped(n_sinks=2)
    assert group.sinks() == sinks
    dbl = p.add_node(_Double())
    assert dbl not in group.sinks()


# ---- state -------------------------------------------------------------------

def test_state_round_trip():
    p, src, group, sinks = _grouped(n_sinks=2, port_type='TiltSeries')
    group.label = 'Visualizations'
    state = pipeline_to_state_dict(p)

    entry = next(n for n in state['pipeline']['nodes']
                 if n['id'] == group.id)
    assert entry['type'] == 'sinkGroup'
    assert entry['label'] == 'Visualizations'
    assert entry['inputPorts'] == {'out': {'type': ['ImageData']}}
    # Declared type and never persistent, whatever the source is.
    assert entry['outputPorts'] == {
        'out': {'type': 'ImageData', 'persistent': False}}
    assert entry['typeInferenceSources'] == {'out': 'out'}
    assert len(state['pipeline']['links']) == 3

    p2 = pipeline_from_state_dict(state)
    group2 = p2.node_by_id(group.id)
    assert isinstance(group2, SinkGroupNode)
    out2 = group2.output_port('out')
    assert isinstance(out2, PassthroughOutputPort)
    assert out2.source is p2.node_by_id(src.id).output_port('out')
    assert out2.port_type == 'TiltSeries'
    assert out2.declared_type == 'ImageData'
    assert [s.id for s in group2.sinks()] == [s.id for s in sinks]

    assert p2.execute().succeeded()
    assert all(s.seen == [1] for s in group2.sinks())


def test_saved_persistent_flag_cannot_flip_a_passthrough():
    state = {
        'schemaVersion': 2,
        'pipeline': {
            'nodes': [
                {'id': 1, 'type': 'sgtest.const', 'label': 'Source'},
                {'id': 2, 'type': 'sinkGroup', 'label': 'Modules',
                 'inputPorts': {'volume': {'type': ['ImageData']}},
                 'outputPorts': {'volume': {'type': 'ImageData',
                                            'persistent': True,
                                            'persistenceMode': 'disk'}}},
                {'id': 3, 'type': 'sgtest.collect', 'label': 'Sink'},
            ],
            'links': [
                {'from': {'node': 1, 'port': 'out'},
                 'to': {'node': 2, 'port': 'volume'}},
                {'from': {'node': 2, 'port': 'volume'},
                 'to': {'node': 3, 'port': 'in'}},
            ],
        },
    }
    p = pipeline_from_state_dict(state)
    group = p.node_by_id(2)
    out = group.output_port('volume')
    assert isinstance(out, PassthroughOutputPort)
    assert out.persistent is False
    assert p.execute().succeeded()
    assert p.node_by_id(3).seen == [1]


# ---- writers -----------------------------------------------------------------

def test_a_group_is_not_a_data_leaf():
    p, src, group, _sinks = _grouped(n_sinks=2)
    assert is_data_leaf(src)
    assert is_unconsumed_port(src.output_port('out'))
    assert not is_data_leaf(group)

    dbl = p.add_node(_Double())
    p.create_link(src.output_port('out'), dbl.input_port('in'))
    assert not is_data_leaf(src)
    assert is_data_leaf(dbl)
