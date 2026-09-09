###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Effective port types: outputs declared 'ImageData' take the concrete
type (TiltSeries, Volume, ...) of what feeds their driving input,
propagated downstream on every link change and run-time type change;
links know whether their types still match."""

import pytest

from tomviz_pipeline import (
    NodeFactory,
    Pipeline,
    PortData,
    SinkGroupNode,
    SinkNode,
    SourceNode,
    TransformNode,
    is_port_type_compatible,
    pipeline_from_state_dict,
    pipeline_to_state_dict,
    register_builtins,
)


class _Source(SourceNode):
    type_name = 'titest.source'

    def __init__(self, port_type='ImageData'):
        super().__init__()
        self.add_output('out', port_type)

    def execute(self):
        port = self.output_port('out')
        port.set_data(PortData(1, port.port_type))
        return True


class _Reader(_Source):
    """Types its output only when it runs, like the file reader."""

    type_name = 'titest.reader'

    def __init__(self):
        super().__init__('ImageData')
        self.found = 'TiltSeries'

    def execute(self):
        self.output_port('out').port_type = self.found
        return super().execute()


class _Filter(TransformNode):
    type_name = 'titest.filter'

    def __init__(self, outputs=('out:ImageData',)):
        super().__init__()
        self.add_input('in', 'ImageData')
        for spec in outputs:
            name, port_type = spec.split(':')
            self.add_output(name, port_type)

    def transform(self, inputs):
        return {p.name: PortData(inputs['in'].payload, p.port_type)
                for p in self.output_ports()}


class _Merge(TransformNode):
    type_name = 'titest.merge'

    def __init__(self):
        super().__init__()
        self.add_input('a', 'ImageData')
        self.add_input('b', 'ImageData')
        self.add_output('out', 'ImageData')

    def transform(self, inputs):
        return {'out': PortData(inputs['a'].payload, 'ImageData')}


class _Sink(SinkNode):
    type_name = 'titest.sink'

    def __init__(self, accepted=('ImageData',)):
        super().__init__()
        self.add_input('in', list(accepted))


@pytest.fixture(autouse=True)
def _registry():
    register_builtins()
    for cls in (_Source, _Reader, _Filter, _Merge, _Sink):
        NodeFactory.register(cls.type_name, cls)


def test_compatibility_uses_the_image_base_type():
    assert is_port_type_compatible('TiltSeries', ['ImageData'])
    assert is_port_type_compatible('Table', ['Table'])
    assert not is_port_type_compatible('Table', ['ImageData'])
    assert not is_port_type_compatible('ImageData', ['Volume'])


def test_declared_and_effective_types():
    port = _Source('Volume').output_port('out')
    assert (port.declared_type, port.port_type) == ('Volume', 'Volume')
    seen = []
    port.effective_type_changed.connect(lambda p, t: seen.append(t))
    port.port_type = 'LabelMap'  # sets both, as setDeclaredType does
    assert (port.declared_type, port.port_type) == ('LabelMap', 'LabelMap')
    assert seen == ['LabelMap']


def test_a_transform_takes_the_type_of_its_input():
    p = Pipeline()
    src = p.add_node(_Source('TiltSeries'))
    flt = p.add_node(_Filter())
    out = flt.output_port('out')
    assert out.port_type == 'ImageData'

    link = p.create_link(src.output_port('out'), flt.input_port('in'))
    assert out.port_type == 'TiltSeries'
    assert out.declared_type == 'ImageData'

    p.remove_link(link)
    assert out.port_type == 'ImageData'


def test_inference_walks_the_whole_chain_and_stops_at_concrete_types():
    p = Pipeline()
    src = p.add_node(_Source('Volume'))
    a = p.add_node(_Filter())
    b = p.add_node(_Filter(('out:LabelMap', 'extra:ImageData')))
    c = p.add_node(_Filter())
    p.create_link(src.output_port('out'), a.input_port('in'))
    p.create_link(a.output_port('out'), b.input_port('in'))
    p.create_link(b.output_port('out'), c.input_port('in'))

    assert a.output_port('out').port_type == 'Volume'
    assert b.output_port('out').port_type == 'LabelMap'  # declared, kept
    assert b.output_port('extra').port_type == 'Volume'  # inferred
    assert c.output_port('out').port_type == 'LabelMap'

    # Re-pointing the head of the chain re-types everything below it.
    other = p.add_node(_Source('TiltSeries'))
    p.create_link(other.output_port('out'), a.input_port('in'))
    assert a.output_port('out').port_type == 'TiltSeries'
    assert b.output_port('extra').port_type == 'TiltSeries'
    assert c.output_port('out').port_type == 'LabelMap'


def test_the_driving_input_can_be_named():
    p = Pipeline()
    a = p.add_node(_Source('TiltSeries'))
    b = p.add_node(_Source('Volume'))
    merge = p.add_node(_Merge())
    p.create_link(a.output_port('out'), merge.input_port('a'))
    p.create_link(b.output_port('out'), merge.input_port('b'))
    assert merge.output_port('out').port_type == 'TiltSeries'  # first input

    merge.set_type_inference_source('out', 'b')
    p.create_link(b.output_port('out'), merge.input_port('b'))  # re-link
    assert merge.output_port('out').port_type == 'Volume'


def test_a_run_time_type_change_propagates_downstream():
    p = Pipeline()
    reader = p.add_node(_Reader())
    flt = p.add_node(_Filter())
    sink = p.add_node(_Sink())
    p.create_link(reader.output_port('out'), flt.input_port('in'))
    p.create_link(flt.output_port('out'), sink.input_port('in'))
    assert flt.output_port('out').port_type == 'ImageData'

    seen = []
    flt.output_port('out').effective_type_changed.connect(
        lambda port, t: seen.append(t))
    assert p.execute().succeeded()
    assert reader.output_port('out').port_type == 'TiltSeries'
    assert flt.output_port('out').port_type == 'TiltSeries'
    assert seen == ['TiltSeries']
    # The transform stamped the inferred type on its data.
    assert flt.output_port('out').data().port_type == 'TiltSeries'

    reader.found = 'Volume'
    reader.mark_stale()
    assert p.execute().succeeded()
    assert flt.output_port('out').port_type == 'Volume'


def test_a_sink_group_passthrough_follows_the_source_type():
    p = Pipeline()
    src = p.add_node(_Source('LabelMap'))
    group = p.add_node(SinkGroupNode())
    out = group.add_passthrough('out', 'ImageData')
    assert out.port_type == 'ImageData'
    p.create_link(src.output_port('out'), group.input_port('out'))
    assert out.port_type == 'LabelMap'
    assert out.declared_type == 'ImageData'
    src.output_port('out').port_type = 'Volume'
    assert out.port_type == 'Volume'


def test_links_track_their_validity():
    p = Pipeline()
    src = p.add_node(_Source('Table'))
    flt = p.add_node(_Filter())
    changes = []
    p.link_validity_changed.connect(
        lambda link, valid: changes.append(valid))
    link = p.create_link(src.output_port('out'), flt.input_port('in'))
    assert link.valid is False  # a table into an image input
    assert changes == []  # invalid from the start: nothing changed

    src.output_port('out').port_type = 'Volume'
    assert link.valid is True
    assert changes == [True]
    src.output_port('out').port_type = 'Table'
    assert link.valid is False
    assert changes == [True, False]


def test_state_files_keep_the_declared_type_and_infer_again_on_load():
    p = Pipeline()
    src = p.add_node(_Source('TiltSeries'))
    flt = p.add_node(_Filter())
    p.create_link(src.output_port('out'), flt.input_port('in'))
    state = pipeline_to_state_dict(p)
    entry = next(n for n in state['pipeline']['nodes'] if n['id'] == flt.id)
    assert entry['outputPorts']['out']['type'] == 'ImageData'

    p2 = pipeline_from_state_dict(state)
    out2 = p2.node_by_id(flt.id).output_port('out')
    assert (out2.declared_type, out2.port_type) == ('ImageData', 'TiltSeries')
