###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Node parameters: the private store, set_parameters() staleness
tracking, the parameters_applied signal, Pipeline.auto_execute wiring,
and parameter round-trips through schema-v2 serialization."""

import numpy as np
import pytest

from tomviz_pipeline import (
    NodeState,
    Pipeline,
    PortData,
    SourceNode,
    TransformNode,
    pipeline_from_state_dict,
    pipeline_to_state_dict,
    register_builtins,
)


class _Const(SourceNode):
    type_name = 'parmtest.const'

    def __init__(self, value=1):
        super().__init__()
        self._parameters['value'] = value
        self.add_output('out', 'ImageData')

    def execute(self):
        self.output_port('out').set_data(
            PortData(self.parameter('value'), 'ImageData'))
        return True


class _Scale(TransformNode):
    type_name = 'parmtest.scale'

    def __init__(self, factor=1):
        super().__init__()
        self._parameters['factor'] = factor
        self.add_input('in', 'ImageData')
        self.add_output('out', 'ImageData')

    def transform(self, inputs):
        return {'out': PortData(
            inputs['in'].payload * self.parameter('factor'), 'ImageData')}


def _chain():
    p = Pipeline()
    src = _Const(3)
    t = _Scale(2)
    p.add_node(src)
    p.add_node(t)
    p.create_link(src.output_port('out'), t.input_port('in'))
    return p, src, t


# ---- core API ------------------------------------------------------------


def test_set_parameters_updates_store_and_marks_stale():
    p, src, t = _chain()
    p.execute()
    assert t.state == NodeState.Current

    t.set_parameters(factor=5)
    assert t.parameter('factor') == 5
    assert t.parameters['factor'] == 5
    assert t.state == NodeState.Stale

    future = p.execute()
    assert future.succeeded()
    assert t.output_port('out').data().payload == 15


def test_set_parameters_cascades_downstream():
    p, src, t = _chain()
    t2 = _Scale(10)
    p.add_node(t2)
    p.create_link(t.output_port('out'), t2.input_port('in'))
    p.execute()
    assert t2.state == NodeState.Current

    t.set_parameters(factor=4)
    assert t.state == NodeState.Stale
    assert t2.state == NodeState.Stale
    # The source is untouched.
    assert src.state == NodeState.Current


def test_parameters_applied_signal():
    t = _Scale(2)
    seen = []
    t.parameters_applied.connect(lambda node, changed: seen.append(changed))
    t.set_parameters(factor=7, other='x')
    assert seen == [{'factor': 7, 'other': 'x'}]
    t.set_parameters()  # no-op: nothing emitted
    assert len(seen) == 1


def test_parameters_view_is_read_only():
    t = _Scale(2)
    with pytest.raises(TypeError):
        t.parameters['factor'] = 9


def test_parameter_getter_default():
    t = _Scale(2)
    assert t.parameter('missing') is None
    assert t.parameter('missing', 42) == 42


def test_parameter_names_cannot_clobber_node_internals():
    t = _Scale(2)
    t.label = 'my label'
    t.set_parameters(label='not a label', state='not a state',
                     progress='not progress')
    # Parameters live in their own store; node attributes are untouched.
    assert t.label == 'my label'
    assert t.state == NodeState.New
    assert t.progress is None
    assert t.parameter('label') == 'not a label'


# ---- pipeline auto-execute -------------------------------------------------


def test_auto_execute_off_by_default():
    p, src, t = _chain()
    p.execute()
    t.set_parameters(factor=5)
    # No auto re-run: output still reflects the old factor.
    assert t.state == NodeState.Stale
    assert t.output_port('out').data().payload == 6


def test_auto_execute_reruns_on_parameters_applied():
    p, src, t = _chain()
    p.execute()
    p.auto_execute = True
    t.set_parameters(factor=5)
    assert t.state == NodeState.Current
    assert t.output_port('out').data().payload == 15


def test_removed_node_no_longer_triggers_auto_execute():
    p, src, t = _chain()
    p.execute()
    p.auto_execute = True
    runs = []
    p.execution_started.connect(lambda f: runs.append(f))
    p.remove_node(t)
    t.set_parameters(factor=9)
    assert runs == []


# ---- serialization round-trips ---------------------------------------------


def test_builtin_transform_parameters_round_trip():
    register_builtins()
    from tomviz_pipeline.core import NodeFactory

    crop = NodeFactory.create('transform.crop')
    crop.set_parameters(bounds=[1, 4, 0, 5, 2, 3])
    entry = crop.serialize()
    assert entry['bounds'] == [1, 4, 0, 5, 2, 3]

    p = Pipeline()
    p.add_node(crop)
    restored = pipeline_from_state_dict(pipeline_to_state_dict(p))
    rcrop = restored.node_by_id(crop.id)
    assert rcrop.parameter('bounds') == [1, 4, 0, 5, 2, 3]


def test_threshold_parameters_drive_transform():
    register_builtins()
    from tomviz_pipeline.core import NodeFactory
    from tomviz_pipeline.dataset import Dataset

    threshold = NodeFactory.create('transform.threshold')
    threshold.set_parameters(minValue=2.0, maxValue=3.0)

    data = Dataset(
        {'s': np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)}, 's')
    data.spacing = [1.0, 1.0, 1.0]
    outputs = threshold.transform({'volume': PortData(data, 'ImageData')})
    mask = outputs['mask'].payload.active_scalars
    assert mask.tolist() == [[[0.0, 1.0], [1.0, 0.0]]]


def test_v2_python_host_routes_parameters_to_backend():
    from tomviz_pipeline.nodes.transforms.python_transform import (
        PythonTransform,
    )
    transform = PythonTransform()
    transform.set_parameters(factor=3.0)
    assert transform._backend.parameters['factor'] == 3.0
    assert transform.parameters['factor'] == 3.0
    assert transform.state == NodeState.New  # New stays New on mark_stale


def test_legacy_host_parameters_serialize_as_arguments():
    from tomviz_pipeline.nodes.transforms.legacy_python import (
        LegacyPythonTransform,
    )
    node = LegacyPythonTransform()
    node.set_parameters(alpha=1.5)
    assert node.serialize().get('arguments') == {'alpha': 1.5}
