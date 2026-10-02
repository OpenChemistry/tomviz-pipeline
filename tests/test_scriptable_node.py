###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""ScriptableNode(definition, kernel): the two-artifact constructor for
scriptable nodes — normalization, dispatch on the definition schema,
strictness, serialization via source capture — and the definition and
script API every scriptable node shares."""

import json

import numpy as np
import pytest

from tomviz_pipeline import (
    DefaultExecutor,
    NodeState,
    Pipeline,
    PortData,
    ScriptableNode,
    definition_schema,
    register_builtins,
)
from tomviz_pipeline.core.state import (
    pipeline_from_state_dict,
    pipeline_to_state_dict,
)
from tomviz_pipeline.dataset import Dataset
from tomviz_pipeline.kernels import SourceKernel, TransformKernel
from tomviz_pipeline.nodes.sources.scriptable import ScriptableSourceNode
from tomviz_pipeline.nodes.transforms.legacy_scriptable import (
    LegacyScriptableTransformNode,
)
from tomviz_pipeline.nodes.transforms.scriptable import ScriptableTransformNode


register_builtins()


class ConstantVolume(SourceKernel):
    """Module-level so inspect.getsource works for round-trip tests."""

    def produce(self, value=1.0, side=2):
        arr = np.full((side, side, side), value, dtype=np.float32)
        return {'volume': Dataset({'Scalars': arr}, active='Scalars')}


class Multiply(TransformKernel):
    def transform(self, inputs, factor=2.0):
        ds = inputs['volume']
        return {'volume': ds.apply_to_each_scalar_array(
            lambda a: a * factor)}


CONSTANT_DEFN = {
    'schemaVersion': 2,
    'name': 'ConstantVolume',
    'outputs': [{'name': 'volume', 'type': 'ImageData'}],
    'parameters': [{'name': 'value', 'type': 'double', 'default': 1.0},
                   {'name': 'side', 'type': 'int', 'default': 2}],
}

MULTIPLY_DEFN = {
    'schemaVersion': 2,
    'name': 'Multiply',
    'inputs':  [{'name': 'volume', 'type': 'ImageData'}],
    'outputs': [{'name': 'volume', 'type': 'ImageData'}],
    'parameters': [{'name': 'factor', 'type': 'double', 'default': 2.0}],
}

MULTIPLY_SCRIPT = '''
from tomviz_pipeline.kernels import TransformKernel

class Multiply(TransformKernel):
    def transform(self, inputs, factor=2.0):
        ds = inputs["volume"]
        return {"volume": ds.apply_to_each_scalar_array(
            lambda a: a * factor)}
'''


def _build_pipeline(source_node, transform_node):
    pipeline = Pipeline()
    source = pipeline.add_node(source_node)
    scale = pipeline.add_node(transform_node)
    pipeline.create_link(source.output_port('volume'),
                         scale.input_port('volume'))
    return pipeline, source, scale


def test_dispatch_and_end_to_end_execution():
    source = ScriptableNode(CONSTANT_DEFN, kernel=ConstantVolume)
    scale = ScriptableNode(MULTIPLY_DEFN, kernel=Multiply)

    assert isinstance(source, ScriptableSourceNode)
    assert isinstance(scale, ScriptableTransformNode)
    assert isinstance(source, ScriptableNode)
    assert isinstance(scale, ScriptableNode)
    assert source.type_name == 'source.python'
    assert scale.type_name == 'transform.python'

    pipeline, source, scale = _build_pipeline(source, scale)
    source.set_parameters(value=21.0)
    scale.set_parameters(factor=2.0)

    assert pipeline.execute().succeeded()
    out = scale.output_port('volume').data().payload
    np.testing.assert_allclose(out.active_scalars,
                               np.full((2, 2, 2), 42.0))


def test_definition_forms(tmp_path):
    # dict, JSON string, and Path all produce equivalent hosts.
    from_dict = ScriptableNode(MULTIPLY_DEFN, kernel=Multiply)
    from_str = ScriptableNode(json.dumps(MULTIPLY_DEFN), kernel=Multiply)
    json_file = tmp_path / 'Multiply.json'
    json_file.write_text(json.dumps(MULTIPLY_DEFN))
    from_path = ScriptableNode(json_file, kernel=Multiply)

    for node in (from_dict, from_str, from_path):
        assert isinstance(node, ScriptableTransformNode)
        assert [p.name for p in node.input_ports()] == ['volume']
        assert [p.name for p in node.output_ports()] == ['volume']
        assert node.parameter('factor') == 2.0
        assert node.label == 'Multiply'


def test_kernel_forms(tmp_path):
    src = ScriptableNode(CONSTANT_DEFN, kernel=ConstantVolume)

    as_script = ScriptableNode(MULTIPLY_DEFN, kernel=MULTIPLY_SCRIPT)
    pipeline, _, scale = _build_pipeline(src, as_script)
    assert pipeline.execute().succeeded()
    out = scale.output_port('volume').data().payload
    np.testing.assert_allclose(out.active_scalars, 2.0)

    script_file = tmp_path / 'Multiply.py'
    script_file.write_text(MULTIPLY_SCRIPT)
    as_path = ScriptableNode(MULTIPLY_DEFN, kernel=script_file)
    assert as_path._backend.script == MULTIPLY_SCRIPT
    assert as_path._backend.kernel_class is None


def test_both_arguments_required():
    with pytest.raises(TypeError, match='definition'):
        ScriptableNode()
    with pytest.raises(TypeError, match='kernel'):
        ScriptableNode(MULTIPLY_DEFN)


def test_invalid_argument_types_rejected():
    with pytest.raises(TypeError, match='definition'):
        ScriptableNode(42, kernel=Multiply)
    with pytest.raises(TypeError, match='kernel'):
        ScriptableNode(MULTIPLY_DEFN, kernel=42)
    with pytest.raises(ValueError, match='JSON'):
        ScriptableNode('not json', kernel=Multiply)
    with pytest.raises(ValueError, match='object'):
        ScriptableNode('[1, 2]', kernel=Multiply)


def test_kernel_shape_mismatch_rejected():
    with pytest.raises(TypeError, match='TransformKernel'):
        ScriptableNode(MULTIPLY_DEFN, kernel=ConstantVolume)
    with pytest.raises(TypeError, match='SourceKernel'):
        ScriptableNode(CONSTANT_DEFN, kernel=Multiply)


def test_bare_host_construction_still_works():
    # The NodeFactory / deserialize path builds hosts with no args.
    assert isinstance(ScriptableSourceNode(), ScriptableSourceNode)
    assert isinstance(ScriptableTransformNode(), ScriptableTransformNode)


def test_state_round_trip_executes_via_captured_script():
    """Class-bound kernels are re-expressed as scripts at serialize
    time; the reloaded pipeline executes without the classes."""
    pipeline, source, scale = _build_pipeline(
        ScriptableNode(CONSTANT_DEFN, kernel=ConstantVolume),
        ScriptableNode(MULTIPLY_DEFN, kernel=Multiply))
    source.set_parameters(value=3.0)
    scale.set_parameters(factor=10.0)

    state = pipeline_to_state_dict(pipeline)
    entry = next(e for e in state['pipeline']['nodes']
                 if e['type'] == 'transform.python')
    assert 'class Multiply(TransformKernel):' in entry['script']
    assert entry['arguments']['factor'] == 10.0

    reloaded = pipeline_from_state_dict(state, reset_to_new=True)
    loaded_scale = reloaded.node_by_id(scale.id)
    assert loaded_scale._backend.kernel_class is None

    assert DefaultExecutor(reloaded).execute() is True
    out = loaded_scale.output_port('volume').data().payload
    np.testing.assert_allclose(out.active_scalars,
                               np.full((2, 2, 2), 30.0))


def test_uncapturable_kernel_serializes_empty_and_fails_shim(tmp_path):
    """A class with no retrievable source (exec-defined) still runs
    in-process, but serializes with an empty script and is rejected by
    the external shim writer before any subprocess is spawned."""
    from tomviz_pipeline import ExternalNodeExecutor, PortData
    from tomviz_pipeline.core import SourceNode

    namespace = {}
    exec('from tomviz_pipeline.kernels import TransformKernel\n'
         'class Ephemeral(TransformKernel):\n'
         '    def transform(self, inputs, factor=2.0):\n'
         "        ds = inputs['volume']\n"
         "        return {'volume': ds.apply_to_each_scalar_array(\n"
         '            lambda a: a * factor)}\n', namespace)
    node = ScriptableNode(MULTIPLY_DEFN, kernel=namespace['Ephemeral'])

    # In-process execution works.
    pipeline, _, scale = _build_pipeline(
        ScriptableNode(CONSTANT_DEFN, kernel=ConstantVolume), node)
    assert pipeline.execute().succeeded()

    # Serialization cannot capture the source.
    assert node.serialize()['script'] == ''

    # The shim writer refuses early (needs a linked input with data).
    feed = SourceNode()
    out = feed.add_output('volume', 'ImageData')
    ds = Dataset({'a': np.zeros((2, 2, 2), dtype=np.float32, order='F')},
                 'a')
    ds.spacing = [1.0, 1.0, 1.0]
    out.set_data(PortData(ds, 'ImageData'))
    shim_pipeline = Pipeline()
    shim_pipeline.add_node(feed)
    detached = ScriptableNode(MULTIPLY_DEFN, kernel=namespace['Ephemeral'])
    shim_pipeline.add_node(detached)
    shim_pipeline.create_link(feed.output_port('volume'),
                              detached.input_port('volume'))
    executor = ExternalNodeExecutor(env_path='/nonexistent')
    assert executor._write_shim_tvh5(detached, tmp_path) is None
    assert not (tmp_path / 'shim.tvh5').exists()


def test_definition_is_pure_interface():
    """No compute-related keys are recognized in the definition — the
    kernel argument is the only compute channel."""
    defn = dict(MULTIPLY_DEFN)
    defn['kernel'] = 'mylab.kernels:DoesNotExist'
    node = ScriptableNode(defn, kernel=Multiply)
    # The unknown key rides along in the description untouched but has
    # no effect on execution — the bound class runs.
    pipeline, _, scale = _build_pipeline(
        ScriptableNode(CONSTANT_DEFN, kernel=ConstantVolume), node)
    assert pipeline.execute().succeeded()


# ---- definition schema ------------------------------------------------------

OFFSET_V1_DEFN = {
    'name': 'Offset',
    'label': 'Offset',
    'parameters': [{'name': 'offset', 'type': 'double', 'default': 5.0}],
}

OFFSET_V1_SCRIPT = '''
def transform(dataset, offset=5.0):
    dataset.active_scalars = offset - dataset.active_scalars
'''


def test_definition_schema():
    assert definition_schema(MULTIPLY_DEFN) == 2
    assert definition_schema({'schemaVersion': 1, 'outputs': []}) == 1
    assert definition_schema(OFFSET_V1_DEFN) == 1
    assert definition_schema({}) == 1
    with pytest.raises(ValueError, match='schemaVersion'):
        definition_schema({'schemaVersion': 3})


def _unversioned(defn):
    return {key: value for key, value in defn.items()
            if key != 'schemaVersion'}


def test_declared_ports_without_schema_version_read_as_v2_with_warning():
    with pytest.warns(DeprecationWarning, match='schemaVersion') as caught:
        assert definition_schema(_unversioned(MULTIPLY_DEFN)) == 2
    assert caught[0].filename == __file__

    with pytest.warns(DeprecationWarning, match='schemaVersion') as caught:
        node = ScriptableNode(_unversioned(MULTIPLY_DEFN), kernel=Multiply)
    assert isinstance(node, ScriptableTransformNode)
    assert caught[0].filename == __file__

    with pytest.warns(DeprecationWarning, match='schemaVersion') as caught:
        node.reconfigure_description(json.dumps(_unversioned(MULTIPLY_DEFN)))
    assert caught[0].filename == __file__


def test_v1_definition_builds_a_legacy_node():
    node = ScriptableNode(OFFSET_V1_DEFN, kernel=OFFSET_V1_SCRIPT)
    assert isinstance(node, LegacyScriptableTransformNode)
    assert isinstance(node, ScriptableNode)
    assert node.type_name == 'transform.legacyPython'
    assert node.label == 'Offset'
    assert [p.name for p in node.input_ports()] == ['volume']
    assert [p.name for p in node.output_ports()] == ['volume']

    pipeline, _, offset = _build_pipeline(
        ScriptableNode(CONSTANT_DEFN, kernel=ConstantVolume), node)
    assert pipeline.execute().succeeded()
    out = offset.output_port('volume').data().payload
    np.testing.assert_allclose(out.active_scalars, 4.0)


def test_v1_definition_rejects_a_kernel_class():
    with pytest.raises(TypeError, match='schema v1'):
        ScriptableNode(OFFSET_V1_DEFN, kernel=Multiply)


def test_initial_parameters_replace_the_defaults_quietly():
    applied = []
    node = ScriptableNode(MULTIPLY_DEFN, kernel=Multiply,
                          parameters={'factor': 5.0})
    node.parameters_applied.connect(lambda *args: applied.append(args))
    assert node.parameter('factor') == 5.0
    assert node.state == NodeState.New
    assert applied == []

    legacy = ScriptableNode(OFFSET_V1_DEFN, kernel=OFFSET_V1_SCRIPT,
                            parameters={'offset': 1.0})
    assert legacy.parameter('offset') == 1.0


def test_initial_parameters_must_be_declared():
    with pytest.raises(ValueError, match="'factr'"):
        ScriptableNode(MULTIPLY_DEFN, kernel=Multiply,
                       parameters={'factr': 5.0})
    # A v1 dataset parameter is an input port, not a value.
    with_dataset = dict(OFFSET_V1_DEFN, parameters=[
        *OFFSET_V1_DEFN['parameters'], {'name': 'other', 'type': 'dataset'}])
    with pytest.raises(ValueError, match="'other'"):
        ScriptableNode(with_dataset, kernel=OFFSET_V1_SCRIPT,
                       parameters={'other': None})


def test_v1_node_round_trips_through_state():
    pipeline, _, offset = _build_pipeline(
        ScriptableNode(CONSTANT_DEFN, kernel=ConstantVolume),
        ScriptableNode(OFFSET_V1_DEFN, kernel=OFFSET_V1_SCRIPT))
    offset.set_parameters(offset=10.0)

    reloaded = pipeline_from_state_dict(pipeline_to_state_dict(pipeline))
    node = reloaded.node_by_id(offset.id)
    assert isinstance(node, LegacyScriptableTransformNode)
    assert node.script == OFFSET_V1_SCRIPT
    assert json.loads(node.json_description) == OFFSET_V1_DEFN
    assert node.parameter('offset') == 10.0


# ---- definition and script accessors ---------------------------------------

def test_accessors_on_every_node_class():
    nodes = [
        (ScriptableNode(CONSTANT_DEFN, kernel=ConstantVolume), ''),
        (ScriptableNode(MULTIPLY_DEFN, kernel=MULTIPLY_SCRIPT),
         MULTIPLY_SCRIPT),
        (ScriptableNode(OFFSET_V1_DEFN, kernel=OFFSET_V1_SCRIPT),
         OFFSET_V1_SCRIPT),
    ]
    for node, script in nodes:
        assert node.script == script
        assert json.loads(node.json_description)['name'] == \
            node.label


def test_set_json_description_does_not_duplicate_ports():
    node = ScriptableTransformNode()
    node.set_json_description(json.dumps(MULTIPLY_DEFN))
    node.set_json_description(json.dumps(MULTIPLY_DEFN))
    assert [p.name for p in node.input_ports()] == ['volume']
    assert [p.name for p in node.output_ports()] == ['volume']


def test_script_setter_replaces_a_bound_class_and_marks_stale():
    pipeline, _, scale = _build_pipeline(
        ScriptableNode(CONSTANT_DEFN, kernel=ConstantVolume),
        ScriptableNode(MULTIPLY_DEFN, kernel=Multiply))
    assert pipeline.execute().succeeded()
    assert scale.state == NodeState.Current

    scale.script = MULTIPLY_SCRIPT.replace('a * factor', 'a * factor + 1')
    assert scale._backend.kernel_class is None
    assert scale.state == NodeState.Stale

    assert pipeline.execute().succeeded()
    out = scale.output_port('volume').data().payload
    np.testing.assert_allclose(out.active_scalars, 3.0)


# ---- reconfigure_description -----------------------------------------------

def _options(*values):
    return [{value.upper(): value} for value in values]


# Takes every parameter the edited definitions declare.
EDITABLE_SCRIPT = MULTIPLY_SCRIPT.replace('factor=2.0)', 'factor=2.0, **_)')

EDITABLE_DEFN = dict(MULTIPLY_DEFN, label='Editable', parameters=[
    {'name': 'factor', 'type': 'double', 'default': 2.0},
    {'name': 'mode', 'type': 'enumeration', 'default': 0,
     'options': _options('a', 'b')},
    {'name': 'count', 'type': 'int', 'default': 1},
    {'name': 'dropped', 'type': 'int', 'default': 1},
])


def test_reconfigure_carries_compatible_values_over():
    pipeline, _, scale = _build_pipeline(
        ScriptableNode(CONSTANT_DEFN, kernel=ConstantVolume),
        ScriptableNode(EDITABLE_DEFN, kernel=EDITABLE_SCRIPT))
    scale.set_parameters(factor=3.0, mode='b', count=5, dropped=7)
    scale.label = 'Renamed'
    assert pipeline.execute().succeeded()

    edited = dict(EDITABLE_DEFN, label='Ignored', parameters=[
        {'name': 'factor', 'type': 'double', 'default': 2.0},
        {'name': 'mode', 'type': 'enumeration', 'default': 0,
         'options': _options('a', 'c')},
        {'name': 'count', 'type': 'double', 'default': 1.5},
        {'name': 'added', 'type': 'string', 'default': 'new'},
    ])
    reset = scale.reconfigure_description(json.dumps(edited))

    assert sorted(reset) == ['count', 'mode']
    assert dict(scale.parameters) == {
        'factor': 3.0, 'mode': 'a', 'count': 1.5, 'added': 'new'}
    assert json.loads(scale.json_description) == edited
    assert scale.label == 'Renamed'
    assert [p.name for p in scale.output_ports()] == ['volume']
    assert scale.state == NodeState.Stale


@pytest.mark.parametrize('change, message', [
    ({'outputs': [{'name': 'other', 'type': 'ImageData'}]}, 'outputs'),
    ({'inputs': []}, 'inputs'),
    ({'schemaVersion': 1}, 'schema'),
])
def test_reconfigure_rejects_port_and_schema_changes(change, message):
    node = ScriptableNode(EDITABLE_DEFN, kernel=MULTIPLY_SCRIPT)
    node.set_parameters(factor=3.0)
    before = node.json_description

    with pytest.raises(ValueError, match=message):
        node.reconfigure_description(json.dumps(dict(EDITABLE_DEFN,
                                                     **change)))
    assert node.json_description == before
    assert node.parameter('factor') == 3.0


@pytest.mark.parametrize('text, message', [
    ('{}', 'emptied'),
    ('not json', 'JSON'),
    ('[1, 2]', 'object'),
])
def test_reconfigure_rejects_invalid_definitions(text, message):
    node = ScriptableNode(EDITABLE_DEFN, kernel=MULTIPLY_SCRIPT)
    with pytest.raises(ValueError, match=message):
        node.reconfigure_description(text)


def test_reconfigure_v1_node():
    node = ScriptableNode(OFFSET_V1_DEFN, kernel=OFFSET_V1_SCRIPT)
    node.set_parameters(offset=8.0)

    edited = dict(OFFSET_V1_DEFN, parameters=[
        {'name': 'offset', 'type': 'double', 'default': 5.0},
        {'name': 'scale', 'type': 'double', 'default': 1.0},
    ])
    assert node.reconfigure_description(json.dumps(edited)) == []
    assert dict(node.parameters) == {'offset': 8.0, 'scale': 1.0}

    with pytest.raises(ValueError, match='dataset parameters'):
        node.reconfigure_description(json.dumps(dict(edited, parameters=[
            {'name': 'other', 'type': 'dataset'}])))
    with pytest.raises(ValueError, match='results'):
        node.reconfigure_description(json.dumps(dict(edited, results=[
            {'name': 'table', 'type': 'table'}])))


# ---- legacy kernels ---------------------------------------------------------

def test_legacy_cancel_support_follows_the_script_base_class():
    node = LegacyScriptableTransformNode()
    backend = node._backend

    node.script = 'class K(tomviz.operators.CompletableOperator): pass'
    assert (backend.supports_cancel, backend.supports_complete) == \
        (True, True)
    node.script = 'class K(tomviz.operators.CancelableOperator): pass'
    assert (backend.supports_cancel, backend.supports_complete) == \
        (True, False)
    node.script = OFFSET_V1_SCRIPT
    assert (backend.supports_cancel, backend.supports_complete) == \
        (False, False)


def test_legacy_script_that_fails_to_load_returns_no_outputs():
    node = ScriptableNode(OFFSET_V1_DEFN, kernel='def transform(:\n')
    data = Dataset({'Scalars': np.zeros((2, 2, 2), dtype=np.float32)},
                   active='Scalars')
    assert node.transform({'volume': PortData(data, 'ImageData')}) == {}


# ---- former names -----------------------------------------------------------

def test_former_names_are_aliases():
    import tomviz_pipeline
    from tomviz_pipeline import kernels, operators
    from tomviz_pipeline.nodes.kernel_backends import KernelBackend
    from tomviz_pipeline.nodes.python_backend import PythonNodeBackend
    from tomviz_pipeline.nodes.python_node import PythonNode
    from tomviz_pipeline.nodes.sources.python_source import PythonSource
    from tomviz_pipeline.nodes.transforms.legacy_python import (
        LegacyPythonTransform,
    )
    from tomviz_pipeline.nodes.transforms.python_transform import (
        PythonTransform,
    )

    assert tomviz_pipeline.PythonNode is ScriptableNode
    assert PythonNode is ScriptableNode
    assert PythonSource is ScriptableSourceNode
    assert PythonTransform is ScriptableTransformNode
    assert LegacyPythonTransform is LegacyScriptableTransformNode
    assert PythonNodeBackend is KernelBackend
    assert operators.Progress is kernels.Progress


def test_execution_context_keeps_its_former_names():
    from tomviz_pipeline._internal import (
        ExecutionContext,
        OperatorWrapper,
        attach_execution_context,
    )
    from tomviz_pipeline.operators import CompletableOperator

    assert OperatorWrapper is ExecutionContext

    # The desktop application sets the former attribute name on kernels
    # and v1 operators; it reaches the context their base classes read.
    for instance in (Multiply(), CompletableOperator()):
        context = ExecutionContext()
        instance._operator_wrapper = context
        assert instance._execution_context is context
        assert instance._operator_wrapper is context
        assert instance.completed is False
        context._completed = True
        assert instance.completed is True

    # A class from an older tomviz install reads only the former name.
    class Foreign:
        pass

    foreign = Foreign()
    context = ExecutionContext()
    attach_execution_context(foreign, context)
    assert foreign._operator_wrapper is context
