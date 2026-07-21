###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""PythonNode(definition, kernel): the two-artifact constructor for
schema-v2 python nodes — normalization, dispatch, strictness, and
serialization via source capture."""

import json

import numpy as np
import pytest

from tomviz_pipeline import (
    DefaultExecutor,
    Pipeline,
    PythonNode,
    register_builtins,
)
from tomviz_pipeline.core.state import (
    pipeline_from_state_dict,
    pipeline_to_state_dict,
)
from tomviz_pipeline.dataset import Dataset
from tomviz_pipeline.kernels import SourceKernel, TransformKernel
from tomviz_pipeline.nodes.sources.python_source import PythonSource
from tomviz_pipeline.nodes.transforms.python_transform import PythonTransform


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
    'name': 'ConstantVolume',
    'outputs': [{'name': 'volume', 'type': 'ImageData'}],
    'parameters': [{'name': 'value', 'type': 'double', 'default': 1.0},
                   {'name': 'side', 'type': 'int', 'default': 2}],
}

MULTIPLY_DEFN = {
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
    source = PythonNode(CONSTANT_DEFN, kernel=ConstantVolume)
    scale = PythonNode(MULTIPLY_DEFN, kernel=Multiply)

    assert isinstance(source, PythonSource)
    assert isinstance(scale, PythonTransform)
    assert isinstance(source, PythonNode)
    assert isinstance(scale, PythonNode)
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
    from_dict = PythonNode(MULTIPLY_DEFN, kernel=Multiply)
    from_str = PythonNode(json.dumps(MULTIPLY_DEFN), kernel=Multiply)
    json_file = tmp_path / 'Multiply.json'
    json_file.write_text(json.dumps(MULTIPLY_DEFN))
    from_path = PythonNode(json_file, kernel=Multiply)

    for node in (from_dict, from_str, from_path):
        assert isinstance(node, PythonTransform)
        assert [p.name for p in node.input_ports()] == ['volume']
        assert [p.name for p in node.output_ports()] == ['volume']
        assert node.parameter('factor') == 2.0
        assert node.label == 'Multiply'


def test_kernel_forms(tmp_path):
    src = PythonNode(CONSTANT_DEFN, kernel=ConstantVolume)

    as_script = PythonNode(MULTIPLY_DEFN, kernel=MULTIPLY_SCRIPT)
    pipeline, _, scale = _build_pipeline(src, as_script)
    assert pipeline.execute().succeeded()
    out = scale.output_port('volume').data().payload
    np.testing.assert_allclose(out.active_scalars, 2.0)

    script_file = tmp_path / 'Multiply.py'
    script_file.write_text(MULTIPLY_SCRIPT)
    as_path = PythonNode(MULTIPLY_DEFN, kernel=script_file)
    assert as_path._backend.script == MULTIPLY_SCRIPT
    assert as_path._backend.kernel_class is None


def test_both_arguments_required():
    with pytest.raises(TypeError, match='definition'):
        PythonNode()
    with pytest.raises(TypeError, match='kernel'):
        PythonNode(MULTIPLY_DEFN)


def test_invalid_argument_types_rejected():
    with pytest.raises(TypeError, match='definition'):
        PythonNode(42, kernel=Multiply)
    with pytest.raises(TypeError, match='kernel'):
        PythonNode(MULTIPLY_DEFN, kernel=42)
    with pytest.raises(ValueError, match='JSON'):
        PythonNode('not json', kernel=Multiply)
    with pytest.raises(ValueError, match='object'):
        PythonNode('[1, 2]', kernel=Multiply)


def test_kernel_shape_mismatch_rejected():
    with pytest.raises(TypeError, match='TransformKernel'):
        PythonNode(MULTIPLY_DEFN, kernel=ConstantVolume)
    with pytest.raises(TypeError, match='SourceKernel'):
        PythonNode(CONSTANT_DEFN, kernel=Multiply)


def test_bare_host_construction_still_works():
    # The NodeFactory / deserialize path builds hosts with no args.
    assert isinstance(PythonSource(), PythonSource)
    assert isinstance(PythonTransform(), PythonTransform)


def test_state_round_trip_executes_via_captured_script():
    """Class-bound kernels are re-expressed as scripts at serialize
    time; the reloaded pipeline executes without the classes."""
    pipeline, source, scale = _build_pipeline(
        PythonNode(CONSTANT_DEFN, kernel=ConstantVolume),
        PythonNode(MULTIPLY_DEFN, kernel=Multiply))
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
    node = PythonNode(MULTIPLY_DEFN, kernel=namespace['Ephemeral'])

    # In-process execution works.
    pipeline, _, scale = _build_pipeline(
        PythonNode(CONSTANT_DEFN, kernel=ConstantVolume), node)
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
    detached = PythonNode(MULTIPLY_DEFN, kernel=namespace['Ephemeral'])
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
    node = PythonNode(defn, kernel=Multiply)
    # The unknown key rides along in the description untouched but has
    # no effect on execution — the bound class runs.
    pipeline, _, scale = _build_pipeline(
        PythonNode(CONSTANT_DEFN, kernel=ConstantVolume), node)
    assert pipeline.execute().succeeded()
