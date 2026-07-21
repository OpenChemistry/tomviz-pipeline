###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Compatibility shim for the pre-extraction `tomviz_pipeline_env` key:
operator JSON descriptions carrying it get an ExternalNodeExecutor at
deserialize time, with the modern per-node `executor` block taking
precedence, and the external shim writer stripping the key so a child
process never recurses."""

import json

import numpy as np

from tomviz_pipeline import PortData, register_builtins
from tomviz_pipeline.core import NodeFactory, Pipeline, SourceNode
from tomviz_pipeline.core.state import pipeline_from_state_dict
from tomviz_pipeline.dataset import Dataset
from tomviz_pipeline.external import ExternalNodeExecutor


register_builtins()


def _legacy_entry(env_path=None, executor=None):
    description = {'name': 'AddOne', 'label': 'Add One'}
    if env_path is not None:
        description['tomviz_pipeline_env'] = env_path
    entry = {
        'label': 'Add One',
        'description': json.dumps(description),
        'script': 'def transform(dataset):\n    pass\n',
    }
    if executor is not None:
        entry['executor'] = executor
    return entry


def _v2_description(env_path=None):
    description = {
        'schemaVersion': 2,
        'name': 'Multiply',
        'inputs': [{'name': 'volume', 'type': 'ImageData'}],
        'outputs': [{'name': 'volume', 'type': 'ImageData'}],
    }
    if env_path is not None:
        description['tomviz_pipeline_env'] = env_path
    return description


def test_legacy_description_env_promotes_external_executor():
    node = NodeFactory.create('transform.legacyPython')
    assert node.deserialize(_legacy_entry(env_path='/some/env'))
    assert isinstance(node.node_executor, ExternalNodeExecutor)
    assert node.node_executor.env_path == '/some/env'


def test_legacy_description_without_env_stays_in_process():
    node = NodeFactory.create('transform.legacyPython')
    assert node.deserialize(_legacy_entry())
    assert node.node_executor is None


def test_explicit_executor_block_wins_over_description_env():
    entry = _legacy_entry(env_path='/legacy/env',
                          executor={'type': 'external',
                                    'envPath': '/explicit/env'})
    node = NodeFactory.create('transform.legacyPython')
    assert node.deserialize(entry)
    # deserialize itself must not promote; the loader restores the
    # explicit block afterwards.
    assert node.node_executor is None

    entry = dict(entry, id=1, type='transform.legacyPython')
    state = {'schemaVersion': 2,
             'pipeline': {'nodes': [entry], 'links': []}}
    pipeline = pipeline_from_state_dict(state)
    loaded = pipeline.node_by_id(1)
    assert isinstance(loaded.node_executor, ExternalNodeExecutor)
    assert loaded.node_executor.env_path == '/explicit/env'


def test_v2_transform_and_source_promote_description_env():
    transform = NodeFactory.create('transform.python')
    assert transform.deserialize(
        {'description': json.dumps(_v2_description('/v2/env'))})
    assert isinstance(transform.node_executor, ExternalNodeExecutor)
    assert transform.node_executor.env_path == '/v2/env'

    description = _v2_description('/v2/env')
    description['inputs'] = []
    source = NodeFactory.create('source.python')
    assert source.deserialize({'description': json.dumps(description)})
    assert isinstance(source.node_executor, ExternalNodeExecutor)
    assert source.node_executor.env_path == '/v2/env'


def test_shim_writer_strips_env_key_and_executor(tmp_path):
    """The child process must never re-promote: the shim's clone entry
    carries neither an executor block nor the description env key."""
    pipeline = Pipeline()
    source = SourceNode()
    out = source.add_output('volume', 'ImageData')
    ds = Dataset({'a': np.zeros((2, 2, 2), dtype=np.float32, order='F')},
                 'a')
    ds.spacing = [1.0, 1.0, 1.0]
    out.set_data(PortData(ds, 'ImageData'))

    node = NodeFactory.create('transform.legacyPython')
    assert node.deserialize(_legacy_entry(env_path='/some/env'))
    pipeline.add_node(source)
    pipeline.add_node(node)
    pipeline.create_link(source.output_port('volume'),
                         node.input_port('volume'))

    shim_path = node.node_executor._write_shim_tvh5(node, tmp_path)
    assert shim_path is not None

    from tomviz_pipeline.state import read_state_json
    state = read_state_json(shim_path)
    clone_entry = next(e for e in state['pipeline']['nodes']
                       if e['type'] == 'transform.legacyPython')
    assert 'executor' not in clone_entry
    assert 'tomviz_pipeline_env' not in clone_entry['description']
    # The rest of the description survives the strip.
    assert json.loads(clone_entry['description'])['name'] == 'AddOne'
