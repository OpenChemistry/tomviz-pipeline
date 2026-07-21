###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Integration tests for ExternalNodeExecutor: running a node in a
subprocess under a (faked) separate Python environment, the serialized
`executor` block round-trip, failure handling, and ProgressReader
message parsing.

POSIX-only: the parent<->child progress channel uses a Unix socket."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from tomviz_pipeline import Dataset, register_builtins
from tomviz_pipeline.core import (
    NodeFactory,
    NodeState,
    Pipeline,
    PortData,
    SourceNode,
    pipeline_from_state_dict,
    pipeline_to_state_dict,
)
from tomviz_pipeline.external import ExternalNodeExecutor, ProgressReader

pytestmark = pytest.mark.skipif(
    os.name != 'posix',
    reason='ExternalNodeExecutor progress channel is a Unix socket')

REPO_SRC = Path(__file__).resolve().parent.parent / 'src'


@pytest.fixture(autouse=True)
def _builtins():
    register_builtins()


@pytest.fixture(scope='module')
def fake_env(tmp_path_factory):
    """A fake Python environment whose bin/tomviz-pipeline wrapper runs
    the CLI from this repo. The executor scrubs PYTHONPATH from the
    child environment, so the wrapper re-exports it."""
    env = tmp_path_factory.mktemp('fake-env')
    bin_dir = env / 'bin'
    bin_dir.mkdir()
    cli = bin_dir / 'tomviz-pipeline'
    cli.write_text(
        '#!/bin/sh\n'
        f'export PYTHONPATH="{REPO_SRC}"\n'
        f'exec "{sys.executable}" -m tomviz_pipeline.cli "$@"\n')
    cli.chmod(0o755)
    return env


def _make_dataset(value=7, shape=(2, 2, 3)):
    arr = np.full(shape, value, dtype=np.uint16, order='F')
    ds = Dataset({'Scalars': arr}, 'Scalars')
    ds.spacing = (1.0, 1.0, 1.0)
    return ds


def _external_pipeline(env_path):
    """Current source with numpy Dataset payload -> external
    convertToFloat. Returns (pipeline, source, transform)."""
    p = Pipeline()
    src = SourceNode()
    out = src.add_output('volume', 'ImageData')
    out.set_data(PortData(_make_dataset(), 'ImageData'))
    p.add_node(src)
    src.state = NodeState.Current

    node = NodeFactory.create('transform.convertToFloat')
    node.node_executor = ExternalNodeExecutor(env_path=str(env_path))
    p.add_node(node)
    p.create_link(src.output_port('volume'), node.input_port('volume'))
    return p, src, node


# ---- basic subprocess execution ---------------------------------------------


def test_external_execution_converts_to_float(fake_env):
    p, src, node = _external_pipeline(fake_env)

    future = p.execute()
    assert future.is_finished()
    assert future.succeeded() is True
    assert node.state == NodeState.Current

    result = node.output_port('output').data().payload
    assert result.active_scalars.dtype == np.float32
    assert np.all(result.active_scalars == 7.0)
    # The input was untouched (the transform deep-copies).
    assert src.output_port('volume').data().payload.active_scalars.dtype \
        == np.uint16


# ---- executor block serialization --------------------------------------------


def test_executor_block_serialized(fake_env):
    node = NodeFactory.create('transform.convertToFloat')
    node.node_executor = ExternalNodeExecutor(env_path=str(fake_env))
    entry = node.serialize()
    assert entry['executor'] == {'type': 'external',
                                 'envPath': str(fake_env)}


def test_executor_block_state_roundtrip(fake_env):
    p = Pipeline()
    node = NodeFactory.create('transform.convertToFloat')
    node.node_executor = ExternalNodeExecutor(env_path=str(fake_env))
    p.add_node(node)

    state = pipeline_to_state_dict(p)
    entry = state['pipeline']['nodes'][0]
    assert entry['executor'] == {'type': 'external',
                                 'envPath': str(fake_env)}

    # register_builtins registered the 'external' executor type, so the
    # block is restored as a live ExternalNodeExecutor.
    restored = pipeline_from_state_dict(state)
    rnode = restored.node_by_id(node.id)
    assert isinstance(rnode.node_executor, ExternalNodeExecutor)
    assert rnode.node_executor.env_path == str(fake_env)


# ---- failure ----------------------------------------------------------------


def test_external_execution_fails_with_empty_env(tmp_path):
    empty_env = tmp_path / 'empty-env'
    empty_env.mkdir()
    p, src, node = _external_pipeline(empty_env)

    future = p.execute()
    assert future.is_finished()
    assert future.succeeded() is False
    assert node.state != NodeState.Current
    assert not node.output_port('output').has_data()


# ---- node state round trip + auto-execute poll -----------------------------

_COUNTING_SOURCE_DESCRIPTION = json.dumps({
    'name': 'CountingSource',
    'outputs': [{'name': 'volume', 'type': 'ImageData'}],
    'parameters': [{'name': 'value', 'type': 'double', 'default': 0.0}],
})

_COUNTING_SOURCE_SCRIPT = '''
import numpy as np
from tomviz_pipeline.dataset import Dataset
from tomviz_pipeline.kernels import SourceKernel


class CountingSource(SourceKernel):
    def produce(self, value=0.0):
        self.state['runs'] = self.state.get('runs', 0) + 1
        arr = np.full((2, 2, 2), value, dtype=np.float32)
        return {'volume': Dataset({'Scalars': arr}, active='Scalars')}

    def should_auto_execute(self, value=0.0):
        if value != 7.5:
            raise ValueError('parameters not forwarded')
        polls = self.state.get('polls', 0) + 1
        self.state['polls'] = polls
        return polls >= 2
'''


def _counting_node(env_path):
    from tomviz_pipeline import PythonNode
    node = PythonNode(_COUNTING_SOURCE_DESCRIPTION,
                      kernel=_COUNTING_SOURCE_SCRIPT)
    node.set_parameters(value=7.5)
    node.node_executor = ExternalNodeExecutor(env_path=str(env_path))
    return node


def test_external_execution_round_trips_node_state(fake_env):
    """With periodic execution on, the bag rides the --node-state
    sidecar into the child and node_state.json back — the counter
    only reaches 2 if both directions work."""
    node = _counting_node(fake_env)
    node.auto_execute_enabled = True
    p = Pipeline()
    p.add_node(node)

    assert p.execute().succeeded() is True
    assert node.user_state == {'runs': 1}

    node.set_parameters(value=8.5)
    assert p.execute().succeeded() is True
    assert node.user_state == {'runs': 2}


def test_external_execution_skips_sidecar_when_unused(fake_env):
    """Older external envs don't know --node-state, so it is only
    passed when the node uses the feature."""
    node = _counting_node(fake_env)
    p = Pipeline()
    p.add_node(node)

    assert p.execute().succeeded() is True
    assert node.user_state == {}


def test_external_should_auto_execute_polls_in_child(fake_env):
    node = _counting_node(fake_env)
    executor = node.node_executor

    assert executor.should_auto_execute(node) is False
    assert node.user_state == {'polls': 1}
    assert executor.should_auto_execute(node) is True
    assert node.user_state == {'polls': 2}


def test_external_should_auto_execute_without_cli_answers_false(tmp_path):
    empty_env = tmp_path / 'empty-env'
    empty_env.mkdir()
    node = _counting_node(empty_env)
    assert node.node_executor.should_auto_execute(node) is False
    assert node.user_state == {}


# ---- ProgressReader message parsing -----------------------------------------


def test_progress_reader_parses_json_lines():
    reader = ProgressReader('unused')
    events = []
    reader.pipeline_started.connect(
        lambda: events.append(('pipeline_started',)))
    reader.pipeline_finished.connect(
        lambda: events.append(('pipeline_finished',)))
    reader.node_started.connect(
        lambda i: events.append(('started', i)))
    reader.node_finished.connect(
        lambda i: events.append(('finished', i)))
    reader.node_error.connect(
        lambda i, m: events.append(('error', i, m)))
    reader.node_progress_maximum.connect(
        lambda i, v: events.append(('maximum', i, v)))
    reader.node_progress_step.connect(
        lambda i, v: events.append(('step', i, v)))
    reader.node_progress_message.connect(
        lambda i, m: events.append(('message', i, m)))
    reader.node_progress_data.connect(
        lambda i, f: events.append(('data', i, f)))

    messages = [
        {'type': 'started'},
        {'type': 'started', 'operator': 2},
        {'type': 'progress.maximum', 'operator': 2, 'value': 10},
        {'type': 'progress.step', 'operator': 2, 'value': 3},
        {'type': 'progress.message', 'operator': 2, 'value': 'working'},
        {'type': 'progress.data', 'operator': 2, 'value': 'im0.tvh5'},
        {'type': 'error', 'operator': 2, 'error': 'boom'},
        {'type': 'finished', 'operator': 2},
        {'type': 'finished'},
    ]
    for obj in messages:
        reader.handle_message(json.dumps(obj))

    assert events == [
        ('pipeline_started',),
        ('started', 2),
        ('maximum', 2, 10),
        ('step', 2, 3),
        ('message', 2, 'working'),
        ('data', 2, 'im0.tvh5'),
        ('error', 2, 'boom'),
        ('finished', 2),
        ('pipeline_finished',),
    ]


def test_progress_reader_ignores_garbage():
    reader = ProgressReader('unused')
    hits = []
    reader.node_started.connect(lambda i: hits.append(i))
    reader.pipeline_started.connect(lambda: hits.append('p'))
    reader.handle_message('')                 # blank line
    reader.handle_message('   \n')            # whitespace only
    reader.handle_message('not json')         # invalid JSON
    reader.handle_message('[1, 2]')           # valid JSON, not an object
    reader.handle_message('{"type": "wat"}')  # unknown type
    assert hits == []
