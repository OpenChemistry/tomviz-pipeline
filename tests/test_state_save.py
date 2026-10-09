###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Saving a session: port metadata and inert sink settings survive a
load/save round trip, save_state writes .tvsm and .tvh5 documents with
the application's extra sections, relative reader paths follow the new
file, and a failed write leaves an existing file alone."""

import copy
import json
import os

import numpy as np
import pytest

from tomviz_pipeline import (
    Node,
    NodeExecutor,
    Pipeline,
    PortData,
    SinkGroupNode,
    SourceNode,
    pipeline_from_state_dict,
    pipeline_to_state_dict,
)
from tomviz_pipeline.dataset import Dataset
from tomviz_pipeline.nodes import register_builtins
from tomviz_pipeline.nodes.sinks import _NODE_KEYS
from tomviz_pipeline.nodes.sources.reader import ReaderSourceNode
from tomviz_pipeline.nodes.transforms.convert_to_float import (
    ConvertToFloatTransform,
)
from tomviz_pipeline.state import (
    build_state,
    load_state,
    read_state_json,
    save_state,
    write_state,
    write_state_tvh5,
)


@pytest.fixture(autouse=True)
def _builtins():
    register_builtins()


COLOR_MAP = {
    'colorSpace': 'Diverging',
    'colors': [0.0, 0.2, 0.3, 0.9, 255.0, 0.7, 0.0, 0.1],
    'points': [0.0, 0.0, 0.5, 0.0, 255.0, 1.0, 0.5, 0.0],
}


def _desktop_state(file_name='data.emd'):
    """A desktop-style document: a reader with port metadata, a sink
    group, a slice the runtime knows nothing about, views and layouts."""
    return {
        'schemaVersion': 2,
        'paletteColor': [0.1, 0.1, 0.1],
        'views': [{'id': 7, 'camera': {'position': [0, 0, 1]}}],
        'layouts': [{'id': 8, 'items': [[{'viewId': 7}]]}],
        'pipeline': {
            'nextNodeId': 4,
            'nodes': [
                {'id': 1, 'type': 'source.reader', 'label': 'data.emd',
                 'fileNames': [file_name],
                 'outputPorts': {'volume': {
                     'type': 'ImageData', 'persistent': True,
                     'metadata': {'colorOpacityMap': COLOR_MAP,
                                  'activeScalars': 'ImageScalars',
                                  'spacing': [1.0, 1.0, 2.0]}}}},
                {'id': 2, 'type': 'sinkGroup', 'label': 'Visualizations',
                 'inputPorts': {'volume': {'type': ['ImageData']}},
                 'outputPorts': {'volume': {
                     'type': 'ImageData', 'persistent': False}},
                 'typeInferenceSources': {'volume': 'volume'}},
                {'id': 3, 'type': 'sink.ruler', 'label': 'Ruler',
                 'inputPorts': {'volume': {'type': ['ImageData']}},
                 'viewId': 7, 'visible': True,
                 'point1': [0.0, 0.0, 0.0], 'point2': [1.0, 2.0, 3.0],
                 'breakpoint': True},
            ],
            'links': [
                {'from': {'node': 1, 'port': 'volume'},
                 'to': {'node': 2, 'port': 'volume'}},
                {'from': {'node': 2, 'port': 'volume'},
                 'to': {'node': 3, 'port': 'volume'}},
            ],
        },
    }


def _entries(state):
    return {entry['id']: entry for entry in state['pipeline']['nodes']}


def _volume(value=1):
    array = np.full((2, 3, 4), value, dtype=np.uint8, order='F')
    return Dataset({'ImageScalars': array}, 'ImageScalars')


# ---- port metadata -------------------------------------------------------


def test_port_metadata_round_trips():
    raw = _desktop_state()
    p = pipeline_from_state_dict(raw)
    port = p.node_by_id(1).output_port('volume')
    saved = raw['pipeline']['nodes'][0]['outputPorts']['volume']['metadata']
    assert port.metadata == saved
    # The port holds its own copy of what the file says.
    saved['activeScalars'] = 'Other'
    assert port.metadata['activeScalars'] == 'ImageScalars'

    port.metadata['colorOpacityMap'] = {'colorSpace': 'RGB'}
    entry = _entries(pipeline_to_state_dict(p))[1]
    metadata = entry['outputPorts']['volume']['metadata']
    assert metadata['colorOpacityMap'] == {'colorSpace': 'RGB'}
    assert metadata['spacing'] == [1.0, 1.0, 2.0]
    # The document is a snapshot: later edits to the port do not leak in.
    port.metadata['colorOpacityMap']['colorSpace'] = 'Lab'
    assert metadata['colorOpacityMap'] == {'colorSpace': 'RGB'}


def test_empty_metadata_is_not_written():
    p = Pipeline()
    source = p.add_node(SourceNode())
    source.add_output('out', 'ImageData')
    entry = _entries(pipeline_to_state_dict(p))[source.id]
    assert entry['outputPorts']['out'] == {
        'type': 'ImageData', 'persistent': True}


def test_passthrough_ports_carry_no_metadata():
    raw = _desktop_state()
    group_ports = raw['pipeline']['nodes'][1]['outputPorts']
    group_ports['volume']['metadata'] = {'activeScalars': 'Stale'}
    p = pipeline_from_state_dict(raw)
    group = p.node_by_id(2)
    assert isinstance(group, SinkGroupNode)
    group.output_port('volume').metadata = {'activeScalars': 'Ignored'}
    entry = _entries(pipeline_to_state_dict(p))[2]
    assert 'metadata' not in entry['outputPorts']['volume']


# ---- inert sinks ---------------------------------------------------------


def test_inert_sink_writes_its_settings_back():
    raw = _desktop_state()
    p = pipeline_from_state_dict(raw)
    sink = p.node_by_id(3)
    assert sink.settings == {'viewId': 7, 'visible': True,
                             'point1': [0.0, 0.0, 0.0],
                             'point2': [1.0, 2.0, 3.0]}

    entry = _entries(pipeline_to_state_dict(p))[3]
    assert entry == raw['pipeline']['nodes'][2]


def test_inert_sink_saves_graph_edits():
    p = pipeline_from_state_dict(_desktop_state())
    sink = p.node_by_id(3)
    sink.label = 'Distance'
    sink.breakpoint = False
    p.remove_link(sink.input_port('volume').link)

    state = pipeline_to_state_dict(p)
    entry = _entries(state)[3]
    assert entry['label'] == 'Distance'
    assert 'breakpoint' not in entry
    assert entry['point2'] == [1.0, 2.0, 3.0]
    assert all(link['to']['node'] != 3
               for link in state['pipeline']['links'])


def test_node_keys_cover_what_node_serialize_writes():
    class _Executor(NodeExecutor):
        type_name = 'stub'

    node = Node()
    node.label = 'All'
    node.breakpoint = True
    node.properties = {'color': 'red'}
    node.type_inference_sources = {'out': 'in'}
    node.node_executor = _Executor()
    node.auto_execute_enabled = True
    node.add_input('in', ['ImageData'])
    node.add_output('out', 'ImageData').metadata = {'label': 'x'}
    node.mark_stale()
    keys = set(node.serialize()) | {'id', 'type'}
    assert keys <= _NODE_KEYS


def test_desktop_document_round_trips_through_the_library():
    raw = _desktop_state()
    p = pipeline_from_state_dict(raw)
    extra = {k: v for k, v in raw.items()
             if k not in ('schemaVersion', 'pipeline')}
    state = build_state(p, extra)

    assert {k: state[k] for k in extra} == extra
    assert state['pipeline']['links'] == raw['pipeline']['links']
    saved = _entries(state)
    for node_id, original in _entries(raw).items():
        assert saved[node_id] == original, node_id


# ---- save_state ----------------------------------------------------------


def _reader_pipeline(tmp_path):
    """A reader whose output already carries data, feeding a transform
    whose output is transient."""
    p = Pipeline()
    reader = p.add_node(ReaderSourceNode())
    reader.file_names = [str(tmp_path / 'data.emd')]
    reader.output_port('volume').set_data(PortData(_volume(3), 'Volume'))
    reader.output_port('volume').metadata = {'activeScalars': 'ImageScalars'}
    to_float = p.add_node(ConvertToFloatTransform())
    p.create_link(reader.output_port('volume'),
                  to_float.input_port('volume'))
    output = to_float.output_port('output')
    output.persistent = False
    output.set_data(PortData(_volume(4), 'Volume'))
    return p, reader, to_float


def test_save_state_tvsm(tmp_path):
    p, reader, to_float = _reader_pipeline(tmp_path)
    path = tmp_path / 'session.tvsm'
    save_state(path, p, {'views': [{'id': 1}]})

    raw = read_state_json(path)
    assert raw['views'] == [{'id': 1}]
    assert 'dataRef' not in json.dumps(raw)
    loaded = load_state(path)
    port = loaded.node_by_id(reader.id).output_port('volume')
    assert port.metadata == {'activeScalars': 'ImageScalars'}
    assert not port.has_data()
    assert loaded.node_by_id(to_float.id).input_port('volume').link


def test_save_state_tvh5_embeds_persistent_ports(tmp_path):
    p, reader, to_float = _reader_pipeline(tmp_path)
    path = tmp_path / 'session.tvh5'
    save_state(path, p, {'views': [{'id': 1}]})

    raw = read_state_json(path)
    assert raw['views'] == [{'id': 1}]
    entries = _entries(raw)
    assert 'dataRef' in entries[reader.id]['outputPorts']['volume']
    assert 'dataRef' not in entries[to_float.id]['outputPorts']['output']

    loaded = load_state(path)
    port = loaded.node_by_id(reader.id).output_port('volume')
    assert port.metadata == {'activeScalars': 'ImageScalars'}
    assert np.all(port.data().payload.active_scalars == 3)
    assert not loaded.node_by_id(to_float.id).output_port('output') \
        .has_data()


def test_tvh5_writer_embeds_every_port_by_default(tmp_path):
    p, _reader, to_float = _reader_pipeline(tmp_path)
    path = tmp_path / 'all.tvh5'
    write_state_tvh5(path, pipeline_to_state_dict(p), p)
    entry = _entries(read_state_json(path))[to_float.id]
    assert 'dataRef' in entry['outputPorts']['output']


def test_relative_reader_paths_follow_the_new_file(tmp_path):
    first = tmp_path / 'first'
    first.mkdir()
    raw = _desktop_state('data/scan.emd')
    absolute = str(tmp_path / 'elsewhere' / 'other.emd')
    raw['pipeline']['nodes'][0]['fileNames'].append(absolute)
    p = pipeline_from_state_dict(raw, state_dir=first)

    second = tmp_path / 'second' / 'deeper'
    second.mkdir(parents=True)
    save_state(second / 'session.tvsm', p)
    names = _entries(read_state_json(second / 'session.tvsm'))[1]
    assert names['fileNames'] == ['../../first/data/scan.emd', absolute]
    # Both files name the same data; the node itself is untouched.
    reader = p.node_by_id(1)
    assert reader.file_names == ['data/scan.emd', absolute]
    assert (second / names['fileNames'][0]).resolve() == \
        reader.resolve_path('data/scan.emd')

    # Without a target directory the paths are written as they are.
    assert _entries(build_state(p))[1]['fileNames'] == reader.file_names


def test_relative_paths_without_a_state_dir_are_from_the_cwd(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = Pipeline()
    reader = p.add_node(ReaderSourceNode())
    reader.file_names = ['scan.emd']
    (tmp_path / 'out').mkdir()
    state = build_state(p, state_dir=tmp_path / 'out')
    assert _entries(state)[reader.id]['fileNames'] == ['../scan.emd']


def test_failed_write_keeps_the_existing_file(tmp_path):
    path = tmp_path / 'session.tvsm'
    path.write_text('previous')
    p = Pipeline()
    state = build_state(p, {'views': {'not', 'json'}})
    with pytest.raises(TypeError):
        write_state(path, state, p)
    assert path.read_text() == 'previous'
    assert os.listdir(tmp_path) == ['session.tvsm']


def test_build_state_copies_extra():
    extra = {'views': [{'id': 1, 'camera': {'position': [0, 0, 1]}}]}
    state = build_state(Pipeline(), extra)
    snapshot = copy.deepcopy(state)
    extra['views'][0]['camera']['position'][2] = 5
    assert state == snapshot
