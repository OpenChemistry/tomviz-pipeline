###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""What kernels print, delivered to their node's output_written signal
(core/output.py), and carried over the progress channel."""

import io
import json
import logging
import sys
import threading
import time

import numpy as np
import pytest

from tomviz_pipeline import Dataset, ScriptableNode, register_builtins
from tomviz_pipeline.core import (
    DefaultExecutor,
    NodeState,
    Pipeline,
    PortData,
    SourceNode,
)
from tomviz_pipeline.core import output
from tomviz_pipeline.external import FilesProgressReader, ProgressReader
from tomviz_pipeline.progress import FilesProgress, ProgressBase

_PRINTING = {'name': 'Printing', 'parameters': []}
_PRINTING_SCRIPT = '''
import sys


def transform(dataset):
    print("hello")
    print("careful", file=sys.stderr)
'''


@pytest.fixture(autouse=True)
def _builtins():
    register_builtins()


def _pipeline(script, capture_output=True, progress=None):
    """A Current source feeding a scriptable transform running
    ``script``. Returns (pipeline, transform)."""
    p = Pipeline()
    p.set_executor(DefaultExecutor(progress=progress,
                                   capture_output=capture_output))
    src = SourceNode()
    out = src.add_output('volume', 'ImageData')
    arr = np.zeros((2, 2, 2), dtype=np.float32, order='F')
    out.set_data(PortData(Dataset({'Scalars': arr}, 'Scalars'), 'ImageData'))
    p.add_node(src)
    src.state = NodeState.Current
    node = ScriptableNode(_PRINTING, kernel=script)
    p.add_node(node)
    p.create_link(src.output_port('volume'), node.input_port('volume'))
    return p, node


def _collect(node):
    texts = []
    node.output_written.connect(lambda _node, text: texts.append(text))
    return texts


# ---- in-process kernels ------------------------------------------------------


def test_what_a_kernel_prints_reaches_its_node(capsys):
    p, node = _pipeline(_PRINTING_SCRIPT)
    texts = _collect(node)
    assert p.execute(node).succeeded() is True
    assert ''.join(texts) == 'hello\ncareful\n'
    # ... and the terminal, as before
    captured = capsys.readouterr()
    assert 'hello' in captured.out
    assert 'careful' in captured.err


def test_without_capture_nothing_is_delivered(capsys):
    p, node = _pipeline(_PRINTING_SCRIPT, capture_output=False)
    texts = _collect(node)
    assert p.execute(node).succeeded() is True
    assert texts == []
    assert 'hello' in capsys.readouterr().out


def test_output_is_delivered_before_the_node_finishes():
    p, node = _pipeline(_PRINTING_SCRIPT)
    events = []
    node.output_written.connect(lambda _n, text: events.append('output'))
    p.executor.node_execution_finished.connect(
        lambda _n, _ok: events.append('finished'))
    p.execute(node)
    assert events == ['output', 'finished']


def test_the_progress_reporter_gets_the_output():
    class Recorder(ProgressBase):
        def __init__(self):
            self.outputs = []

        def output(self, node_id, text):
            self.outputs.append((node_id, text))

    recorder = Recorder()
    p, node = _pipeline(_PRINTING_SCRIPT, progress=recorder)
    p.execute(node)
    assert recorder.outputs == [(node.id, 'hello\ncareful\n')]


# ---- the capture itself ------------------------------------------------------


def test_a_running_block_delivers_about_every_interval(monkeypatch):
    monkeypatch.setattr(output, 'FLUSH_INTERVAL', 0.05)
    texts = []
    with output.captured(texts.append):
        print('first')
        deadline = time.monotonic() + 2
        while not texts and time.monotonic() < deadline:
            time.sleep(0.01)
        assert texts == ['first\n']  # delivered while still running
        print('second')
    assert texts == ['first\n', 'second\n']


def test_other_threads_pass_through(capsys):
    texts = []
    with output.captured(texts.append):
        thread = threading.Thread(target=lambda: print('elsewhere'))
        thread.start()
        thread.join()
        print('here')
    assert texts == ['here\n']
    assert 'elsewhere' in capsys.readouterr().out


def test_too_much_output_is_counted_as_dropped(monkeypatch):
    monkeypatch.setattr(output, 'MAX_PENDING', 10)
    texts = []
    with output.captured(texts.append):
        sys.stdout.write('0123456789')
        sys.stdout.write('abcde')
        sys.stdout.write('fgh')
    assert texts == ['0123456789\n[8 characters of output dropped]\n']


def test_replaced_streams_are_captured_again(monkeypatch):
    replacement = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', replacement)
    texts = []
    with output.captured(texts.append):
        print('again')
    assert texts == ['again\n']
    assert replacement.getvalue() == 'again\n'


# ---- the progress channel ----------------------------------------------------


def test_json_progress_sends_output_messages(tmp_path):
    FilesProgress(str(tmp_path)).output(2, 'hi\n')
    (message,) = [json.loads(f.read_text()) for f in tmp_path.iterdir()]
    assert message == {'type': 'output', 'operator': 2, 'value': 'hi\n'}


def test_the_reader_emits_output_and_skips_unknown_types(caplog):
    reader = ProgressReader('unused')
    received = []
    reader.node_output.connect(lambda i, text: received.append((i, text)))
    reader.handle_message(
        json.dumps({'type': 'output', 'operator': 2, 'value': 'hi\n'}))
    with caplog.at_level(logging.WARNING, logger='tomviz_pipeline'):
        reader.handle_message(json.dumps({'type': 'later', 'operator': 2}))
        reader.handle_message(json.dumps({'type': 'later'}))
    assert received == [(2, 'hi\n')]
    assert [r.levelno for r in caplog.records] == [logging.WARNING] * 2


def test_the_files_reader_keeps_the_order_written(tmp_path):
    progress = FilesProgress(str(tmp_path))
    for step in range(12):
        progress.output(2, f'{step}\n')
    reader = FilesProgressReader(str(tmp_path))
    received = []
    reader.node_output.connect(lambda _i, text: received.append(text))
    reader._drain()
    assert received == [f'{step}\n' for step in range(12)]
