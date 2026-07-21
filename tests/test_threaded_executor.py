###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Tests for ThreadedExecutor: worker-thread execution, cooperative
cancellation, breakpoints, pending-run replacement on re-submit,
cancel_and_wait, pause/unpause, and EventQueue marshaling. Synchronization
uses threading.Event handshakes inside the test nodes — never bare
sleeps."""

import threading
import time

from tomviz_pipeline.core import (
    EventQueue,
    NodeExecState,
    NodeState,
    Pipeline,
    PortData,
    SourceNode,
    ThreadedExecutor,
    TransformNode,
)

TIMEOUT = 10


class _Source(SourceNode):
    type_name = 'test.threaded.source'

    def __init__(self, value=1.0):
        super().__init__()
        self.value = value
        self.add_output('out', 'ImageData')

    def execute(self):
        self.output_port('out').set_data(
            PortData(self.value, 'ImageData'))
        return True


class _Double(TransformNode):
    type_name = 'test.threaded.double'

    def __init__(self):
        super().__init__()
        self.run_count = 0
        self.add_input('in', 'ImageData')
        self.add_output('out', 'ImageData')

    def transform(self, inputs):
        self.run_count += 1
        return {'out': PortData(inputs['in'].payload * 2, 'ImageData')}


class _Blocking(TransformNode):
    """Parks in transform() until released or canceled. `entered` lets
    the test thread know the node is running; cancel-vs-release races
    are resolved deterministically in favor of cancel (a final check
    after the wait loop), matching how tests order the two events."""

    type_name = 'test.threaded.blocking'

    def __init__(self):
        super().__init__()
        self.add_input('in', 'ImageData')
        self.add_output('out', 'ImageData')
        self.entered = threading.Event()
        self.release = threading.Event()
        self.enter_count = 0

    def transform(self, inputs):
        self.enter_count += 1
        self.entered.set()
        while not self.release.is_set():
            if self.is_cancel_requested():
                return None
            time.sleep(0.01)
        if self.is_cancel_requested():
            return None
        return {'out': PortData(inputs['in'].payload, 'ImageData')}


def _threaded_pipeline(*nodes):
    """Chain nodes left to right and attach a ThreadedExecutor."""
    p = Pipeline()
    executor = ThreadedExecutor()
    p.set_executor(executor)
    for n in nodes:
        p.add_node(n)
    for up, down in zip(nodes, nodes[1:]):
        p.create_link(up.output_port('out'), down.input_port('in'))
    return p, executor


# ---- basic ------------------------------------------------------------------


def test_basic_threaded_execution():
    src = _Source(value=3.0)
    t = _Double()
    p, executor = _threaded_pipeline(src, t)

    main_ident = threading.get_ident()
    started = []
    finished = []
    executor.node_execution_started.connect(
        lambda node: started.append((node, threading.get_ident())))
    executor.node_execution_finished.connect(
        lambda node, ok: finished.append((node, ok)))

    future = p.execute()
    assert future.wait(TIMEOUT)
    assert future.succeeded()
    assert src.state == NodeState.Current
    assert t.state == NodeState.Current
    assert t.output_port('out').data().payload == 6.0

    assert [node for node, _ in started] == [src, t]
    assert finished == [(src, True), (t, True)]
    # Direct connections run on the worker thread, not this one.
    for _, ident in started:
        assert ident != main_ident


# ---- cancellation ------------------------------------------------------------


def test_cancel_execution_from_main_thread():
    src = _Source()
    block = _Blocking()
    tail = _Double()
    p, executor = _threaded_pipeline(src, block, tail)

    canceled_hits = []
    executor.canceled.connect(lambda: canceled_hits.append(1))

    future = p.execute()
    assert block.entered.wait(TIMEOUT)
    p.cancel_execution()

    assert future.wait(TIMEOUT)
    assert future.was_canceled() is True
    assert future.succeeded() is False
    assert canceled_hits == [1]

    # The canceled node left its state untouched; the trailing node
    # never ran.
    assert block.state == NodeState.New
    assert block.exec_state == NodeExecState.Canceled
    assert tail.state != NodeState.Current
    assert tail.run_count == 0
    assert src.state == NodeState.Current


# ---- breakpoints ------------------------------------------------------------


def test_breakpoint_stops_before_node():
    src = _Source()
    bp = _Double()
    tail = _Double()
    bp.breakpoint = True
    p, executor = _threaded_pipeline(src, bp, tail)

    hits = []
    p.breakpoint_reached.connect(lambda node: hits.append(node))

    future = p.execute()
    assert future.wait(TIMEOUT)
    assert future.succeeded() is False
    assert future.was_canceled() is False
    assert hits == [bp]
    # Upstream ran; the breakpoint node and everything after did not.
    assert src.state == NodeState.Current
    assert bp.state == NodeState.New
    assert bp.run_count == 0
    assert tail.state == NodeState.New
    assert tail.run_count == 0


# ---- submit while running ---------------------------------------------------


def test_submit_while_running_cancels_first_and_runs_second():
    src = _Source(value=5.0)
    block = _Blocking()
    p, executor = _threaded_pipeline(src, block)

    first = p.execute()
    assert block.entered.wait(TIMEOUT)

    # Re-submitting while running: the in-flight run is canceled at its
    # next poll and the new plan starts once the worker exits.
    second = p.execute()
    block.release.set()

    assert first.wait(TIMEOUT)
    assert second.wait(TIMEOUT)
    assert first.was_canceled() is True
    assert first.succeeded() is False
    assert second.was_canceled() is False
    assert second.succeeded() is True
    # The node ran twice: once canceled, once to completion.
    assert block.enter_count == 2
    assert block.state == NodeState.Current
    assert block.output_port('out').data().payload == 5.0


# ---- cancel_and_wait --------------------------------------------------------


def test_cancel_and_wait_joins_worker():
    src = _Source()
    block = _Blocking()
    p, executor = _threaded_pipeline(src, block)

    future = p.execute()
    assert block.entered.wait(TIMEOUT)

    executor.cancel_and_wait(timeout=TIMEOUT)
    # The worker has exited and nothing is pending.
    assert executor.is_running() is False
    assert future.is_finished()
    assert future.was_canceled() is True


# ---- pause ------------------------------------------------------------------


def test_pause_defers_and_unpause_auto_executes():
    src = _Source(value=2.0)
    t = _Double()
    p, executor = _threaded_pipeline(src, t)

    finished_evt = threading.Event()
    finished = []

    def on_finished(fut):
        finished.append(fut)
        finished_evt.set()

    p.execution_finished.connect(on_finished)

    p.set_paused(True)
    paused_future = p.execute()
    assert paused_future.is_started() is False
    assert paused_future.is_finished() is False
    # Nothing ran while paused.
    assert src.state == NodeState.New
    assert t.state == NodeState.New

    # Un-pausing auto-executes because nodes are not Current.
    p.set_paused(False)
    assert finished_evt.wait(TIMEOUT)
    assert len(finished) == 1
    fresh = finished[0]
    assert fresh is not paused_future
    assert fresh.wait(TIMEOUT)
    assert fresh.succeeded() is True
    assert src.state == NodeState.Current
    assert t.state == NodeState.Current
    assert paused_future.is_finished() is False


# ---- EventQueue marshaling --------------------------------------------------


def test_event_queue_marshals_handler_to_main_thread():
    src = _Source()
    t = _Double()
    p, executor = _threaded_pipeline(src, t)

    events = EventQueue()
    idents = []
    p.execution_finished.connect(
        lambda fut: idents.append(threading.get_ident()), events)

    future = p.execute()
    assert future.wait(TIMEOUT)

    # The emission was queued (not delivered on the worker); drain it
    # here and confirm the handler ran on this thread.
    deadline = time.monotonic() + TIMEOUT
    while not idents and time.monotonic() < deadline:
        events.process(block=True, timeout=0.1)
    assert idents == [threading.get_ident()]
