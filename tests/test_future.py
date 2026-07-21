###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Tests for ExecutionFuture: wait(), the succeeded/was_canceled state
matrix, finished/canceled signal ordering, idempotent resolution, and the
unstarted futures returned by a paused Pipeline."""

import threading

from tomviz_pipeline.core import (
    ExecutionFuture,
    NodeState,
    Pipeline,
    SourceNode,
)


# ---- wait ------------------------------------------------------------------


def test_wait_returns_false_before_finish_and_true_after():
    f = ExecutionFuture()
    assert f.wait(0.05) is False
    assert not f.is_finished()
    f._finish(True)
    assert f.wait(0) is True
    assert f.wait(5) is True
    assert f.is_finished()


def test_wait_unblocks_waiter_thread():
    f = ExecutionFuture()
    results = []

    def waiter():
        results.append(f.wait(10))

    t = threading.Thread(target=waiter)
    t.start()
    f._finish(True)
    t.join(10)
    assert not t.is_alive()
    assert results == [True]


# ---- succeeded / was_canceled combinations ---------------------------------


def test_unfinished_future_never_reports_success():
    f = ExecutionFuture()
    assert f.succeeded() is False
    assert f.was_canceled() is False


def test_finish_success():
    f = ExecutionFuture()
    f._finish(True)
    assert f.succeeded() is True
    assert f.was_canceled() is False


def test_finish_failure():
    f = ExecutionFuture()
    f._finish(False)
    assert f.succeeded() is False
    assert f.was_canceled() is False


def test_finish_canceled():
    f = ExecutionFuture()
    f._finish(False, canceled=True)
    assert f.succeeded() is False
    assert f.was_canceled() is True
    assert f.is_finished()


def test_cancellation_overrides_success_flag():
    f = ExecutionFuture()
    f._finish(True, canceled=True)
    assert f.succeeded() is False
    assert f.was_canceled() is True


# ---- signals ----------------------------------------------------------------


def test_finished_signal_fires_without_canceled_on_success():
    f = ExecutionFuture()
    events = []
    f.finished.connect(lambda fut: events.append(('finished', fut)))
    f.canceled.connect(lambda fut: events.append(('canceled', fut)))
    f._finish(True)
    assert events == [('finished', f)]


def test_canceled_signal_fires_before_finished():
    f = ExecutionFuture()
    events = []
    f.finished.connect(lambda fut: events.append(('finished', fut)))
    f.canceled.connect(lambda fut: events.append(('canceled', fut)))
    f._finish(False, canceled=True)
    assert events == [('canceled', f), ('finished', f)]


def test_finish_is_idempotent():
    f = ExecutionFuture()
    count = []
    f.finished.connect(lambda fut: count.append(1))
    f._finish(True)
    # A second resolution is ignored entirely.
    f._finish(False, canceled=True)
    assert count == [1]
    assert f.succeeded() is True
    assert f.was_canceled() is False


# ---- is_started -------------------------------------------------------------


def test_fresh_future_is_not_started():
    f = ExecutionFuture()
    assert f.is_started() is False
    assert f.is_finished() is False


def test_execute_while_paused_returns_unstarted_future():
    p = Pipeline()
    node = SourceNode()
    node.add_output('volume', 'ImageData')
    p.add_node(node)

    p.set_paused(True)
    future = p.execute()
    assert future.is_started() is False
    assert future.is_finished() is False
    assert future.succeeded() is False
    # Nothing ran.
    assert node.state == NodeState.New


def test_unpausing_runs_and_resolves_a_fresh_started_future():
    p = Pipeline()
    node = SourceNode()
    node.add_output('volume', 'ImageData')
    p.add_node(node)

    p.set_paused(True)
    paused_future = p.execute()

    seen = []
    p.execution_finished.connect(lambda fut: seen.append(fut))
    # Un-pausing auto-executes (blocking DefaultExecutor by default).
    p.set_paused(False)
    assert len(seen) == 1
    fresh = seen[0]
    assert fresh is not paused_future
    assert fresh.is_started() is True
    assert fresh.is_finished() is True
    assert fresh.succeeded() is True
    assert node.state == NodeState.Current
    # The paused future never started and never resolves.
    assert paused_future.is_started() is False
    assert paused_future.is_finished() is False
