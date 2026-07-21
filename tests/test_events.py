###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Tests for the Signal / EventQueue observer primitives in
tomviz_pipeline.core.events: direct emission, disconnect semantics
(including mid-emit), exception swallowing, and queued delivery."""

import logging
import threading
import time

from tomviz_pipeline.core import EventQueue, Signal


# ---- Signal: direct connections ------------------------------------------


def test_direct_emit_calls_handlers_in_connect_order():
    sig = Signal('sig')
    calls = []
    sig.connect(lambda: calls.append('a'))
    sig.connect(lambda: calls.append('b'))
    sig.connect(lambda: calls.append('c'))
    sig.emit()
    assert calls == ['a', 'b', 'c']


def test_emit_forwards_arguments():
    sig = Signal('sig')
    seen = []
    sig.connect(lambda *args: seen.append(args))
    sig.emit(1, 'two')
    assert seen == [(1, 'two')]


def test_disconnect_via_connection_handle():
    sig = Signal('sig')
    calls = []
    conn = sig.connect(lambda: calls.append('x'))
    assert conn.alive
    assert sig.connection_count() == 1
    conn.disconnect()
    assert not conn.alive
    assert sig.connection_count() == 0
    sig.emit()
    assert calls == []


def test_disconnect_by_callback():
    sig = Signal('sig')
    calls = []

    def handler():
        calls.append('x')

    sig.connect(handler)
    sig.connect(handler)  # duplicate connection on purpose
    assert sig.disconnect(handler) is True
    sig.emit()
    assert calls == []
    # Nothing left to disconnect.
    assert sig.disconnect(handler) is False


def test_handler_exception_is_swallowed_and_logged(caplog):
    sig = Signal('boomsig')
    calls = []

    def bad():
        raise RuntimeError('boom')

    sig.connect(bad)
    sig.connect(lambda: calls.append('after'))
    with caplog.at_level(logging.ERROR, logger='tomviz_pipeline'):
        sig.emit()  # must not raise
    # The handler after the raising one still ran.
    assert calls == ['after']
    # The failure was logged, naming the signal.
    assert 'boomsig' in caplog.text


def test_connect_during_emit_takes_effect_next_emit():
    sig = Signal('sig')
    calls = []

    def late():
        calls.append('late')

    def first():
        calls.append('first')
        sig.connect(late)

    sig.connect(first)
    sig.emit()
    # `late` was connected while emitting: not delivered this emit.
    assert calls == ['first']
    sig.emit()
    assert calls == ['first', 'first', 'late']


def test_disconnect_during_emit_prevents_delivery():
    sig = Signal('sig')
    calls = []
    conns = {}

    def a():
        calls.append('a')
        conns['b'].disconnect()

    def b():
        calls.append('b')

    sig.connect(a)
    conns['b'] = sig.connect(b)
    sig.emit()
    # `b` was disconnected by `a` mid-emit and must not be delivered.
    assert calls == ['a']


# ---- EventQueue: queued connections ---------------------------------------


def test_queued_connection_defers_until_process():
    sig = Signal('sig')
    events = EventQueue()
    calls = []
    sig.connect(lambda v: calls.append(v), events)
    sig.emit(42)
    # Emission only enqueues: nothing delivered yet.
    assert calls == []
    assert not events.empty()
    assert events.process() == 1
    assert calls == [42]
    assert events.empty()
    assert events.process() == 0


def test_queued_connection_disconnected_before_process_not_delivered():
    sig = Signal('sig')
    events = EventQueue()
    calls = []
    conn = sig.connect(lambda: calls.append('x'), events)
    sig.emit()
    conn.disconnect()
    events.process()
    assert calls == []


def test_process_block_times_out_when_empty():
    events = EventQueue()
    start = time.monotonic()
    assert events.process(block=True, timeout=0.2) == 0
    assert time.monotonic() - start >= 0.1


def test_process_block_wakes_on_post_from_other_thread():
    sig = Signal('sig')
    events = EventQueue()
    idents = []
    sig.connect(lambda: idents.append(threading.get_ident()), events)

    t = threading.Thread(target=sig.emit)
    t.start()
    delivered = events.process(block=True, timeout=10)
    t.join(10)
    assert not t.is_alive()
    assert delivered == 1
    # Queued handlers run on the processing thread, not the emitter.
    assert idents == [threading.get_ident()]


def test_join_unblocks_after_processing():
    sig = Signal('sig')
    events = EventQueue()
    calls = []
    sig.connect(lambda: calls.append('x'), events)
    sig.emit()

    joined = threading.Event()

    def waiter():
        events.join()
        joined.set()

    t = threading.Thread(target=waiter)
    t.start()
    # The posted event has not been processed: join() must still block.
    assert not joined.wait(0.2)
    assert events.process() == 1
    assert joined.wait(10)
    t.join(10)
    assert not t.is_alive()
    assert calls == ['x']
