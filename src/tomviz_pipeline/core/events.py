###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Thread-safe observer primitives: Signal, EventQueue, and
AsyncioDispatcher.

This is a deliberately small stand-in for Qt's signals/slots. A Signal
invokes its handlers synchronously on the emitting thread (Qt's "direct
connection"). When a handler must run on a specific thread — typically an
application's UI thread while a ThreadedExecutor emits from its worker —
connect it through a *dispatcher*: emission then only hands the call to
the dispatcher, which delivers it where the application lives (Qt's
"queued connection").

Two dispatchers ship with the library, and any object with a
``_post(connection, args)`` method works as one:

- EventQueue — the application thread drains pending deliveries itself
  with process() / run_forever() (fits Qt: a QTimer calling process()).
- AsyncioDispatcher — deliveries are scheduled onto an asyncio event
  loop with call_soon_threadsafe; coroutine handlers become tasks.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

logger = logging.getLogger('tomviz_pipeline')


class Connection:
    """Handle returned by Signal.connect. Call disconnect() to remove the
    handler; safe to call from any thread, including from within the
    handler itself while the signal is being emitted."""

    __slots__ = ('callback', 'dispatcher', '_alive')

    def __init__(self, callback: Callable, dispatcher=None):
        self.callback = callback
        self.dispatcher = dispatcher
        self._alive = True

    @property
    def alive(self) -> bool:
        return self._alive

    def disconnect(self):
        self._alive = False


class Signal:
    """A minimal thread-safe signal.

    Handlers connected without a dispatcher run synchronously on the
    thread that calls emit() — for executor signals that is usually a
    worker thread, so such handlers must be thread-safe. Handlers
    connected with a dispatcher (EventQueue, AsyncioDispatcher) are
    marshaled: emit() posts the invocation and returns; the dispatcher's
    thread or event loop runs it later.

    A handler that raises is logged and skipped; it never breaks the
    emitter or the remaining handlers.
    """

    def __init__(self, name: str = ''):
        self.name = name
        self._lock = threading.Lock()
        self._connections: list[Connection] = []

    def connect(self, callback: Callable,
                dispatcher=None) -> Connection:
        """Register a handler. With no dispatcher it runs synchronously
        on the emitting thread; with one (EventQueue,
        AsyncioDispatcher, or any object providing
        ``_post(connection, args)``) delivery is deferred to wherever
        that dispatcher runs handlers."""
        conn = Connection(callback, dispatcher)
        with self._lock:
            self._connections.append(conn)
        return conn

    def disconnect(self, callback: Callable) -> bool:
        """Disconnect every connection using `callback`. Returns True if
        at least one connection was removed."""
        found = False
        with self._lock:
            for conn in self._connections:
                if conn.callback == callback and conn.alive:
                    conn.disconnect()
                    found = True
            self._prune_locked()
        return found

    def connection_count(self) -> int:
        with self._lock:
            return sum(1 for c in self._connections if c.alive)

    def emit(self, *args: Any):
        with self._lock:
            self._prune_locked()
            connections = list(self._connections)
        for conn in connections:
            if not conn.alive:
                continue
            if conn.dispatcher is not None:
                conn.dispatcher._post(conn, args)
            else:
                self._invoke(conn, args)

    def _invoke(self, conn: Connection, args: tuple):
        try:
            conn.callback(*args)
        except Exception:
            name = self.name or 'signal'
            logger.exception('Unhandled exception in %s handler', name)

    def _prune_locked(self):
        if any(not c.alive for c in self._connections):
            self._connections = [c for c in self._connections if c.alive]


class EventQueue:
    """A queue of pending signal deliveries, drained by the thread that
    owns it.

    Typical use in an application with a main loop::

        events = EventQueue()
        pipeline.execution_finished.connect(on_finished, events)
        ...
        while True:          # main loop
            events.process(block=True, timeout=0.1)

    join() blocks until every event posted so far has been processed,
    which lets an executor wait for the application to observe a node's
    completion before starting the next node (the same barrier the C++
    ThreadedExecutor implements with a semaphore).
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()

    def _post(self, conn: Connection, args: tuple):
        self._queue.put((conn, args))

    def process(self, block: bool = False, timeout: float = None) -> int:
        """Deliver pending events on the calling thread. With block=True,
        wait up to `timeout` seconds for the first event. Returns the
        number of events delivered."""
        delivered = 0
        while True:
            try:
                if block and delivered == 0:
                    conn, args = self._queue.get(block=True, timeout=timeout)
                else:
                    conn, args = self._queue.get_nowait()
            except queue.Empty:
                return delivered
            try:
                if conn.alive:
                    try:
                        conn.callback(*args)
                    except Exception:
                        logger.exception(
                            'Unhandled exception in queued signal handler')
            finally:
                self._queue.task_done()
            delivered += 1

    def run_forever(self, poll_interval: float = 0.1,
                    stop_event: threading.Event = None):
        """Process events until stop_event is set. Meant to be the body
        of a dedicated dispatcher thread."""
        if stop_event is None:
            stop_event = threading.Event()
        self._stop_event = stop_event
        while not stop_event.is_set():
            self.process(block=True, timeout=poll_interval)

    def stop(self):
        stop_event = getattr(self, '_stop_event', None)
        if stop_event is not None:
            stop_event.set()

    def join(self):
        """Block until all events posted so far have been processed.
        Only call this from a thread that is NOT responsible for
        processing the queue, or it will deadlock."""
        self._queue.join()

    def empty(self) -> bool:
        return self._queue.empty()


class AsyncioDispatcher:
    """Deliver signal handlers onto an asyncio event loop.

    Create one per loop, on the loop (it captures
    asyncio.get_running_loop(); pass ``loop=`` explicitly to construct
    it elsewhere), then pass it as the second argument to
    Signal.connect() — the handler then always runs on the loop's
    thread, regardless of which thread emits::

        dispatcher = AsyncioDispatcher()
        pipeline.execution_finished.connect(on_finished, dispatcher)

    A handler that is a coroutine function is scheduled as a task on
    the loop instead of being called synchronously; its exceptions are
    logged, like any other handler's. Deliveries posted after the loop
    has closed are dropped (with a debug log) rather than raised into
    the emitting thread.
    """

    def __init__(self, loop=None):
        import asyncio
        if loop is None:
            # Raises RuntimeError when called outside a running loop —
            # better now than a dispatcher bound to nothing.
            loop = asyncio.get_running_loop()
        self._loop = loop

    @property
    def loop(self):
        return self._loop

    def _post(self, conn: Connection, args: tuple):
        try:
            self._loop.call_soon_threadsafe(self._deliver, conn, args)
        except RuntimeError:
            # Loop closed (application shutting down while a worker
            # still emits). Dropping is the EventQueue-equivalent of
            # nobody draining anymore.
            logger.debug(
                'AsyncioDispatcher: event loop closed; dropping event')

    def _deliver(self, conn: Connection, args: tuple):
        # Runs on the loop thread.
        if not conn.alive:
            return
        import asyncio
        import inspect
        try:
            result = conn.callback(*args)
        except Exception:
            logger.exception(
                'Unhandled exception in asyncio signal handler')
            return
        if inspect.iscoroutine(result):
            task = asyncio.ensure_future(result, loop=self._loop)
            task.add_done_callback(self._log_task_exception)

    @staticmethod
    def _log_task_exception(task):
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            logger.error('Unhandled exception in async signal handler',
                         exc_info=exception)
