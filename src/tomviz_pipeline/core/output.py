###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""What kernels print, as their node's output.

An executor built with ``capture_output=True`` runs each in-process node
inside :func:`captured`: whatever the kernel writes to ``sys.stdout`` or
``sys.stderr`` from the executing thread is collected and handed to the
node's ``output_written`` signal about once a second, and once more when
the kernel returns (before the node's finished signals). The text still
reaches the original streams too, so a terminal shows it as before.

Capturing replaces ``sys.stdout`` and ``sys.stderr`` with stand-ins that
route by thread: writes from a thread running a captured node go to that
node, every other write passes straight through. They are installed the
first time a node is captured and stay (re-installed if something swapped
the streams since). What bypasses ``sys.stdout`` is not seen: output of
C code writing to the file descriptors directly, and of threads or
subprocesses a kernel starts itself.

In a child process run by ``ExternalNodeExecutor``, the CLI captures the
same way and forwards the output over the progress channel as ``output``
messages, which the parent turns back into the node's signal.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from typing import Callable

logger = logging.getLogger('tomviz_pipeline')

#: Seconds between deliveries of a running kernel's output.
FLUSH_INTERVAL = 1.0
#: Characters held between deliveries; the rest is counted as dropped.
MAX_PENDING = 1_000_000

# Thread id -> the capture of the node running on it
_captures: dict[int, '_Capture'] = {}
_install_lock = threading.Lock()
_flusher: threading.Thread | None = None


class _RoutingStream:
    """Stands in for ``sys.stdout`` / ``sys.stderr``: a write from a
    thread with a capture is also given to it; everything else about the
    stream is the original's."""

    def __init__(self, original):
        self._original = original

    def write(self, text):
        capture = _captures.get(threading.get_ident())
        if capture is not None:
            capture.add(text)
        return self._original.write(text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


class _Capture:
    """The output of one node run, delivered in order: ``add`` never
    waits for a delivery, ``flush`` hands what is pending to ``emit``."""

    def __init__(self, emit: Callable[[str], None]):
        self._emit = emit
        self._pending: list[str] = []
        self._size = 0
        self._dropped = 0
        self._lock = threading.Lock()  # the pending text
        self._emit_lock = threading.Lock()  # one delivery at a time

    def add(self, text):
        if not isinstance(text, str) or not text:
            return
        with self._lock:
            if self._size >= MAX_PENDING:
                self._dropped += len(text)
                return
            self._pending.append(text)
            self._size += len(text)

    def flush(self):
        with self._emit_lock:
            with self._lock:
                text = ''.join(self._pending)
                dropped = self._dropped
                self._pending.clear()
                self._size = 0
                self._dropped = 0
            if dropped:
                if text and not text.endswith('\n'):
                    text += '\n'
                text += f'[{dropped} characters of output dropped]\n'
            if not text:
                return
            try:
                self._emit(text)
            except Exception:
                logger.exception('Delivering a node\'s output failed')


def _install():
    """Put the routing stand-ins in place (again, if something replaced
    them since)."""
    global _flusher
    with _install_lock:
        for name in ('stdout', 'stderr'):
            stream = getattr(sys, name)
            if stream is not None and not isinstance(stream, _RoutingStream):
                setattr(sys, name, _RoutingStream(stream))
        if _flusher is None:
            _flusher = threading.Thread(
                target=_flush_loop, name='tomviz-pipeline-output',
                daemon=True)
            _flusher.start()


def _flush_loop():
    while True:
        time.sleep(FLUSH_INTERVAL)
        for capture in list(_captures.values()):
            capture.flush()


@contextlib.contextmanager
def captured(emit: Callable[[str], None]):
    """Capture what the calling thread prints while the block runs,
    handing it to ``emit(text)`` about every ``FLUSH_INTERVAL`` seconds
    (from a helper thread) and when the block ends (from this one)."""
    _install()
    ident = threading.get_ident()
    capture = _Capture(emit)
    previous = _captures.get(ident)
    _captures[ident] = capture
    try:
        yield
    finally:
        if previous is None:
            _captures.pop(ident, None)
        else:
            _captures[ident] = previous
        capture.flush()
