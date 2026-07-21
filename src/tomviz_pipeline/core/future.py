###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""ExecutionFuture: a handle on one pipeline execution. Mirrors the C++
ExecutionFuture (finished/canceled signals, succeeded flag) and adds
wait() since Python callers have no event loop to park on."""

from __future__ import annotations

import threading

from .events import Signal


class ExecutionFuture:
    """Handle for a single call to Pipeline.execute*().

    Signals:
      finished(future)  — execution ended (success or not)
      canceled(future)  — execution was canceled (finished also fires)

    A future returned while the pipeline is paused never started and may
    never finish; is_started() distinguishes that case.
    """

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._started = False
        self._succeeded = False
        self._was_canceled = False
        self.finished = Signal('finished')
        self.canceled = Signal('canceled')

    # ---- inspection ------------------------------------------------------

    def is_started(self) -> bool:
        return self._started

    def is_finished(self) -> bool:
        return self._event.is_set()

    def succeeded(self) -> bool:
        return self._event.is_set() and self._succeeded

    def was_canceled(self) -> bool:
        return self._was_canceled

    def wait(self, timeout: float = None) -> bool:
        """Block until the execution finishes. Returns False on timeout.
        Do not call from a thread that must service the executor's event
        queue, or from within a signal handler of the same execution."""
        return self._event.wait(timeout)

    # ---- resolution (called by Pipeline/executors) -----------------------

    def _mark_started(self):
        self._started = True

    def _finish(self, success: bool, canceled: bool = False):
        with self._lock:
            if self._event.is_set():
                return
            self._succeeded = bool(success) and not canceled
            self._was_canceled = canceled
            self._event.set()
        if canceled:
            self.canceled.emit(self)
        self.finished.emit(self)
