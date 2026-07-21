###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Pipeline-level executors: walk an execution plan and run each node.

DefaultExecutor runs the plan synchronously on the calling thread.
ThreadedExecutor runs it on a worker thread with cooperative cancellation
at node boundaries, mirroring the C++ ThreadedExecutor semantics
(pending-run replacement, cancel_and_wait, breakpoints).

Where each *individual node* runs is decided by the orthogonal
NodeExecutor layer (in-process by default, subprocess for nodes with an
ExternalNodeExecutor)."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from .events import EventQueue, Signal
from .future import ExecutionFuture
from .node import (
    NodeExecState,
    NodeState,
    SinkNode,
    SourceNode,
    TransformNode,
)
from .node_executor import InternalNodeExecutor

logger = logging.getLogger('tomviz_pipeline')


class PipelineExecutor:
    """Base class for pipeline-level executors.

    Signals (emitted on the executing thread — connect through an
    EventQueue to marshal them elsewhere):
      node_execution_started(node)
      node_execution_finished(node, success)
      execution_complete(success)
      canceled()
      breakpoint_reached(node)
    """

    def __init__(self, pipeline=None, progress=None,
                 honor_breakpoints: bool = True):
        self.pipeline = pipeline
        self.progress = progress
        self.honor_breakpoints = honor_breakpoints
        self._cancel_requested = threading.Event()
        self._current_node = None

        self.node_execution_started = Signal('node_execution_started')
        self.node_execution_finished = Signal('node_execution_finished')
        self.execution_complete = Signal('execution_complete')
        self.canceled = Signal('canceled')
        self.breakpoint_reached = Signal('breakpoint_reached')

    # ---- interface -------------------------------------------------------

    def submit(self, plan, future: ExecutionFuture = None) -> ExecutionFuture:
        """Run an execution plan (a topo-sorted node list, normally built
        by Pipeline.execution_plan). Returns the future resolving this
        run; a blocking executor resolves it before returning."""
        raise NotImplementedError

    def cancel(self):
        """Request cooperative cancellation: the plan stops at the next
        node boundary, and the currently running node is asked to stop
        (observed by operators that poll their cancel flag). A node that
        never polls runs to completion."""
        self._cancel_requested.set()
        node = self._current_node
        if node is not None:
            node.cancel_execution()

    def is_running(self) -> bool:
        return False

    def cancel_and_wait(self, timeout: float = None):
        self.cancel()

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def is_source(node) -> bool:
        return isinstance(node, SourceNode)

    @staticmethod
    def is_transform(node) -> bool:
        return isinstance(node, TransformNode)

    @staticmethod
    def is_sink(node) -> bool:
        return isinstance(node, SinkNode)

    # ---- shared machinery ------------------------------------------------

    def _run_plan(self, plan, skip_current: bool = False,
                  manage_residency: bool = True):
        """Walk the plan. Returns (failed, canceled, breakpoint_hit).

        skip_current=True reproduces the legacy full-order walk (used by
        the blocking execute() API and the CLI runner): Current nodes are
        skipped at runtime instead of being pruned by the planner.

        manage_residency=True enables the C++ executors' in-flight
        handle model: before a node runs, its inputs' upstream payloads
        are take()n from the producer ports (lazily, on the first
        consumer; shared across all in-plan consumers) and delivered as
        handles. When the plan ends the in-flight table drops, so
        transient outputs nobody retained release and persistent-OnDisk
        outputs spill to their cache files. Leaf outputs are never taken
        and stay pinned on their ports. The legacy path runs with
        manage_residency=False and simply leaves every payload pinned.
        """
        failed = False
        canceled = False
        breakpoint_hit = False
        # Strong handles for payloads taken off producer ports during
        # this plan, keyed by id(OutputPort). Dropped at end of plan.
        inflight: dict = {}

        try:
            for node in plan:
                if self._cancel_requested.is_set():
                    canceled = True
                    break

                if isinstance(node, SinkNode) and not node.executable:
                    # Inert sinks are never executed and their state is
                    # left untouched so app-side reloads still consume
                    # them. Their inputs are also never taken, so a
                    # producer feeding only sinks keeps its data pinned.
                    continue

                if self.honor_breakpoints and node.breakpoint:
                    breakpoint_hit = True
                    self.breakpoint_reached.emit(node)
                    break

                # If any upstream node ended up Stale, cascade.
                if any(u.state == NodeState.Stale
                       for u in node.upstream_nodes()):
                    node.state = NodeState.Stale
                    continue

                if skip_current and node.state == NodeState.Current:
                    continue

                if manage_residency:
                    self._deliver_inputs(node, inflight)
                try:
                    ok = self._run_node(node)
                finally:
                    if manage_residency:
                        for port in node.input_ports():
                            port.clear_handle()
                if not ok:
                    if (self._cancel_requested.is_set()
                            or node.is_cancel_requested()):
                        canceled = True
                        break
                    failed = True
                    # Keep walking: sibling branches that don't depend on
                    # the failed node still run.
        finally:
            inflight.clear()

        return failed, canceled, breakpoint_hit

    @staticmethod
    def _deliver_inputs(node, inflight: dict):
        """Deliver upstream payload handles to a node's input ports,
        taking each producer port's payload at most once per plan (the
        first consumer triggers the take; later consumers share the same
        handle)."""
        for port in node.input_ports():
            link = port.link
            if link is None:
                continue
            source = link.from_port
            key = id(source)
            handle = inflight.get(key)
            if handle is None:
                handle = source.take()
                if handle is None:
                    # Nothing anywhere (not even on disk). The node's
                    # execute() observes the missing input and fails as
                    # before.
                    continue
                inflight[key] = handle
            port.set_handle(handle)

    def _run_node(self, node) -> bool:
        node.reset_execution_flags()
        node.exec_state = NodeExecState.Running
        self._current_node = node

        if self.progress is not None:
            self.progress.started(node.id)
            node.progress = self.progress

        self.node_execution_started.emit(node)

        label = node.label or type(node).__name__
        logger.info("Executing '%s' (id=%d)", label, node.id)
        node_executor = node.node_executor or InternalNodeExecutor.instance()
        try:
            ok = node_executor.execute(node)
        except Exception:
            logger.exception("Node '%s' (id=%d) raised", label, node.id)
            ok = False

        self._current_node = None

        if self.progress is not None:
            self.progress.finished(node.id)

        if ok:
            node.state = NodeState.Current
            node.exec_state = NodeExecState.Idle
        elif node.is_cancel_requested():
            # A canceled node did not touch its outputs; leave its state
            # alone so the next plan simply re-includes it.
            node.exec_state = NodeExecState.Canceled
            logger.info("Node '%s' (id=%d) canceled", label, node.id)
        else:
            node.exec_state = NodeExecState.Failed
            node.state = NodeState.Stale
            for downstream in node.downstream_nodes():
                downstream.mark_stale()
            logger.error("Node '%s' (id=%d) failed", label, node.id)

        self.node_execution_finished.emit(node, ok)
        self._post_node_barrier()
        return ok

    def _post_node_barrier(self):
        """Hook run after each node's signals are emitted. Overridden by
        ThreadedExecutor to wait for the application's event queue."""

    def _finish_run(self, future: ExecutionFuture, failed: bool,
                    canceled: bool, breakpoint_hit: bool) -> bool:
        success = not failed and not canceled and not breakpoint_hit
        if canceled:
            self.canceled.emit()
        self.execution_complete.emit(success)
        future._finish(success, canceled=canceled)
        return success


class DefaultExecutor(PipelineExecutor):
    """Blocking executor: runs the plan synchronously on the calling
    thread. Also provides the legacy execute() API used by the CLI
    runner, which walks the full topological order, skipping sinks and
    Current nodes at runtime."""

    def __init__(self, pipeline=None, progress=None,
                 honor_breakpoints: bool = True):
        super().__init__(pipeline, progress, honor_breakpoints)
        self._running = False

    def execute(self, plan=None) -> bool:
        """Legacy blocking API. Returns True if no node failed. With no
        plan, walks the whole pipeline in topological order (raises
        RuntimeError on a cycle), skipping already-Current nodes, and
        leaves transient outputs in place for leaf writers."""
        if plan is None:
            plan = self.pipeline.execution_order()
        failed, _, _ = self._execute(plan, ExecutionFuture(),
                                     skip_current=True,
                                     manage_residency=False)
        return not failed

    def submit(self, plan, future: ExecutionFuture = None) -> ExecutionFuture:
        if future is None:
            future = ExecutionFuture()
        self._execute(plan, future, skip_current=False,
                      manage_residency=True)
        return future

    def is_running(self) -> bool:
        return self._running

    def _execute(self, plan, future, skip_current, manage_residency):
        if self._running:
            raise RuntimeError(
                'DefaultExecutor.execute is not re-entrant; use '
                'ThreadedExecutor for concurrent executions')
        self._running = True
        self._cancel_requested.clear()
        future._mark_started()
        try:
            failed, canceled, breakpoint_hit = self._run_plan(
                plan, skip_current=skip_current,
                manage_residency=manage_residency)
        finally:
            self._running = False
        self._finish_run(future, failed, canceled, breakpoint_hit)
        return failed, canceled, breakpoint_hit


class ThreadedExecutor(PipelineExecutor):
    """Runs each plan on a dedicated worker thread.

    Semantics ported from the C++ ThreadedExecutor:
    - submit() while a plan is running does not queue behind it: the new
      plan becomes *pending*, the in-flight run is canceled at its next
      node boundary, and the pending plan starts when the worker exits.
      A pending plan displaced by yet another submit() resolves its
      future as canceled.
    - cancel() stops at the next node boundary and forwards the request
      to the currently running node.
    - cancel_and_wait() additionally joins the worker; while waiting it
      keeps servicing sync_queue so a worker blocked on the per-node
      barrier can unwind.
    - sync_queue (optional EventQueue): after each node, the worker
      blocks until the application thread has drained the queue — the
      same barrier the C++ executor uses so per-node UI updates complete
      before the next node starts. Handlers must be connected through
      that queue for this to be meaningful.
    """

    def __init__(self, pipeline=None, progress=None,
                 honor_breakpoints: bool = True,
                 sync_queue: Optional[EventQueue] = None):
        super().__init__(pipeline, progress, honor_breakpoints)
        self.sync_queue = sync_queue
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._pending = None  # (plan, future) waiting for the worker

    def submit(self, plan, future: ExecutionFuture = None) -> ExecutionFuture:
        if future is None:
            future = ExecutionFuture()
        displaced = None
        with self._lock:
            if self.is_running():
                if self._pending is not None:
                    displaced = self._pending[1]
                self._pending = (plan, future)
                self._request_cancel()
            else:
                self._start(plan, future)
        if displaced is not None:
            displaced._finish(False, canceled=True)
        return future

    def cancel(self):
        pending = None
        with self._lock:
            if self._pending is not None:
                pending = self._pending[1]
                self._pending = None
            self._request_cancel()
        if pending is not None:
            pending._finish(False, canceled=True)

    def cancel_and_wait(self, timeout: float = None):
        self.cancel()
        thread = self._thread
        if (thread is None or not thread.is_alive()
                or thread is threading.current_thread()):
            return
        waited = 0.0
        step = 0.05
        while thread.is_alive():
            if self.sync_queue is not None:
                # Keep draining events so a worker parked on the
                # per-node barrier can unwind (mirrors the C++
                # destructor pumping queued MetaCall events).
                self.sync_queue.process()
            thread.join(step)
            waited += step
            if timeout is not None and waited >= timeout:
                break

    def is_running(self) -> bool:
        thread = self._thread
        return ((thread is not None and thread.is_alive())
                or self._pending is not None)

    # ---- internals -------------------------------------------------------

    def _request_cancel(self):
        self._cancel_requested.set()
        node = self._current_node
        if node is not None:
            node.cancel_execution()

    def _start(self, plan, future):
        self._cancel_requested.clear()
        self._thread = threading.Thread(
            target=self._worker, args=(plan, future),
            name='tomviz-pipeline-executor', daemon=True)
        self._thread.start()

    def _worker(self, plan, future):
        future._mark_started()
        try:
            failed, canceled, breakpoint_hit = self._run_plan(
                plan, skip_current=False, manage_residency=True)
        except Exception:
            logger.exception('Unexpected error in pipeline worker')
            failed, canceled, breakpoint_hit = True, False, False
        self._finish_run(future, failed, canceled, breakpoint_hit)
        with self._lock:
            pending = self._pending
            self._pending = None
            if pending is not None:
                self._start(*pending)

    def _post_node_barrier(self):
        if self.sync_queue is not None:
            self.sync_queue.join()
