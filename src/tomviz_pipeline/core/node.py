###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Core graph types: Node, SourceNode, TransformNode, SinkNode, Link, Port,
InputPort, OutputPort, PortData, NodeState, NodeExecState. Mirrors the
class shapes of the C++ pipeline library in tomviz."""

from __future__ import annotations

import enum
import logging
import os
import threading
import weakref
from types import MappingProxyType
from typing import Any, Mapping, Optional

from .events import Signal
from .persistence import (
    DataLocation,
    PersistenceMode,
    PipelineSettings,
    TransformPersistenceDefault,
    create_cache_file,
    persistence_mode_from_string,
    read_port_data_from_file,
    write_port_data_to_file,
)

logger = logging.getLogger('tomviz_pipeline')

DEFAULT_AUTO_EXECUTE_INTERVAL_SECONDS = 30


class NodeState(enum.Enum):
    """Dirty-tracking state. New = never ran, Stale = inputs or parameters
    changed since the last run, Current = outputs are up to date."""

    New = 'New'
    Stale = 'Stale'
    Current = 'Current'


class NodeExecState(enum.Enum):
    """Transient execution indicator, orthogonal to NodeState."""

    Idle = 'Idle'
    Running = 'Running'
    Failed = 'Failed'
    Canceled = 'Canceled'


class PortData:
    """Typed payload flowing through a port. Mirrors the C++ PortData which
    holds a std::any plus a PortType. We carry an arbitrary Python payload
    (typically a tomviz_pipeline.dataset.Dataset) and a string port type."""

    __slots__ = ('payload', 'port_type')

    def __init__(self, payload: Any, port_type: str = 'ImageData'):
        self.payload = payload
        self.port_type = port_type

    def __repr__(self):
        return f'PortData(type={self.port_type!r})'


class Port:
    def __init__(self, name: str, port_type: str):
        self.name = name
        self.port_type = port_type
        self.node: Optional[Node] = None


class PortDataHandle:
    """Strong reference to a port's payload — the Python analogue of the
    C++ shared_ptr<PortData>. Hold it (or any reference to it) to keep
    the payload materialized in memory; when the last reference drops,
    the owning port's persistence policy decides what happens: release
    (transient), keep pinned anyway (persistent InMemory ports never let
    go of theirs), or spill to a disk cache file (persistent OnDisk)."""

    __slots__ = ('data', '__weakref__')

    def __init__(self, data: PortData):
        self.data = data


def _dead_ref():
    return None


def _remove_cache_file(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass


class _DiskCache:
    """The cache-file path of an OutputPort, held separately so the
    port's finalizer can be registered at construction and pick up
    whatever path exists when the port dies. ``dead`` is set by that
    finalizer: a handle released in the same garbage collection may
    still resolve its weakref to the port (Python 3.15 no longer clears
    weakrefs before running other finalizers) and must not spill for a
    port that is already gone."""

    __slots__ = ('path', 'dead')

    def __init__(self):
        self.path: Optional[str] = None
        self.dead = False

    def remove(self):
        path, self.path = self.path, None
        if path is not None:
            _remove_cache_file(path)

    def finalize(self):
        self.dead = True
        self.remove()


class InputPort(Port):
    """Accepts a single incoming Link. accepted_types is an informational
    list — type validation is intentionally lax in the Python runtime.

    During plan execution the pipeline executor delivers the upstream
    payload as a PortDataHandle (set_handle) so transient upstream data
    stays alive exactly for the consumers that need it; outside a plan,
    data() falls back to a non-loading peek through the link."""

    def __init__(self, name: str, accepted_types):
        if isinstance(accepted_types, str):
            accepted_types = [accepted_types]
        self.accepted_types = list(accepted_types)
        super().__init__(name, self.accepted_types[0])
        self.link: Optional[Link] = None
        self._handle: Optional[PortDataHandle] = None

    def set_handle(self, handle: Optional[PortDataHandle]):
        self._handle = handle

    def clear_handle(self):
        self._handle = None

    def handle(self) -> Optional[PortDataHandle]:
        return self._handle

    def has_data(self) -> bool:
        if self._handle is not None:
            return True
        return self.link is not None and self.link.from_port.has_data()

    def data(self) -> Optional[PortData]:
        if self._handle is not None:
            return self._handle.data
        if self.link is None:
            return None
        return self.link.from_port.data()


class OutputPort(Port):
    """Holds the node's result for one output.

    Persistence model (mirrors the C++ OutputPort):

    - ``persistent == False`` (transient): the payload lives only while
      some PortDataHandle is held — by the port itself until an executor
      take()s it for the first in-plan consumer, then by the plan's
      in-flight table. When the last handle drops the payload is
      released; the planner re-includes the producer when a consumer
      needs the data again.
    - persistent + PersistenceMode.InMemory: the port pins a strong
      handle forever (legacy behavior).
    - persistent + PersistenceMode.OnDisk: like transient, but when the
      last handle drops the payload is spilled to a temp cache file
      ($TOMVIZ_PORT_CACHE_DIR or the system temp dir) and lazily
      reloaded by materialize()/take(). Payloads the serializer cannot
      handle are dropped — the planner re-runs the producer.

    data() is a non-loading peek (returns None for data that only lives
    on disk); materialize() is the loading read — the payload stays
    alive only while the returned handle is held.

    Signals:
      data_changed(port)
      data_location_changed(port, DataLocation)
    """

    def __init__(self, name: str, port_type: str, persistent: bool = True,
                 persistence_mode: PersistenceMode = PersistenceMode.InMemory):
        super().__init__(name, port_type)
        self._persistent = bool(persistent)
        self._mode = persistence_mode
        self._strong: Optional[PortDataHandle] = None
        self._weak = _dead_ref
        self._generation = 0
        self._disk_lock = threading.Lock()
        self._on_disk = False
        self._disk = _DiskCache()
        # Registered now rather than at spill time: a handle can be
        # released while the port itself is being garbage collected, and
        # a finalizer created during that collection never runs, leaving
        # the cache file behind.
        weakref.finalize(self, _DiskCache.finalize, self._disk)
        self.outgoing_links: list[Link] = []
        self.data_changed = Signal('data_changed')
        self.data_location_changed = Signal('data_location_changed')

    # ---- persistence configuration -------------------------------------

    @property
    def persistent(self) -> bool:
        return self._persistent

    @persistent.setter
    def persistent(self, value: bool):
        value = bool(value)
        if value == self._persistent:
            return
        self._persistent = value
        self._reconcile_persistence()

    @property
    def persistence_mode(self) -> PersistenceMode:
        return self._mode

    @persistence_mode.setter
    def persistence_mode(self, mode: PersistenceMode):
        if mode == self._mode:
            return
        self._mode = mode
        self._reconcile_persistence()

    def _reconcile_persistence(self):
        """Enforce the (persistent, mode) invariants immediately after a
        change. Handle drops happen outside the disk lock — the release
        callback takes it."""
        if not self._persistent:
            # Transient = no persistence at all: drop the pin (the
            # release callback sees persistent False and just lets go)
            # and any disk cache.
            self._strong = None
            self._drop_disk_cache()
        elif self._mode == PersistenceMode.InMemory:
            # Pin now: recover a live handle, else load from disk; then
            # the disk cache is no longer needed.
            if self._strong is None:
                handle = self._weak()
                if handle is None:
                    handle = self._reload_from_disk()
                self._strong = handle
            self._drop_disk_cache()
            if self._strong is not None:
                self.data_location_changed.emit(self, DataLocation.InMemory)
        else:
            # Persistent OnDisk: release the pin. If the port was the
            # sole holder the release callback spills to disk right now;
            # if a consumer still holds a handle, the spill happens when
            # they drop it (the callback reads the current mode).
            self._strong = None

    # ---- data ------------------------------------------------------------

    def data_location(self) -> DataLocation:
        if self._weak() is not None:
            return DataLocation.InMemory
        with self._disk_lock:
            if self._on_disk:
                return DataLocation.OnDisk
        return DataLocation.Nowhere

    def has_data(self) -> bool:
        return self.data_location() != DataLocation.Nowhere

    def data(self) -> Optional[PortData]:
        """Non-loading peek: the payload if it is materialized anywhere,
        None otherwise (including data that only lives on disk — use
        materialize() for a loading read)."""
        handle = self._weak()
        if handle is not None:
            return handle.data
        return None

    def materialize(self) -> Optional[PortDataHandle]:
        """Loading read: return a handle to the payload, reloading a
        persistent-OnDisk port's data from its cache file if needed. The
        payload stays materialized only while the returned handle is
        held; dropping it re-applies the persistence policy."""
        handle = self._weak()
        if handle is not None:
            return handle
        if self._persistent and self._mode == PersistenceMode.OnDisk:
            return self._reload_from_disk()
        return None

    def take(self) -> Optional[PortDataHandle]:
        """Hand the port's payload handle to the caller (executors call
        this for the first in-plan consumer). On transient and
        persistent-OnDisk ports this moves the pin out — residency is
        then governed by whoever holds the returned handle; a
        persistent-InMemory port shares its handle but keeps its pin.
        Falls back to any live handle, then to a disk reload."""
        handle = self._strong
        if handle is not None:
            if (self._persistent
                    and self._mode == PersistenceMode.InMemory):
                return handle
            self._strong = None
            return handle
        handle = self._weak()
        if handle is not None:
            return handle
        return self._reload_from_disk()

    def set_data(self, data: Optional[PortData]):
        if data is None:
            # Compatibility alias: setting no data clears the port.
            self.clear_data()
            return
        # Invalidate release callbacks for the previous payload before
        # dropping our pin on it, so a stale spill can't clobber the new
        # data's cache.
        self._generation += 1
        with self._disk_lock:
            self._on_disk = False
        self._strong = self._wrap(data)
        self.data_changed.emit(self)
        self.data_location_changed.emit(self, DataLocation.InMemory)

    def clear_data(self):
        self._generation += 1
        self._strong = None
        self._weak = _dead_ref
        with self._disk_lock:
            self._on_disk = False
        self.data_changed.emit(self)
        self.data_location_changed.emit(self, DataLocation.Nowhere)

    # ---- internals -------------------------------------------------------

    def _wrap(self, data: PortData) -> PortDataHandle:
        handle = PortDataHandle(data)
        weakref.finalize(handle, OutputPort._on_handle_released,
                         weakref.ref(self), self._generation, data)
        self._weak = weakref.ref(handle)
        return handle

    @staticmethod
    def _on_handle_released(port_ref, generation: int, data: PortData):
        # Fires when the last reference to a PortDataHandle drops. Reads
        # the port's CURRENT policy (not the one at creation time) —
        # that is what makes runtime mode switching correct.
        port = port_ref()
        if (port is None or port._disk.dead
                or port._generation != generation):
            return
        if port._persistent and port._mode == PersistenceMode.OnDisk:
            port._swap_to_disk(data)

    def _swap_to_disk(self, data: PortData):
        with self._disk_lock:
            if self._disk.path is None:
                try:
                    self._disk.path = create_cache_file()
                except OSError:
                    logger.exception(
                        "Failed to create cache file for port '%s'",
                        self.name)
                    return
            ok = write_port_data_to_file(data, self._disk.path)
            self._on_disk = ok
        if ok:
            self.data_location_changed.emit(self, DataLocation.OnDisk)
        # On failure the payload is simply lost; the planner sees
        # has_data() == False and re-runs the producer.

    def _reload_from_disk(self) -> Optional[PortDataHandle]:
        with self._disk_lock:
            if not self._on_disk or self._disk.path is None:
                return None
            data = read_port_data_from_file(self._disk.path)
            if data is None:
                return None
            return self._wrap(data)

    def _drop_disk_cache(self):
        with self._disk_lock:
            self._on_disk = False
            self._disk.remove()


class Link:
    def __init__(self, from_port: OutputPort, to_port: InputPort):
        self.from_port = from_port
        self.to_port = to_port


class Node:
    """Base graph node. Holds typed input/output ports, a label, a state,
    and a free-form properties dict. Subclasses override execute().

    Signals:
      state_changed(node, NodeState)
      exec_state_changed(node, NodeExecState)
      parameters_applied(node, changed) — set_parameters() was called
      parameters_updated(node, changed) — the node's own implementation
        changed parameter values during a run (apply_parameter_updates)
    """

    type_name: str = ''

    def __init__(self):
        self.id: int = -1
        self.label: str = ''
        self._state: NodeState = NodeState.New
        self._exec_state: NodeExecState = NodeExecState.Idle
        self.breakpoint: bool = False
        self.properties: dict[str, Any] = {}
        # Node configuration that drives execute(), kept in its own
        # store so parameter names can never collide with node
        # internals (label, state, progress, ...). Read through
        # `parameters` / parameter(); write through set_parameters(),
        # which marks the node stale. Keys use the schema-v2 spelling
        # (e.g. 'minValue') so subclasses serialize them 1:1.
        # `properties` is different: free-form application metadata
        # that has no effect on execution.
        self._parameters: dict[str, Any] = {}
        self.type_inference_sources: dict[str, str] = {}
        # Runtime-only per-node state bag, surfaced to schema-v2 kernels
        # as `self.state`. Mirrors the C++ Node::userState(): preserved
        # across executions, deliberately never serialized.
        self.user_state: dict = {}
        # Periodic execution. An application polls
        # query_should_auto_execute() every interval — through the
        # node's executor, so external nodes are asked in their
        # environment — and re-executes when the answer is True.
        self.auto_execute_enabled: bool = False
        self.auto_execute_interval_seconds: int = \
            DEFAULT_AUTO_EXECUTE_INTERVAL_SECONDS
        self._input_ports: list[InputPort] = []
        self._output_ports: list[OutputPort] = []
        # Progress hooks. The pipeline executor sets `progress` before
        # execute() is called; nodes (e.g. LegacyPythonTransform) forward
        # operator progress updates through it.
        self.progress = None
        # Where this node runs: None means in-process (the executor falls
        # back to InternalNodeExecutor). An ExternalNodeExecutor instance
        # runs the node in a subprocess under a different Python env.
        self.node_executor = None
        self._cancel_event = threading.Event()
        self._complete_event = threading.Event()
        self._total_progress_steps = 0
        self._progress_step = 0
        self._progress_message = ''
        self.state_changed = Signal('state_changed')
        self.exec_state_changed = Signal('exec_state_changed')
        self.parameters_applied = Signal('parameters_applied')
        self.parameters_updated = Signal('parameters_updated')
        self.progress_maximum_changed = Signal('progress_maximum_changed')
        self.progress_step_changed = Signal('progress_step_changed')
        self.progress_message_changed = Signal('progress_message_changed')

    # ---- state -----------------------------------------------------------

    @property
    def state(self) -> NodeState:
        return self._state

    @state.setter
    def state(self, value: NodeState):
        if value == self._state:
            return
        self._state = value
        self.state_changed.emit(self, value)

    @property
    def exec_state(self) -> NodeExecState:
        return self._exec_state

    @exec_state.setter
    def exec_state(self, value: NodeExecState):
        if value == self._exec_state:
            return
        self._exec_state = value
        self.exec_state_changed.emit(self, value)

    def mark_stale(self):
        """Mark this node's outputs as out of date and cascade the mark to
        every downstream node. Maintains the planner's invariant that no
        Current node sits downstream of a non-Current one."""
        if self._state == NodeState.Stale:
            return
        # A New node has never run; it stays New, but downstream results
        # that were computed from its previous wiring still go stale.
        if self._state != NodeState.New:
            self.state = NodeState.Stale
        for downstream in self.downstream_nodes():
            downstream.mark_stale()

    # ---- parameters ------------------------------------------------------

    def _parameter_store(self) -> dict:
        """The dict backing this node's parameters. Subclasses that keep
        their parameters elsewhere (e.g. the Python-node backends)
        override this so the parameters API operates on their store."""
        return self._parameters

    @property
    def parameters(self) -> Mapping:
        """Read-only live view of the node's parameters. Mutate through
        set_parameters() so staleness is tracked."""
        return MappingProxyType(self._parameter_store())

    def parameter(self, name: str, default: Any = None) -> Any:
        return self._parameter_store().get(name, default)

    def set_parameters(self, **params: Any):
        """Update parameters, mark this node (and everything downstream)
        stale, and emit parameters_applied(node, changed) — the Python
        analogue of the C++ parametersApplied flow. A Pipeline with
        auto_execute enabled re-executes in response.

        Deserialization code must NOT use this: loaders write into the
        parameter store directly so restoring a state file neither
        cascades staleness nor triggers re-execution."""
        if not params:
            return
        self._parameter_store().update(params)
        self.mark_stale()
        self.parameters_applied.emit(self, dict(params))

    def apply_parameter_updates(self, updates: Mapping):
        """Install parameter values the node's *own implementation*
        changed while running — a schema-v2 kernel's
        ``self.set_parameter()`` — and emit
        ``parameters_updated(node, changed)`` for values that actually
        differ.

        Deliberately the quiet counterpart of set_parameters(): nothing
        is marked stale and ``parameters_applied`` is not emitted. The
        run that made the change is deemed to have consumed the new
        values, and a re-execution request from inside a run would
        cancel that very run under ThreadedExecutor (and re-enter
        execute() through Pipeline.auto_execute). An application
        connects ``parameters_updated`` to refresh its parameter UI; a
        periodic-execution hook that wants its new values used returns
        True from should_auto_execute."""
        store = self._parameter_store()
        changed = {
            name: value for name, value in updates.items()
            if name not in store or store[name] != value
        }
        if not changed:
            return
        store.update(changed)
        self.parameters_updated.emit(self, dict(changed))

    # ---- ports -----------------------------------------------------------

    def add_input(self, name: str, accepted_types) -> InputPort:
        port = InputPort(name, accepted_types)
        port.node = self
        self._input_ports.append(port)
        return port

    def add_output(self, name: str, port_type: str,
                   persistent: Optional[bool] = None,
                   persistence_mode: Optional[PersistenceMode] = None
                   ) -> OutputPort:
        """Add an output port. With persistent/persistence_mode omitted,
        the node class's default policy applies (persistent InMemory for
        plain nodes and sources; the PipelineSettings transform default
        for TransformNode). Explicit arguments always win."""
        if persistent is None and persistence_mode is None:
            persistent, persistence_mode = self._default_output_persistence()
        elif persistent is None:
            # An explicit mode implies a persistent port.
            persistent = True
        elif persistence_mode is None:
            persistence_mode = PersistenceMode.InMemory
        port = OutputPort(name, port_type, persistent=persistent,
                          persistence_mode=persistence_mode)
        port.node = self
        self._output_ports.append(port)
        return port

    def _default_output_persistence(self):
        """(persistent, mode) applied to outputs added without explicit
        persistence arguments."""
        return True, PersistenceMode.InMemory

    def input_ports(self) -> list[InputPort]:
        return list(self._input_ports)

    def output_ports(self) -> list[OutputPort]:
        return list(self._output_ports)

    def input_port(self, name: str) -> Optional[InputPort]:
        for p in self._input_ports:
            if p.name == name:
                return p
        return None

    def output_port(self, name: str) -> Optional[OutputPort]:
        for p in self._output_ports:
            if p.name == name:
                return p
        return None

    def upstream_nodes(self) -> list['Node']:
        nodes: list[Node] = []
        for port in self._input_ports:
            if port.link is not None and port.link.from_port.node is not None:
                nodes.append(port.link.from_port.node)
        return nodes

    def downstream_nodes(self) -> list['Node']:
        nodes: list[Node] = []
        for port in self._output_ports:
            for link in port.outgoing_links:
                if link.to_port.node is not None:
                    nodes.append(link.to_port.node)
        return nodes

    # ---- execution -------------------------------------------------------

    def execute(self) -> bool:
        """Run this node. Returns True on success, False on failure.
        Default implementation is a no-op success."""
        return True

    def cancel_execution(self):
        """Request cooperative cancellation of a running execution. The
        request is observed by code that polls is_cancel_requested()
        (operator wrappers do) and is forwarded to the node's executor so
        an external subprocess can be signaled too."""
        self._cancel_event.set()
        if self.node_executor is not None:
            self.node_executor.cancel(self)

    def complete_execution(self):
        """Request an early, successful stop of an iterative execution."""
        self._complete_event.set()
        if self.node_executor is not None:
            self.node_executor.complete(self)

    def is_cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def is_complete_requested(self) -> bool:
        return self._complete_event.is_set()

    def reset_execution_flags(self):
        """Called by executors immediately before running the node."""
        self._cancel_event.clear()
        self._complete_event.clear()

    # ---- progress --------------------------------------------------------
    # Mirrors the C++ Node progress API. Set by whatever runs the node —
    # an in-process operator wrapper or the ExternalNodeExecutor
    # forwarding subprocess messages; observed via the signals.

    def set_total_progress_steps(self, value: int):
        self._total_progress_steps = int(value)
        self.progress_maximum_changed.emit(self, self._total_progress_steps)

    def total_progress_steps(self) -> int:
        return self._total_progress_steps

    def set_progress_step(self, value: int):
        self._progress_step = int(value)
        self.progress_step_changed.emit(self, self._progress_step)

    def progress_step(self) -> int:
        return self._progress_step

    def set_progress_message(self, message: str):
        self._progress_message = message
        self.progress_message_changed.emit(self, message)

    def progress_message(self) -> str:
        return self._progress_message

    def reset_progress(self):
        self._total_progress_steps = 0
        self._progress_step = 0
        self._progress_message = ''

    # ---- periodic execution ----------------------------------------------

    def query_should_auto_execute(self) -> bool:
        """Ask the node's implementation whether a periodic execution
        should happen now. Runs user code (the schema-v2 kernel's
        should_auto_execute hook), so never call it while the node is
        executing. The base implementation never requests a re-run."""
        return False

    # ---- serialization ---------------------------------------------------

    def serialize(self) -> dict:
        """Produce the schema-v2 JSON entry for this node, minus the `id`
        and `type` keys which the pipeline serializer injects. Mirrors the
        C++ Node::serialize."""
        data: dict[str, Any] = {'label': self.label}
        if self._state != NodeState.New:
            data['state'] = self._state.value
        if self.breakpoint:
            data['breakpoint'] = True
        if self.properties:
            data['properties'] = dict(self.properties)
        if self.type_inference_sources:
            data['typeInferenceSources'] = dict(self.type_inference_sources)
        if (self.node_executor is not None
                and self.node_executor.type_name):
            data['executor'] = self.node_executor.serialize()
        # user_state is runtime-only scratch space and stays out.
        if (self.auto_execute_enabled
                or self.auto_execute_interval_seconds
                != DEFAULT_AUTO_EXECUTE_INTERVAL_SECONDS):
            data['autoExecute'] = {
                'enabled': self.auto_execute_enabled,
                'intervalSeconds': self.auto_execute_interval_seconds,
            }
        if self._output_ports:
            ports = {}
            for port in self._output_ports:
                entry = {
                    'type': port.port_type,
                    'persistent': port.persistent,
                }
                # Written only for persistent OnDisk ports so older
                # readers keep working (parity with C++ Node::serialize).
                if (port.persistent
                        and port.persistence_mode == PersistenceMode.OnDisk):
                    entry['persistenceMode'] = 'disk'
                ports[port.name] = entry
            data['outputPorts'] = ports
        if self._input_ports:
            ports = {}
            for port in self._input_ports:
                ports[port.name] = {'type': list(port.accepted_types)}
            data['inputPorts'] = ports
        return data

    def deserialize(self, data: dict) -> bool:
        """Apply the schema-v2 JSON entry for this node. Subclasses override
        to read their type-specific fields and call super().deserialize."""
        if 'label' in data:
            self.label = data['label']
        s = data.get('state')
        if s == 'Stale':
            self._state = NodeState.Stale
        elif s == 'Current':
            self._state = NodeState.Current
        if data.get('breakpoint'):
            self.breakpoint = True
        if 'properties' in data:
            self.properties = dict(data['properties'])
        if 'typeInferenceSources' in data:
            self.type_inference_sources = dict(data['typeInferenceSources'])
        if 'autoExecute' in data:
            auto_exec = data['autoExecute'] or {}
            self.auto_execute_interval_seconds = int(auto_exec.get(
                'intervalSeconds', DEFAULT_AUTO_EXECUTE_INTERVAL_SECONDS))
            self.auto_execute_enabled = bool(auto_exec.get('enabled', False))
        # Output ports may carry a saved declaredType — apply it if present
        # so downstream type inference matches what the C++ side saved.
        if 'outputPorts' in data:
            for name, entry in data['outputPorts'].items():
                port = self.output_port(name)
                if port is None:
                    continue
                t = entry.get('type')
                if t:
                    port.port_type = t
                # persistent before persistenceMode, as in C++; an absent
                # persistenceMode key means InMemory (or transient).
                if 'persistent' in entry:
                    port.persistent = bool(entry['persistent'])
                if 'persistenceMode' in entry:
                    port.persistence_mode = persistence_mode_from_string(
                        entry['persistenceMode'])
        return True


class SourceNode(Node):
    """A node with only output ports. Subclasses populate output port data
    in execute() (e.g. by reading a file)."""

    type_name = 'source.generic'

    def deserialize(self, data: dict) -> bool:
        # Generic SourceNode declares its outputs only via the JSON
        # outputPorts map — create matching ports if they don't already exist
        # so Node.deserialize can apply per-port state (including the
        # persistent/persistenceMode pair, which base deserialize sets).
        if 'outputPorts' in data:
            for name, entry in data['outputPorts'].items():
                if self.output_port(name) is None:
                    self.add_output(name, entry.get('type', 'ImageData'))
        return super().deserialize(data)


class TransformNode(Node):
    """A node with both input and output ports. Subclasses implement
    transform(inputs) → {output_name: PortData}; execute() pulls inputs
    from connected ports, calls transform, and stores results on outputs.

    Outputs added without explicit persistence arguments follow the
    pipeline-wide PipelineSettings.transform_persistence_default (the
    library default keeps data in memory; applications like tomviz set
    'disk' or 'transient' to bound RAM use). The default is applied only
    when the port is added — later settings changes affect new ports
    only, matching the C++ behavior."""

    def _default_output_persistence(self):
        default = PipelineSettings.instance().transform_persistence_default
        if default == TransformPersistenceDefault.OnDisk:
            return True, PersistenceMode.OnDisk
        if default == TransformPersistenceDefault.Transient:
            return False, PersistenceMode.InMemory
        return True, PersistenceMode.InMemory

    def apply_default_persistence(self, port: OutputPort):
        """Apply the pipeline-wide transform default to a port that was
        built directly (bypassing add_output). Mode is set before the
        persistent flag so a single reconcile runs with the final mode
        in place."""
        persistent, mode = self._default_output_persistence()
        port.persistence_mode = mode
        port.persistent = persistent

    def execute(self) -> bool:
        inputs: dict[str, PortData] = {}
        for port in self._input_ports:
            data = port.data()
            if data is None:
                # Missing input — treat as failure so downstream cascades
                # mark stale.
                return False
            inputs[port.name] = data

        result = self.transform(inputs)
        if result is None:
            return False

        for name, data in result.items():
            port = self.output_port(name)
            if port is not None:
                port.set_data(data)
        return True

    def transform(self, inputs: dict[str, PortData]) -> dict[str, PortData]:
        raise NotImplementedError(
            f'{type(self).__name__}.transform is not implemented')


class SinkNode(Node):
    """A node with only input ports — an endpoint that observes data
    (visualization, export, ...) rather than producing it.

    By default sinks are inert: executors skip them entirely and leave
    their state untouched, matching the headless tomviz runtime where
    sinks only exist so state files load cleanly. An application that
    wants sinks to act sets `executable = True` on its subclass and
    implements consume(inputs)."""

    executable: bool = False

    def execute(self) -> bool:
        inputs: dict[str, PortData] = {}
        for port in self._input_ports:
            data = port.data()
            if data is None:
                return False
            inputs[port.name] = data
        return self.consume(inputs)

    def consume(self, inputs: dict[str, PortData]) -> bool:
        return True
