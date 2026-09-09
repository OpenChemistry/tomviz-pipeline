###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Pipeline: the directed acyclic graph of Node objects connected by Link
objects, plus execution orchestration (planning, pause, cancel, futures).
Behavior mirrors the C++ Pipeline class in tomviz."""

from __future__ import annotations

from typing import Optional

from .events import Signal
from .future import ExecutionFuture
from .node import InputPort, Link, Node, NodeState, OutputPort


class Pipeline:
    """Directed graph of Node objects connected by Link objects.

    Signals:
      node_added(node)
      node_removed(node)
      link_created(link)
      link_removed(link)
      execution_started(future)
      execution_finished(future)
      paused_changed(bool)
      breakpoint_reached(node)
      link_validity_changed(link, valid)

    Effective port types (Node.recompute_effective_types) are propagated
    downstream whenever a link is created or removed and whenever an
    output's type changes at run time (a reader finding a tilt series).
    """

    def __init__(self):
        self.nodes: list[Node] = []
        self.links: list[Link] = []
        self._next_node_id: int = 1
        self._paused = False
        self._executor = None
        # When True, a node's set_parameters() triggers execute() —
        # the C++ pipeline's parametersApplied wiring. Off by default:
        # batch/CLI flows configure nodes without side effects.
        self.auto_execute = False
        self._param_connections: dict[int, object] = {}

        self.node_added = Signal('node_added')
        self.node_removed = Signal('node_removed')
        self.link_created = Signal('link_created')
        self.link_removed = Signal('link_removed')
        self.execution_started = Signal('execution_started')
        self.execution_finished = Signal('execution_finished')
        self.paused_changed = Signal('paused_changed')
        self.breakpoint_reached = Signal('breakpoint_reached')
        self.link_validity_changed = Signal('link_validity_changed')
        self._type_connections: dict[int, object] = {}

    # ---- graph mutation --------------------------------------------------

    def add_node(self, node: Node) -> Node:
        """Add a node to the graph and return it (handy for the
        `scale = pipeline.add_node(node_from_kernel(...))` idiom)."""
        if node.id < 0:
            node.id = self._next_node_id
            self._next_node_id += 1
        else:
            self._next_node_id = max(self._next_node_id, node.id + 1)
        self.nodes.append(node)
        self._param_connections[id(node)] = node.parameters_applied.connect(
            self._on_node_parameters_applied)
        self._type_connections[id(node)] = node.output_type_changed.connect(
            self._on_output_type_changed)
        self.node_added.emit(node)
        return node

    def remove_node(self, node: Node):
        """Remove a node and every link touching it. Downstream nodes are
        marked stale. Do not call while the pipeline is executing this
        node — cancel first (see clear())."""
        for port in node.input_ports():
            if port.link is not None:
                self.remove_link(port.link)
        for port in node.output_ports():
            for link in list(port.outgoing_links):
                self.remove_link(link)
        self.nodes.remove(node)
        for table in (self._param_connections, self._type_connections):
            connection = table.pop(id(node), None)
            if connection is not None:
                connection.disconnect()
        self.node_removed.emit(node)

    def clear(self):
        """Remove all nodes and links. If an execution is in flight it is
        canceled and joined first so no worker touches freed nodes."""
        if self._executor is not None:
            self._executor.cancel_and_wait()
        for node in list(self.nodes):
            self.remove_node(node)
        self._next_node_id = 1

    def set_node_id(self, node: Node, node_id: int):
        node.id = node_id
        self._next_node_id = max(self._next_node_id, node_id + 1)

    def node_by_id(self, node_id: int) -> Optional[Node]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def create_link(self, from_port: OutputPort,
                    to_port: InputPort) -> Link:
        """Connect an output port to an input port. An input port accepts
        a single link: an existing one is replaced. Raises ValueError if
        the link would create a cycle or the output port refuses the
        input (OutputPort.can_accept_link)."""
        if self.would_create_cycle(from_port, to_port):
            raise ValueError('Link would create a cycle')
        if not from_port.can_accept_link(to_port):
            raise ValueError(
                f"Port '{from_port.name}' does not accept a link to "
                f"'{to_port.name}'")
        if to_port.link is not None:
            self.remove_link(to_port.link)
        link = Link(from_port, to_port)
        from_port.outgoing_links.append(link)
        to_port.set_link(link)
        self.links.append(link)
        link._validity_connection = link.validity_changed.connect(
            lambda valid, link=link: self.link_validity_changed.emit(
                link, valid))
        if to_port.node is not None:
            to_port.node.mark_stale()
        self.link_created.emit(link)
        if to_port.node is not None:
            self._propagate_effective_types(to_port.node)
        return link

    def remove_link(self, link: Link):
        if link in link.from_port.outgoing_links:
            link.from_port.outgoing_links.remove(link)
        if link.to_port.link is link:
            link.to_port.set_link(None)
        if link in self.links:
            self.links.remove(link)
        connection = getattr(link, '_validity_connection', None)
        if connection is not None:
            connection.disconnect()
        if link.to_port.node is not None:
            link.to_port.node.mark_stale()
        self.link_removed.emit(link)
        if link.to_port.node is not None:
            self._propagate_effective_types(link.to_port.node)

    # ---- effective types -------------------------------------------------

    def _propagate_effective_types(self, start: Node):
        """Recompute the effective types of ``start`` and everything
        downstream (forward BFS), rechecking the links on the way. Mirrors
        the C++ Pipeline::propagateEffectiveTypes."""
        queue = [start]
        visited: set[int] = set()
        while queue:
            node = queue.pop(0)
            if id(node) in visited:
                continue
            visited.add(id(node))
            node.recompute_effective_types()
            for output in node.output_ports():
                for link in output.outgoing_links:
                    link.recheck()
                    downstream = link.to_port.node
                    if downstream is not None and id(downstream) not in visited:
                        queue.append(downstream)

    def _on_output_type_changed(self, node: Node, port: OutputPort, _type):
        # A type set from outside inference (a reader typing its output
        # when it runs): the consumers follow. recompute on `node` itself
        # is a no-op for a concrete declared type, which such a change is.
        if not any(n is node for n in self.nodes):
            return
        for link in list(port.outgoing_links):
            link.recheck()
            if link.to_port.node is not None:
                self._propagate_effective_types(link.to_port.node)

    # ---- topology --------------------------------------------------------

    def upstream_nodes(self, node: Node) -> list[Node]:
        return node.upstream_nodes()

    def downstream_nodes(self, node: Node) -> list[Node]:
        return node.downstream_nodes()

    def roots(self) -> list[Node]:
        return [n for n in self.nodes
                if not any(p.link is not None for p in n.input_ports())]

    def leaves(self) -> list[Node]:
        return [n for n in self.nodes
                if not any(p.outgoing_links for p in n.output_ports())]

    def would_create_cycle(self, from_port: OutputPort,
                           to_port: InputPort) -> bool:
        """True if linking from_port → to_port would create a cycle,
        i.e. from_port's node is reachable downstream of to_port's node
        (self-loops included)."""
        source = from_port.node
        target = to_port.node
        if source is None or target is None:
            return False
        if source is target:
            return True
        stack = [target]
        seen = set()
        while stack:
            n = stack.pop()
            if n is source:
                return True
            if id(n) in seen:
                continue
            seen.add(id(n))
            stack.extend(n.downstream_nodes())
        return False

    def is_valid(self) -> bool:
        """True if the graph is acyclic."""
        try:
            self.execution_order()
        except RuntimeError:
            return False
        return True

    def execution_order(self) -> list[Node]:
        """Kahn topological sort of the whole graph. Stable: ties broken
        by node id, then by the order nodes were added. Raises
        RuntimeError if the graph contains a cycle."""
        return self._topo_sort(self.nodes)

    def _topo_sort(self, nodes: list[Node]) -> list[Node]:
        """Kahn topological sort of `nodes`, considering only edges whose
        endpoints are both in `nodes`."""
        members = {id(n) for n in nodes}
        in_degree: dict[int, int] = {id(n): 0 for n in nodes}
        for link in self.links:
            fn = link.from_port.node
            tn = link.to_port.node
            if (fn is not None and tn is not None
                    and id(fn) in members and id(tn) in members):
                in_degree[id(tn)] += 1

        def sort_key(n: Node):
            return n.id if n.id >= 0 else 1 << 30

        ready = [n for n in nodes if in_degree[id(n)] == 0]
        ready.sort(key=sort_key)

        order: list[Node] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for d in n.downstream_nodes():
                if id(d) not in members:
                    continue
                in_degree[id(d)] -= 1
                if in_degree[id(d)] == 0:
                    ready.append(d)
            ready.sort(key=sort_key)

        if len(order) != len(nodes):
            raise RuntimeError('Pipeline contains a cycle')
        return order

    # ---- execution planning ----------------------------------------------

    def execution_plan(self, targets: list[Node] = None) -> list[Node]:
        """Build the smallest topo-sorted list of nodes to run so that
        every target is up to date. With no targets, every leaf is a
        target (full pipeline run).

        A target enters the plan when it is not Current. From each node in
        the plan the walk continues backwards over input links: the
        upstream node is included when it is not Current OR when the
        specific output the consumer reads has been evicted (transient
        data released after a previous plan). Current nodes with live
        outputs are pruned, which is what makes re-execution incremental.
        """
        if targets is None:
            targets = self.leaves()
        plan_ids: set[int] = set()
        plan_nodes: list[Node] = []
        stack: list[Node] = []

        def include(node: Node):
            if id(node) not in plan_ids:
                plan_ids.add(id(node))
                plan_nodes.append(node)
                stack.append(node)

        for t in targets:
            if t.state != NodeState.Current:
                include(t)

        while stack:
            consumer = stack.pop()
            self._include_upstream(consumer, include)

        return self._topo_sort(plan_nodes)

    def upstream_execution_plan(self, target: Node) -> list[Node]:
        """Plan that makes `target`'s inputs current without running the
        target itself."""
        plan_ids: set[int] = set()
        plan_nodes: list[Node] = []
        stack: list[Node] = []

        def include(node: Node):
            if id(node) not in plan_ids:
                plan_ids.add(id(node))
                plan_nodes.append(node)
                stack.append(node)

        self._include_upstream(target, include)
        while stack:
            consumer = stack.pop()
            self._include_upstream(consumer, include)

        return self._topo_sort(plan_nodes)

    @staticmethod
    def _include_upstream(consumer: Node, include):
        for port in consumer.input_ports():
            link = port.link
            if link is None:
                continue
            upstream = link.from_port.node
            if upstream is None:
                continue
            if (upstream.state != NodeState.Current
                    or not link.from_port.has_data()):
                include(upstream)

    # ---- execution -------------------------------------------------------

    @property
    def executor(self):
        """The pipeline-level executor. Lazily a DefaultExecutor."""
        if self._executor is None:
            from .executor import DefaultExecutor
            self.set_executor(DefaultExecutor(self))
        return self._executor

    def set_executor(self, executor):
        if self._executor is not None:
            self._executor.breakpoint_reached.disconnect(
                self._on_breakpoint_reached)
        self._executor = executor
        if executor is not None:
            executor.pipeline = self
            executor.breakpoint_reached.connect(self._on_breakpoint_reached)

    def _on_breakpoint_reached(self, node: Node):
        self.breakpoint_reached.emit(node)

    def _on_node_parameters_applied(self, node: Node, params: dict):
        # set_parameters already marked the node stale; with
        # auto_execute on, bring the pipeline current again (C++ wires
        # TransformNode::parametersApplied to execute the same way).
        if self.auto_execute:
            self.execute()

    def is_executing(self) -> bool:
        return self._executor is not None and self._executor.is_running()

    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool):
        paused = bool(paused)
        if paused == self._paused:
            return
        self._paused = paused
        self.paused_changed.emit(paused)
        if not paused and any(n.state != NodeState.Current
                              for n in self.nodes):
            self.execute()

    def execute(self, target: Node = None) -> ExecutionFuture:
        """Execute the whole pipeline (every leaf a target) or just the
        subgraph needed to bring `target` up to date. Returns an
        ExecutionFuture; with the blocking DefaultExecutor it is already
        finished when this returns."""
        targets = None if target is None else [target]
        return self._run_plan(self.execution_plan(targets))

    def execute_upstream_of(self, target: Node) -> ExecutionFuture:
        """Bring target's inputs up to date without running target."""
        return self._run_plan(self.upstream_execution_plan(target))

    def execute_when_idle(self):
        """Run the pipeline once the in-flight execution (if any) ends,
        without canceling it. Used e.g. when adding a sink: upstream
        results are not invalidated, so don't interrupt them."""
        if not self.is_executing():
            self.execute()
            return

        conn = None

        def on_finished(_future):
            conn.disconnect()
            self.execute()

        conn = self.execution_finished.connect(on_finished)

    def cancel_execution(self):
        if self._executor is not None:
            self._executor.cancel()

    def _run_plan(self, plan: list[Node]) -> ExecutionFuture:
        if self._paused:
            # Parity with the C++ pipeline: return an unstarted future.
            # Un-pausing re-executes and produces a fresh future.
            return ExecutionFuture()

        if not plan:
            # Nothing to do: finish immediately. Note execution_finished
            # fires without a preceding execution_started, as in C++.
            future = ExecutionFuture()
            future._finish(True)
            self.execution_finished.emit(future)
            return future

        # Wire the future before submitting: with the blocking
        # DefaultExecutor the run completes inside submit().
        future = ExecutionFuture()
        future.finished.connect(
            lambda f: self.execution_finished.emit(f))
        self.execution_started.emit(future)
        self.executor.submit(plan, future)
        return future
