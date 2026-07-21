###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Per-node executors: decide *where/how a single node runs*, orthogonal
to the pipeline-level executors that walk the plan. Mirrors the C++
NodeExecutor / InternalNodeExecutor / NodeExecutorFactory.

The in-process InternalNodeExecutor is the implicit default. The
tomviz-flavored layer registers 'external' (subprocess under a chosen
Python environment); other applications can register their own."""

from __future__ import annotations

from typing import Callable, Optional

from .node import Node


class NodeExecutor:
    """Runs one node. execute() blocks until the node is done and returns
    success. cancel()/complete() forward cooperative stop requests to
    wherever the node is actually running (e.g. a subprocess)."""

    # Serialized into the node's schema-v2 `executor` block. An empty
    # type_name means "do not serialize" (the in-process default).
    type_name: str = ''

    def execute(self, node: Node) -> bool:
        raise NotImplementedError

    def cancel(self, node: Node):
        pass

    def complete(self, node: Node):
        pass

    def should_auto_execute(self, node: Node) -> bool:
        """Ask whether ``node`` wants a periodic execution now (the
        auto-execute poll). The hook runs in the same environment
        execute() would use, so a node whose script only imports in its
        external env still answers correctly. Blocks until done. The
        default evaluates in-process via Node.query_should_auto_execute().
        """
        return node.query_should_auto_execute()

    def serialize(self) -> dict:
        return {'type': self.type_name}

    def deserialize(self, data: dict) -> bool:
        return True


class InternalNodeExecutor(NodeExecutor):
    """Run the node in-process by calling node.execute()."""

    _instance: Optional['InternalNodeExecutor'] = None

    @classmethod
    def instance(cls) -> 'InternalNodeExecutor':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def execute(self, node: Node) -> bool:
        return node.execute()


class NodeExecutorFactory:
    """type string → NodeExecutor constructor registry, used to round-trip
    the per-node `executor` block in schema-v2 state files."""

    _creators: dict[str, Callable[[], NodeExecutor]] = {}

    @classmethod
    def register(cls, type_name: str, ctor: Callable[[], NodeExecutor]):
        cls._creators[type_name] = ctor

    @classmethod
    def create(cls, type_name: str) -> Optional[NodeExecutor]:
        ctor = cls._creators.get(type_name)
        if ctor is None:
            return None
        return ctor()

    @classmethod
    def known_types(cls) -> list[str]:
        return list(cls._creators.keys())
