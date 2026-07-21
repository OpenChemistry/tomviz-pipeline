###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Type-string → Node constructor registry. Mirrors the C++ NodeFactory in
naming so schema-v2 state files round-trip without renaming.

The core registry starts empty. The tomviz-flavored layer registers the
built-in tomviz node types via tomviz_pipeline.nodes.register_builtins();
applications embedding the engine register their own types."""

from typing import Callable, Optional

from .node import Node


class NodeFactory:
    _creators: dict[str, Callable[[], Node]] = {}

    @classmethod
    def register(cls, type_name: str, ctor: Callable[[], Node]):
        cls._creators[type_name] = ctor

    @classmethod
    def create(cls, type_name: str) -> Optional[Node]:
        ctor = cls._creators.get(type_name)
        if ctor is None:
            return None
        node = ctor()
        if node is not None and node.type_name != type_name:
            # Several type strings may share one class (e.g. the inert
            # sink placeholder); remember which one created this instance
            # so serialization round-trips the original type.
            node.type_name = type_name
        return node

    @classmethod
    def known_types(cls) -> list[str]:
        return list(cls._creators.keys())
