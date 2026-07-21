###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Internal helpers for the operator runtimes: locate the transform
function / operator class inside a loaded operator module, apply the
automatic transform decorators, and back cancel / complete polling via
OperatorWrapper."""

from __future__ import annotations

from types import MethodType
from typing import Any, Callable
import inspect
import json


class OperatorWrapper(object):
    """Backs `self.canceled` / `self.completed` on operators running
    under the pure-Python pipeline runtime. The flags can be flipped
    out-of-band by the parent process via a transport-specific
    ControlChannel, which is polled lazily on every getter access —
    no reader thread needed on the subprocess side.

    When a graph `node` is supplied, the getters also reflect its
    in-process cancel / complete events, so ThreadedExecutor.cancel()
    reaches operators that poll these flags even when no parent-process
    control channel exists."""

    def __init__(self, control_channel=None, node=None):
        self._channel = control_channel
        self._node = node
        self._canceled = False
        self._completed = False

    @property
    def canceled(self) -> bool:
        if self._channel is not None:
            self._channel.poll(self)
        if self._canceled:
            return True
        return (self._node is not None
                and self._node.is_cancel_requested())

    @property
    def completed(self) -> bool:
        if self._channel is not None:
            self._channel.poll(self)
        if self._completed:
            return True
        return (self._node is not None
                and self._node.is_complete_requested())


def find_operator_class(transform_module):
    from tomviz_pipeline._compat import operator_base_classes
    base_classes = operator_base_classes()
    operator_class = None
    classes = inspect.getmembers(transform_module, inspect.isclass)
    for (name, cls) in classes:
        if issubclass(cls, base_classes):
            if operator_class is not None:
                raise Exception('Multiple operators define in module, only '
                                'one operator can be defined per module.')

            operator_class = cls

    return operator_class


def _find_function(module, function_name):
    # Finds a function in the module with a given "function_name"
    # Returns `None` if it is not found
    functions = inspect.getmembers(module, inspect.isfunction)
    for (name, func) in functions:
        if name == function_name:
            return func


def find_transform_from_module(transform_module):
    # This tries to first find transform(), and then transform_scalars()
    f = _find_function(transform_module, 'transform')
    if f is None:
        f = _find_function(transform_module, 'transform_scalars')

    return f


def find_transform_function(transform_module, op=None):
    # op is accepted for ABI compat with legacy OperatorPython but ignored.
    del op

    transform_function = find_transform_from_module(transform_module)
    if transform_function is None:
        cls = find_operator_class(transform_module)
        if cls is None:
            raise Exception('Unable to locate transform function.')

        o = cls.__new__(cls)
        # _operator_wrapper is read by CompletableOperator/CancelableOperator
        # __init__ and during transform(); install the pure-Python fallback.
        o._operator_wrapper = OperatorWrapper(None)
        cls.__init__(o)

        transform_function = None
        if _operator_method_was_implemented(o, 'transform'):
            transform_function = o.transform
        elif _operator_method_was_implemented(o, 'transform_scalars'):
            transform_function = o.transform_scalars

    if transform_function is None:
        raise Exception('Unable to locate transform function.')

    return transform_function


def has_decorator(func: Callable,
                  decorator_marker: str = '_is_my_decorator') -> bool:
    """Check if a function was already decorated with a decorator name"""
    # Check the function itself
    if getattr(func, decorator_marker, False):
        return True

    # Traverse the __wrapped__ chain
    current = func
    while hasattr(current, '__wrapped__'):
        current = current.__wrapped__
        if getattr(current, decorator_marker, False):
            return True

    return False


def apply_decorator(func: Callable, decorator: Callable) -> Callable:
    # Apply the decorator, taking into account different behavior for
    # MethodType callables
    if isinstance(func, MethodType):
        # It's a bound method
        tmp_func = decorator(func.__func__)
        return MethodType(tmp_func, func.__self__)

    # Unbound function
    return decorator(func)


def add_transform_decorators(transform_method: Callable,
                             operator_dict: dict[str, Any]) -> Callable:
    """Optionally add any transform wrappers that we need to add

    Currently, this adds `@apply_to_each_array` automatically if
    `"apply_to_each_array": false` is not set within the json
    description, and if the decorator was not already applied.
    """
    add_apply_to_each_array = True
    operator_description = operator_dict.get('description')
    if operator_description:
        description_json = json.loads(operator_description)
        if not description_json.get('apply_to_each_array', True):
            # It was intentionally disabled in the json
            add_apply_to_each_array = False

    if transform_method.__name__ == 'transform_scalars':
        # This is an old transform function. We don't want to do any
        # kind of automatic modifications to the old ones.
        add_apply_to_each_array = False

    if add_apply_to_each_array:
        # First, make sure it wasn't already decorated
        if not has_decorator(transform_method, 'apply_to_each_array'):
            # Decorate it!
            from tomviz_pipeline.utils import apply_to_each_array
            transform_method = apply_decorator(transform_method,
                                               apply_to_each_array)

    return transform_method


def _operator_method_was_implemented(obj, method):
    # It would be nice if there were an easier way to do this, but
    # I am not currently aware of an easier way.
    # Exclude the operator base-class layer itself — from either import
    # package (scripts may subclass a real tomviz install's classes).
    base_modules = ('tomviz_pipeline.operators', 'tomviz.operators')
    bases = [b for b in inspect.getmro(type(obj))
             if b.__module__ not in base_modules]

    for base in bases:
        if method in vars(base):
            return True

    return False
