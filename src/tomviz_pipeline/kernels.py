###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""User-facing base classes for schema-v2 operator scripts.

A *kernel* is the compute payload hosted inside a graph node: a schema-v2
operator script defines exactly one subclass of :class:`SourceKernel` or
:class:`TransformKernel`. The runtime (``PythonSource`` /
``PythonTransform`` via ``PythonNodeBackend``) instantiates it fresh for
each execution, injects an ``_operator_wrapper`` for progress / cancel /
completion plumbing, and calls the user method:

  * ``SourceKernel.produce(self, **params) -> dict | None`` — for nodes
    with no inputs that emit output port data (file readers, synthetic
    generators).

  * ``TransformKernel.transform(self, inputs, **params) -> dict | None``
    — for nodes that consume an ``inputs`` dict and return an outputs
    dict.

Both methods return a dict mapping output port names to payloads, or
``None`` to signal that no output was produced (cancellation, error,
or any condition where the node legitimately has nothing to emit). A
``None`` return is treated by the framework as an execution failure;
downstream nodes are not advanced.

For ImageData / Volume / TiltSeries ports the payload is a
:class:`tomviz_pipeline.dataset.Dataset`-compatible object; for Table /
Molecule ports it's a :class:`tomviz_pipeline.table.Table` /
:class:`tomviz_pipeline.molecule.Molecule` (built with
``tomviz.utils.make_spreadsheet`` / ``make_molecule``).

Progress / cancel / completion follow the existing
``tomviz_pipeline.operators`` contract —
:class:`tomviz_pipeline.operators.Progress` is reused under the hood so
the multi-port routing of ``self.progress.data = X`` Just Works.

Kernels are deliberately NOT graph nodes: they carry no ports, links, or
pipeline state (that is :class:`tomviz_pipeline.core.Node` and friends).
Keeping this surface tiny is what lets scripts embedded in state files
stay valid for years and run unchanged inside the tomviz application and
in this standalone runtime.

The canonical import for new scripts is ``tomviz_pipeline.kernels``.
Older scripts import ``tomviz.nodes.SourceNode`` /
``tomviz.nodes.TransformNode``; the aliases installed by
``tomviz_pipeline._compat.install_script_module_aliases()`` resolve
those names to these same classes.
"""

from __future__ import annotations

from tomviz_pipeline.operators import Progress


class Kernel:
    """Base class for all schema-v2 kernels.

    Provides ``self.progress``, ``self.canceled``, ``self.completed``,
    and ``self.state``. Don't subclass directly — subclass
    :class:`SourceKernel` or :class:`TransformKernel` instead so the
    runtime can dispatch correctly.

    ``self.state`` is a dict the framework preserves across executions
    of the node (a fresh instance is created for every run, so plain
    instance attributes do not survive). It is shared by ``produce`` /
    ``transform`` and :meth:`should_auto_execute`, works under both
    internal and external execution, and is intentionally *not* saved
    into state files — every session starts with an empty dict. Keep
    its values JSON-serializable (bool/int/float/str, lists and
    string-keyed dicts thereof); anything else is dropped with a
    warning when the framework collects the dict after a run.

    :meth:`set_parameter` lets the kernel write back to the node's own
    parameters (the values it receives as keyword arguments) — e.g. a
    reader advancing a frame index, or an algorithm publishing the
    threshold it auto-detected so the user sees it in the parameter
    panel. Changes are collected after the user method returns and
    installed on the node; see the method for the exact semantics.
    """

    def __new__(cls, *args, **kwargs):
        """:meta private:"""
        obj = super().__new__(cls)
        obj.progress = Progress(obj)
        # Placeholder so __init__ can touch self.state; the runtime
        # replaces it with the node's persistent bag before calling
        # produce/transform/should_auto_execute.
        obj.state = {}
        # Parameter write-back plumbing. The runtime fills
        # _parameter_spec (name -> description entry) and
        # _parameter_values (current values) before calling the user
        # method and collects _parameter_updates afterwards. With no
        # spec installed (a host that predates the feature) updates
        # are accepted unvalidated and go nowhere.
        obj._parameter_spec = None
        obj._parameter_values = {}
        obj._parameter_updates = {}
        return obj

    @property
    def canceled(self) -> bool:
        """True when the user has requested cancellation. Long-running
        ``produce`` / ``transform`` implementations should poll this and
        bail out when set."""
        return self._operator_wrapper.canceled

    @property
    def completed(self) -> bool:
        """True when the user has requested early completion (e.g. an
        iterative algorithm should stop with its current best result).
        Iterative ``produce`` / ``transform`` implementations should
        poll this and return what they have."""
        return self._operator_wrapper.completed

    def should_auto_execute(self, **parameters) -> bool:
        """Decide whether a periodic execution should happen now.

        When the user enables "Periodic Execution" for a node (Execution
        tab), the application periodically calls this hook with the
        node's current parameter values — the same keyword names
        ``produce`` / ``transform`` receive. Return ``True`` to request
        that the node be marked stale and the pipeline re-executed.

        The hook should be cheap: it runs on a timer, and its job is
        deciding, not computing. Use ``self.state`` for the bookkeeping
        the decision needs (e.g. the modification time of the file that
        was last processed); mutations made here are preserved even
        when ``False`` is returned.

        The default implementation never requests a re-run.
        """
        return False

    def parameter(self, name: str, default=None):
        """Current value of the node parameter ``name`` as this kernel
        sees it: the value passed as a keyword argument, or the one most
        recently given to :meth:`set_parameter` during this call."""
        return self._parameter_values.get(name, default)

    def set_parameter(self, name: str, value):
        """Change the value of one of the node's parameters.

        ``name`` must be a parameter declared in the operator JSON
        description; anything else raises ``ValueError``. ``value`` is
        coerced to the declared type (``double`` → float, ``int`` →
        int, ``bool`` → bool, string-like types → str, ``enumeration``
        → one of the declared option values) and rejected with
        ``ValueError`` when it can't be.

        Updates are collected when the user method returns (even when
        it raised) and installed on the node. The run that made the
        change is deemed to have consumed the new value: nothing is
        marked stale and no re-execution is triggered — the node's
        next run simply receives the new value, and an application is
        notified through the node's ``parameters_updated`` signal so it
        can refresh its parameter UI. Inside
        :meth:`should_auto_execute` the return value alone decides
        whether a re-run happens; return ``True`` to have the node run
        with the values you just set.

        Updates cross the external-execution boundary the same way
        ``self.state`` does. They are not applied when the kernel runs
        inside a tomviz application build that predates the feature.
        """
        spec = self._parameter_spec
        if spec is not None:
            if name not in spec:
                declared = ', '.join(sorted(spec)) or '(none)'
                raise ValueError(
                    f"'{name}' is not a parameter of this node; the "
                    f'description declares: {declared}')
            value = _coerce_parameter_value(name, spec[name], value)
        self._parameter_updates[name] = value
        self._parameter_values[name] = value

    @staticmethod
    def create_dataset() -> 'Dataset':  # noqa: F821
        """Return a new empty :class:`tomviz_pipeline.dataset.Dataset`.

        Source kernels that need to emit a fresh dataset should call
        this and populate the result via the Dataset API
        (``set_scalars``, ``spacing``, ...). The script then runs
        unchanged across runtimes.
        """
        from tomviz_pipeline.dataset import Dataset
        return Dataset()


def _coerce_parameter_value(name: str, param: dict, value):
    """Coerce ``value`` to the type ``param`` (an entry of the
    description's ``parameters`` array) declares, raising
    ``ValueError`` when it doesn't fit. Values are always left
    JSON-serializable. Distinct from the backend's default coercion:
    an enumeration's *default* is an option index, while its runtime
    value is the option's value."""
    ptype = param.get('type', '')
    try:
        if ptype == 'enumeration':
            options = param.get('options') or []
            allowed = [next(iter(opt.values())) for opt in options
                       if isinstance(opt, dict) and opt]
            if value not in allowed:
                raise ValueError(
                    f'{value!r} is not one of the declared options '
                    f'{allowed}')
            return value
        if isinstance(value, (list, tuple)):
            return list(value)
        if ptype == 'double':
            return float(value)
        if ptype in ('int', 'integer'):
            return int(value)
        if ptype in ('bool', 'boolean'):
            return bool(value)
        if ptype in ('string', 'file', 'save_file', 'directory'):
            return str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"parameter '{name}' ({ptype}) cannot take {value!r}: "
            f'{exc}') from None
    return value


class SourceKernel(Kernel):
    """A kernel that produces output port data without consuming inputs.

    Override :meth:`produce` to compute the outputs.

    .. code-block:: python

        from tomviz_pipeline.kernels import SourceKernel

        class MySource(SourceKernel):
            def produce(self, radius=1.0):
                return {'volume': make_sphere(radius)}
    """

    def produce(self, **params) -> dict | None:
        """Compute and return this source's outputs.

        :param params: parameter values, named per the operator JSON
            description's ``parameters`` array.
        :returns: dict mapping output port names (declared in the JSON
            ``outputs`` array) to payload objects. Return ``None``
            (or simply ``return`` with no value) to signal cancellation
            or error — the framework treats it as no output produced.

        :meta private:
        """
        raise NotImplementedError(
            f'{type(self).__name__}.produce is not implemented')


class TransformKernel(Kernel):
    """A kernel that consumes inputs and produces outputs.

    Override :meth:`transform` to compute the outputs from the inputs.

    .. code-block:: python

        from tomviz_pipeline.kernels import TransformKernel

        class AddConstant(TransformKernel):
            def transform(self, inputs, constant=0.0):
                ds = inputs['volume']
                ds.active_scalars = ds.active_scalars + constant
                return {'volume': ds}
    """

    def transform(self, inputs: dict, **params) -> dict | None:
        """Compute and return this transform's outputs.

        :param inputs: dict mapping input port names (declared in the
            JSON ``inputs`` array) to payload objects.
        :param params: parameter values, named per the operator JSON
            description's ``parameters`` array.
        :returns: dict mapping output port names (declared in the JSON
            ``outputs`` array) to payload objects. Return ``None``
            (or simply ``return`` with no value) to signal cancellation
            or error — the framework treats it as no output produced.

        :meta private:
        """
        raise NotImplementedError(
            f'{type(self).__name__}.transform is not implemented')
