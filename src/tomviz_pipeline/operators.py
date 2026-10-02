###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""The v1 (legacy) kernel authoring API: Operator, CancelableOperator,
CompletableOperator. v1 scripts embedded in state files subclass these
(via the `tomviz.operators` alias installed by tomviz_pipeline._compat).
Progress lives in tomviz_pipeline.kernels and is re-exported here."""

from __future__ import annotations

from ._internal import AttributeAlias
from .dataset import Dataset
from .kernels import Progress


class Operator:
    """
    The base operator class from which all operators should be derived.

    Progress can be utilized and modified via the `self.progress` object.
    Details about the `self.progress` object interface can be see in the
    `tomviz_pipeline.kernels.Progress` class.
    """
    # Former name of the runtime's ``_execution_context``, which the tomviz
    # desktop application still sets.
    _operator_wrapper = AttributeAlias('_execution_context')

    def __new__(cls, *args, **kwargs):
        """
        :meta private:
        """
        obj = super(Operator, cls).__new__(cls)
        obj.progress = Progress(obj)

        return obj

    def transform_scalars(self, data):
        """
        This method should be overridden by subclasses to implement the
        operations the operator should perform.

        :meta private:
        """
        raise NotImplementedError('Must be implemented by subclass')

    def transform(self, dataset: Dataset, *args: tuple,
                  **kwargs: dict) -> dict | None:
        """
        Apply transformations to the dataset in order to obtain the
        desired output.

        Typically, the arrays on the dataset are modified in place,
        and no return is necessary. However, in situations like 3D
        reconstructions, a dictionary should be returned with
        a key such as "reconstruction" and a value of a child dataset
        containing the reconstruction.

        :meta private:
        """
        raise NotImplementedError('Must be implemented by subclass')


class CancelableOperator(Operator):
    """
    A cancelable operator allows the user to interrupt the execution of the
    operator. The canceled property can be using in the transform(...)
    method to break out when the operator is canceled.

    To utilize this class, the `transform()` function must be defined
    as a method within a class that inherits from `Progress` within the
    operator module. For example:

    .. code-block:: python

        import tomviz.operators

        class MyCancelableOperator(tomviz.operators.CancelableOperator):
            def transform(self, dataset, ...):
                while not self.canceled:
                    # Do work

    Note how `self.canceled` is checked in order to determine whether the
    operator was canceled or not. If it was canceled, the `transform()`
    function should abort.

    This operator may also report progress as well in `self.progress`.
    See the generic `Operator` class for details.
    """
    @property
    def canceled(self) -> bool:
        """
        :returns True if the operator has been canceled, False otherwise.
        """
        return self._execution_context.canceled


class CompletableOperator(CancelableOperator):
    """
    A completable operator allows a user to interrupt the execution of
    the operator using either "cancel" or "complete". The
    completable property can be used in the transform(...) method to break out
    when an operator is finished early, like if an iterative algorithm is a
    reasonable quality before the designated iterations are reached. Use
    similar to "cancel", but be sure to return data.

    To utilize this class, the `transform()` function must be defined
    as a method within a class that inherits from `CompletableOperator`
    within the operator module. For example:

    .. code-block:: python

        import tomviz.operators

        class MyCompletableOperator(tomviz.operators.CompletableOperator):
            def transform(self, dataset, ...):
                while True:
                    if self.completed:
                        # Break out of the iterative loop and return the
                        # current results.

    Note how `self.completed` is checked in order to determine whether the
    operator was completed or not. If completed, the transform function
    should skip to the end and return the current dataset.

    `self.canceled` may also be used in this operator, since it inherits
    from `CancelableOperator` as well.

    This operator may also report progress as well in `self.progress`.
    See the generic `Operator` class for details.
    """
    @property
    def completed(self) -> bool:
        """
        :returns True if the operator is early completed (from Button),
        False otherwise
        """
        return self._execution_context.completed
