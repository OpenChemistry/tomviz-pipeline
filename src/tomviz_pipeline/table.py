###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Pure-Python payload for ``Table`` output ports.

Mirrors the role vtkTable plays in the C++ application: an ordered set
of named columns (numeric numpy arrays or lists of strings) plus the
chart metadata tomviz attaches (axes labels, log-scale flags) — carried
as plain attributes, the same pattern :class:`tomviz_pipeline.dataset.
Dataset` uses for tilt angles. Applications convert to their native
table type (e.g. vtkTable) at the visualization boundary."""

from __future__ import annotations

from typing import Optional

import numpy as np


class Table:
    """Tabular data flowing through a ``Table`` port.

    ``columns`` maps column name -> column values, preserving insertion
    order. Numeric columns are 1-D numpy arrays; string columns are
    lists of ``str``. All columns must have the same length."""

    def __init__(self, columns=None,
                 axes_labels: Optional[tuple] = None,
                 axes_log_scale: Optional[tuple] = None):
        self.columns: dict = {}
        if columns:
            for name, values in dict(columns).items():
                self.set_column(name, values)
        # (x_label, y_label) hints for chart viewers, or None.
        self.axes_labels = axes_labels
        # (x_is_log, y_is_log) hints for chart viewers, or None.
        self.axes_log_scale = axes_log_scale

    def set_column(self, name: str, values) -> None:
        """Add or replace a column. Numeric input is coerced to a 1-D
        numpy array; string input is stored as a list of str."""
        if _is_string_sequence(values):
            column = [str(v) for v in values]
        else:
            column = np.asarray(values)
            if column.ndim != 1:
                raise ValueError(
                    f'Table column {name!r} must be 1-D, '
                    f'got shape {column.shape}')
        if self.columns:
            expected = self.num_rows
            if len(column) != expected:
                raise ValueError(
                    f'Table column {name!r} has {len(column)} rows, '
                    f'expected {expected}')
        self.columns[name] = column

    def column(self, name: str):
        return self.columns[name]

    @property
    def column_names(self) -> list:
        return list(self.columns.keys())

    @property
    def num_columns(self) -> int:
        return len(self.columns)

    @property
    def num_rows(self) -> int:
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    def __repr__(self):
        return (f'Table(columns={self.column_names!r}, '
                f'rows={self.num_rows})')


def _is_string_sequence(values) -> bool:
    if isinstance(values, np.ndarray):
        return values.dtype.kind in ('U', 'S', 'O')
    try:
        return all(isinstance(v, (str, bytes, np.str_)) for v in values) \
            and len(values) > 0
    except TypeError:
        return False
