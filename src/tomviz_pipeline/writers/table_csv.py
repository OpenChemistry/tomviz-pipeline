###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""CSV writer for Table port payloads. Column names form the header,
each subsequent row is the values across all columns. Accepts the
library's :class:`tomviz_pipeline.table.Table` natively, plus vtkTable
(duck-typed) for payloads produced by scripts that build vtk objects
directly."""

import csv
from pathlib import Path

from tomviz_pipeline.table import Table


def _column_value(column, row: int):
    """Read a single value from a vtkAbstractArray-like column, picking
    GetValue() for string arrays and GetTuple1() for numeric arrays."""
    # vtkStringArray exposes GetValue; vtkDataArray subclasses expose
    # GetTuple1. Try the numeric path first since it's the common case.
    get_tuple = getattr(column, 'GetTuple1', None)
    if get_tuple is not None:
        try:
            return get_tuple(row)
        except (TypeError, ValueError):
            pass
    get_value = getattr(column, 'GetValue', None)
    if get_value is not None:
        return get_value(row)
    raise TypeError(
        f'Unsupported vtkTable column type: {type(column).__name__}')


def _write_table(table: Table, target_path: Path) -> None:
    columns = list(table.columns.values())
    with open(target_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(table.column_names)
        for r in range(table.num_rows):
            # float() matches the doubles vtk's GetTuple1 used to
            # produce, keeping the CSV text identical.
            w.writerow([c[r] if isinstance(c, list) else float(c[r])
                        for c in columns])


def _write_vtk_table(payload, target_path: Path) -> None:
    columns = [payload.GetColumn(i)
               for i in range(payload.GetNumberOfColumns())]
    headers = [c.GetName() or f'column_{i}' for i, c in enumerate(columns)]
    rows = payload.GetNumberOfRows()

    with open(target_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in range(rows):
            w.writerow([_column_value(c, r) for c in columns])


def write_table_csv(payload, target_path: Path) -> None:
    """Write a Table (or vtkTable) to CSV. Raises TypeError if the
    payload isn't recognizably tabular."""
    if isinstance(payload, Table):
        _write_table(payload, target_path)
        return
    if (hasattr(payload, 'GetNumberOfColumns')
            and hasattr(payload, 'GetNumberOfRows')
            and hasattr(payload, 'GetColumn')):
        _write_vtk_table(payload, target_path)
        return
    raise TypeError(
        f'CSV writer expected a Table, got {type(payload).__name__}')
