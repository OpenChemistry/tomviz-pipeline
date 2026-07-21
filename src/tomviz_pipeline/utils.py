###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Helpers used by operator scripts (reachable as ``tomviz.utils`` via
tomviz_pipeline._compat) and by the automatic ``apply_to_each_array``
transform decoration. Pure numpy — ``make_spreadsheet`` and
``make_molecule`` build the library's own :class:`~tomviz_pipeline.
table.Table` / :class:`~tomviz_pipeline.molecule.Molecule` payloads,
so no vtk is required."""

from __future__ import annotations

import copy
import functools
import math

import numpy as np

from tomviz_pipeline.dataset import Dataset


def zoom_shape(input: np.ndarray, zoom: np.ndarray) -> tuple[int]:
    """
    Returns the shape of the output array for scipy.ndimage.zoom
    """

    if isinstance(zoom, (int, float,)):
        zoom = [zoom] * input.ndim

    return tuple(
        [int(round(i * j)) for i, j in zip(input.shape, zoom)])


def _minmax(coor, minc, maxc):
    if coor[0] < minc[0]:
        minc[0] = coor[0]
    if coor[0] > maxc[0]:
        maxc[0] = coor[0]
    if coor[1] < minc[1]:
        minc[1] = coor[1]
    if coor[1] > maxc[1]:
        maxc[1] = coor[1]

    return minc, maxc


def rotate_shape(input: np.ndarray, angle: float,
                 axes: tuple[int, int]) -> tuple[int]:
    """
    Returns the shape of the output array of scipy.ndimage.rotate
    derived from: https://github.com/scipy/scipy/blob/v0.16.1/scipy/ndimage/ \
    interpolation.py #L578. We are duplicating the code here so we can generate
    an array of the right shape and array order to pass into the rotate
    function.
    """

    axes = list(axes)
    rank = input.ndim
    if axes[0] < 0:
        axes[0] += rank
    if axes[1] < 0:
        axes[1] += rank
    if axes[0] < 0 or axes[1] < 0 or axes[0] > rank or axes[1] > rank:
        raise RuntimeError('invalid rotation plane specified')
    if axes[0] > axes[1]:
        axes = axes[1], axes[0]
    angle = np.pi / 180 * angle
    m11 = math.cos(angle)
    m12 = math.sin(angle)
    m21 = -math.sin(angle)
    m22 = math.cos(angle)
    matrix = np.array([[m11, m12],
                       [m21, m22]], dtype=np.float64)
    iy = input.shape[axes[0]]
    ix = input.shape[axes[1]]
    mtrx = np.array([[m11, -m21],
                     [-m12, m22]], dtype=np.float64)
    minc = [0, 0]
    maxc = [0, 0]
    coor = np.dot(mtrx, [0, ix])
    minc, maxc = _minmax(coor, minc, maxc)
    coor = np.dot(mtrx, [iy, 0])
    minc, maxc = _minmax(coor, minc, maxc)
    coor = np.dot(mtrx, [iy, ix])
    minc, maxc = _minmax(coor, minc, maxc)
    oy = int(maxc[0] - minc[0] + 0.5)
    ox = int(maxc[1] - minc[1] + 0.5)
    offset = np.zeros((2,), dtype=np.float64)
    offset[0] = float(oy) / 2.0 - 0.5
    offset[1] = float(ox) / 2.0 - 0.5
    offset = np.dot(matrix, offset)
    tmp = np.zeros((2,), dtype=np.float64)
    tmp[0] = float(iy) / 2.0 - 0.5
    tmp[1] = float(ix) / 2.0 - 0.5
    offset = tmp - offset
    output_shape = list(input.shape)
    output_shape[axes[0]] = oy
    output_shape[axes[1]] = ox

    return output_shape


def apply_to_each_array(func):
    """
    This decorator causes an operator `transform()` function to
    automatically run one time for every array.

    For example, for the rotation operator:

    .. code-block:: python

        @apply_to_each_array
        def transform(dataset, rotation_angle=90.0, rotation_axis=0):
            # ...

    The `transform()` function will be executed one time for every
    array. When executing the `transform()` function, the `dataset`
    object will only contain a single array on `dataset.active_scalars`
    each time.

    This allows an operator `transform()` function to be written in
    a way that appears to only operate on one array, but then automatically
    be ran multiple times to apply to each array.

    The final `dataset` object will automatically contain each of the
    transformed arrays on it.
    """
    is_method = (
        func.__name__ != func.__qualname__ and
        '.<locals>.' not in func.__qualname__
    )
    dataset_idx = 1 if is_method else 0

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        dataset = args[dataset_idx]
        if dataset.num_scalars == 1:
            # Just run the function like we normally would...
            return func(*args, **kwargs)

        num_arrays = dataset.num_scalars
        array_names = dataset.scalars_names
        active_name = dataset.active_name

        # Run the function multiple times. Each time with a single,
        # different array on the dataset.
        all_arrays = [dataset.arrays[name] for name in array_names]
        dataset.arrays.clear()
        orig_dataset = dataset

        output_arrays = []
        results = []
        for i, name in enumerate(array_names):
            if i == num_arrays - 1:
                # Use the original dataset for the final one
                dataset = orig_dataset
            else:
                dataset = copy.deepcopy(orig_dataset)

            dataset.arrays[name] = all_arrays[i]
            dataset.active_name = name

            # Put the dataset where it belongs in the argument list
            new_args = (
                list(args[:dataset_idx]) +
                [dataset] +
                list(args[dataset_idx + 1:])
            )
            print('Transforming array:', name)
            result = func(*new_args, **kwargs)
            results.append(result)

            output_arrays.append(dataset.arrays[name])

        # The metadata should have been modified on this dataset
        # object from the last call to the function
        dataset.arrays.clear()
        for name, array in zip(array_names, output_arrays):
            dataset.arrays[name] = array

        dataset.active_name = active_name

        # For any data sources in the result, add all the scalars to it
        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, Dataset):
                    # Rename the active array
                    v.rename_active(array_names[-1])
                    # Go back through the other results and set scalars on
                    # this one
                    for i, other_result in enumerate(results[:-1]):
                        if (
                            isinstance(other_result, dict) and
                            isinstance(other_result.get(k), Dataset)
                        ):
                            other_dataset = other_result[k]
                            other_dataset.rename_active(array_names[i])
                            v.set_scalars(other_dataset.active_name,
                                          other_dataset.active_scalars)
                    # Restore the original active so the merged output
                    # doesn't end up on whatever was processed last.
                    if active_name in v.scalars_names:
                        v.active_name = active_name

        return result

    return wrapper


def pad_array(array: np.ndarray, padding: int, tilt_axis: int) -> np.ndarray:
    """Add padding to an array. Ignore the tilt axis.

    The resulting padded array can eventually be depadded by calling the
    `depad_array()` function.
    """
    if padding <= 0:
        return array

    pad_list = []
    for i in range(3):
        pad_list.append([0, 0] if i == tilt_axis else [padding, padding])

    return np.pad(array, pad_list)


def depad_array(array: np.ndarray, padding: int,
                tilt_axis: int) -> np.ndarray:
    """Remove padding from an array. Ignore the tilt axis."""
    if padding <= 0:
        return array

    slice_list = []
    for i in range(3):
        start = padding if i != tilt_axis else 0
        end = padding * -1 if i != tilt_axis else array.shape[i]
        slice_list.append(slice(start, end))
    return array[tuple(slice_list)]


def make_spreadsheet(column_names: list[str], table: np.ndarray,
                     axes_labels: tuple[str, str] = None,
                     axes_log_scale: tuple[bool, bool] = None,
                     ) -> 'Table':  # noqa: F821
    """Make a spreadsheet object to use within Tomviz

    If returned from an operator, this will ultimately appear within the
    pipeline, and will be save-able to a JSON file.

    The output of this function ought to be included in the returned
    dictionary. For example:

    .. code-block:: python

        spreadsheet = utils.make_spreadsheet(column_names, table)

        return {
            'table_data': spreadsheet,
        }
    """
    # column_names is a list of strings
    # table is a 2D numpy.ndarray
    # returns a Table object that stores the table content
    from tomviz_pipeline.table import Table

    table = np.asarray(table)

    if (table.shape[1] != len(column_names)):
        print('Warning: table number of columns differs from number of '
              'column names')
        return

    result = Table()
    for (column, name) in enumerate(column_names):
        # float32 columns, matching the vtkFloatArray the in-app
        # implementation builds.
        result.set_column(name, table[:, column].astype(np.float32))

    if axes_labels is not None:
        result.axes_labels = (str(axes_labels[0]), str(axes_labels[1]))

    if axes_log_scale is not None:
        result.axes_log_scale = (bool(axes_log_scale[0]),
                                 bool(axes_log_scale[1]))

    return result


def make_molecule(atomic_numbers, positions, bonds=None,
                  bond_orders=None) -> 'Molecule':  # noqa: F821
    """Make a molecule object to use within Tomviz.

    :param atomic_numbers: one atomic number per atom.
    :param positions: atom coordinates, shape (num_atoms, 3) — a flat
        length-3N sequence is also accepted.
    :param bonds: optional (num_bonds, 2) pairs of atom indices — a
        flat length-2M sequence is also accepted.
    :param bond_orders: optional per-bond order; defaults to 1.

    Like :func:`make_spreadsheet`, the returned object ought to be
    included in the dictionary returned by an operator, under the key
    of a ``Molecule`` output port.
    """
    from tomviz_pipeline.molecule import Molecule
    return Molecule(atomic_numbers, positions, bonds, bond_orders)
