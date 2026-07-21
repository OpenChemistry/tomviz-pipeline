###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""EMD and Data Exchange HDF5 readers/writers used by the pipeline CLI
and operator helpers."""

import collections
import copy
import os
from pathlib import Path

import h5py
import numpy as np

from tomviz_pipeline.dataset import Dataset

DIMS = ['dim1', 'dim2', 'dim3']
ANGLE_UNITS = [b'[deg]', b'[rad]']

Dim = collections.namedtuple('Dim', 'path values name units')


def _read_dataset(dataset, options=None):
    if options is None or 'subsampleSettings' not in options:
        return dataset[:]

    strides = options.get('subsampleSettings', {}).get('strides', [1] * 3)
    bnds = options.get('subsampleSettings', {}).get('volumeBounds', [-1] * 6)

    if len(bnds) != 6 or any(x < 0 for x in bnds):
        for i in range(3):
            bnds[i * 2] = 0
            bnds[i * 2 + 1] = dataset.shape[i]

    return dataset[bnds[0]:bnds[1]:strides[0],
                   bnds[2]:bnds[3]:strides[1],
                   bnds[4]:bnds[5]:strides[2]]


def _swap_dims(dims, i, j):
    tmp = Dim(dims[i].path, dims[j].values, dims[j].name, dims[j].units)
    dims[j] = Dim(dims[j].path, dims[i].values, dims[i].name, dims[i].units)
    dims[i] = tmp


def _read_emd(path, options=None):
    def bytes_to_str(x):
        if isinstance(x, np.bytes_):
            return x.decode('utf-8')
        return x

    with h5py.File(path, 'r') as f:
        tomography = f['data/tomography']

        dims = []
        for dim in DIMS:
            dims.append(Dim(dim,
                            tomography[dim][:],
                            bytes_to_str(tomography[dim].attrs['name']),
                            bytes_to_str(tomography[dim].attrs['units'])))

        data = tomography['data']
        name = data.attrs.get('name', 'ImageScalars')
        if isinstance(name, (np.ndarray, list, tuple)):
            name = name[0]
        if isinstance(name, (bytes, bytearray)):
            name = name.decode()

        arrays = [(name, _read_dataset(data, options))]

        tomviz_scalars = tomography.get('tomviz_scalars')
        if isinstance(tomviz_scalars, h5py.Group):
            def is_hard_link(name):
                link = tomviz_scalars.get(name, getlink=True)
                return isinstance(link, h5py.HardLink)

            keys = list(filter(is_hard_link, tomviz_scalars.keys()))
            channel_datasets = [
                (key, _read_dataset(tomviz_scalars[key], options))
                for key in keys
            ]
            arrays += channel_datasets

        tilt_axis = None
        if (
            dims[0].name in ('angles', b'angles') or
            dims[0].units in ANGLE_UNITS
        ):
            arrays = [(name, np.transpose(data, [2, 1, 0])) for (name, data)
                      in arrays]
            _swap_dims(dims, 0, -1)
            tilt_axis = 2

        # EMD is row major; VTK expects column major.
        arrays = [(name, np.asfortranarray(data)) for (name, data) in arrays]

        output = {
            'arrays': arrays,
            'dims': dims,
            'tilt_axis': tilt_axis,
            'metadata': {},
        }

        if dims is not None and dims[-1].name in ('angles', b'angles'):
            output['tilt_angles'] = dims[-1].values[:].astype(np.float64)

        if 'scan_ids' in tomography:
            output['scan_ids'] = tomography['scan_ids'][:].astype(np.int32)

        return output


def _get_arrays_for_writing(dataset):
    """Return ``(primary_name, primary_array, extra_arrays)``.

    Primary is the first scalar in insertion order — it goes into
    ``<group>/data`` and lands at idx 0 on read. Preserves the
    dataset's iteration order across the round-trip so
    ``apply_to_each_array`` merges land at the same indices in both
    executors. The active scalar is marked separately by the caller
    via ``active_scalar_name`` when it differs from the primary.
    """
    tilt_angles = dataset.tilt_angles
    tilt_axis = dataset.tilt_axis

    names = list(dataset.scalars_names)
    if not names:
        return None, None, {}
    primary_name = names[0]
    primary_array = dataset.scalars(primary_name)

    extra_arrays = {}
    for name in names[1:]:
        extra_arrays[name] = dataset.scalars(name)

    if tilt_angles is not None and tilt_axis == 2:
        primary_array = np.transpose(primary_array, [2, 1, 0])
        extra_arrays = {name: np.transpose(array, [2, 1, 0]) for (name, array)
                        in extra_arrays.items()}

    primary_array = np.ascontiguousarray(primary_array)
    extra_arrays = {name: np.ascontiguousarray(array) for (name, array)
                    in extra_arrays.items()}

    if primary_array.dtype == np.float16:
        primary_array = primary_array.astype(np.float32)
    for key, value in extra_arrays.items():
        if value.dtype == np.float16:
            extra_arrays[key] = value.astype(np.float32)

    return primary_name, primary_array, extra_arrays


def _get_dims_for_writing(dataset, data, default_dims=None):
    tilt_angles = dataset.tilt_angles
    tilt_axis = dataset.tilt_axis
    spacing = dataset.spacing

    if default_dims is None:
        dims = []
        names = ['x', 'y', 'z']
        for i, name in enumerate(names):
            values = np.array(range(data.shape[i]))
            dims.append(Dim('dim%d' % (i + 1), values, name, '[n_m]'))
    else:
        dims = copy.deepcopy(default_dims)

    if spacing is not None:
        for i, dim in enumerate(dims):
            end = data.shape[i] * spacing[i]
            res = np.arange(0, end, spacing[i])
            dims[i] = Dim(dim.path, res, dim.name, dim.units)

    if tilt_angles is not None:
        if tilt_axis == 2:
            _swap_dims(dims, 0, -1)
        units = dims[0].units if dims[0].units in ANGLE_UNITS else '[deg]'
        dims[0] = Dim(dims[0].path, tilt_angles, 'angles', units)
    else:
        if dims[0].name in ('angles', b'angles'):
            dims[0] = Dim(dims[0].path, dims[0].values, 'x', '[n_m]')

    return dims


def _write_emd_node_into(node_group, dataset, dims=None):
    """Write the per-EMD-node layout *directly* into ``node_group`` —
    the volume dataset, dim1/2/3, tomviz_scalars, and (optional)
    scan_ids all sit at ``<node_group>/...``. Mirrors C++
    ``EmdFormat::writeNode(writer, path, image)``.

    The standalone-file writer (:func:`_write_emd`) wraps this with
    the outer ``/data/tomography`` group structure; the tvh5 writer
    calls this directly on each port subgroup, where the EMD node
    *is* the port group (matching what
    ``Tvh5Format::populatePayloadData`` and
    ``EmdFormat::readNode(reader, path, ...)`` expect)."""
    primary_name, primary_array, extra_arrays = \
        _get_arrays_for_writing(dataset)

    node_group.attrs.create('emd_group_type', 1, dtype='uint32')
    data = node_group.create_dataset('data', data=primary_array)
    data.attrs['name'] = primary_name

    dims = _get_dims_for_writing(dataset, data, dims)

    for dim in dims:
        # The C++ EmdFormat reader reads dim arrays via
        # readData<float>, whose strict H5Tequal type check rejects
        # any storage type other than H5T_IEEE_F32LE. Numpy's default
        # for both range() (int64) and arange-with-float-step (float64)
        # would fail there — coerce to float32 unconditionally.
        d = node_group.create_dataset(
            dim.path,
            data=np.asarray(dim.values, dtype=np.float32))
        d.attrs['name'] = dim.name
        d.attrs['units'] = dim.units

    tomviz_scalars = node_group.create_group('tomviz_scalars')
    if extra_arrays:
        for (name, array) in extra_arrays.items():
            tomviz_scalars.create_dataset(name, data=array)

    # Soft-link primary into tomviz_scalars so the group lists every
    # scalar including the one stored inline at /data.
    tomviz_scalars[primary_name] = h5py.SoftLink(data.name)

    # The reader applies this after array layout to override the
    # default of "active = /data".
    if dataset.active_name and dataset.active_name != primary_name:
        node_group.attrs['active_scalar_name'] = dataset.active_name

    if dataset.scan_ids is not None:
        node_group.create_dataset(
            'scan_ids',
            data=np.asarray(dataset.scan_ids, dtype=np.int32))


def _write_emd(path, dataset, dims=None):
    """Write a standalone ``.emd`` file. Mirrors C++ ``EmdFormat::write``:
    sets version_major/minor on the file root, creates
    ``/data/tomography``, and writes the EMD-node layout there."""
    with h5py.File(path, 'w') as f:
        f.attrs.create('version_major', 0, dtype='uint32')
        f.attrs.create('version_minor', 2, dtype='uint32')
        data_group = f.create_group('data')
        tomography_group = data_group.create_group('tomography')
        _write_emd_node_into(tomography_group, dataset, dims)


def _read_data_exchange(path, options=None):
    path = Path(path)
    with h5py.File(path, 'r') as f:
        g = f['/exchange']

        dataset_names = ['data', 'data_dark', 'data_white', 'theta']
        datasets = {}
        for name in dataset_names:
            if name in g:
                if name == 'theta':
                    datasets[name] = g[name][:]
                    continue
                datasets[name] = _read_dataset(g[name], options)

        to_fortran = (options is None or
                      not options.get('keep_c_ordering', False))

        if to_fortran:
            datasets = {name: np.asfortranarray(data)
                        for (name, data) in datasets.items()}
            if 'theta' in datasets:
                swap_keys = ['data', 'data_dark', 'data_white']
                for key in swap_keys:
                    if key in datasets:
                        datasets[key] = np.transpose(datasets[key], [2, 1, 0])

        tilt_axis = None
        if 'theta' in datasets:
            tilt_axis = 2 if to_fortran else 0

        output = {
            'arrays': [(path.stem, datasets.get('data'))],
            'data_dark': datasets.get('data_dark'),
            'data_white': datasets.get('data_white'),
            'tilt_angles': datasets.get('theta'),
            'tilt_axis': tilt_axis,
            'metadata': {},
        }
        return output


def _is_data_exchange(path):
    exts = ['.h5', '.hdf5']
    if not any([path.suffix.lower() == x for x in exts]):
        return False
    with h5py.File(path, 'r') as f:
        return '/exchange/data' in f


def load_dataset(data_file_path, read_options=None):
    """Read an EMD or Data Exchange file into a tomviz Dataset."""
    data_file_path = Path(data_file_path)
    if _is_data_exchange(data_file_path):
        output = _read_data_exchange(data_file_path, read_options)
    else:
        output = _read_emd(data_file_path, read_options)

    arrays = output['arrays']
    dims = output.get('dims')
    metadata = output.get('metadata', {})

    (active_array, _) = arrays[0]
    arrays = {name: array for (name, array) in arrays}

    data = Dataset(arrays, active_array)
    data.file_name = os.path.abspath(data_file_path)
    data.metadata = metadata

    if 'data_dark' in output:
        data.dark = output['data_dark']
    if 'data_white' in output:
        data.white = output['data_white']
    if 'tilt_angles' in output:
        data.tilt_angles = output['tilt_angles']
    if 'tilt_axis' in output:
        data.tilt_axis = output['tilt_axis']
    if 'scan_ids' in output:
        data.scan_ids = output['scan_ids']
    if dims is not None:
        data.spacing = [float(d.values[1] - d.values[0]) for d in dims]

    data.dims = dims

    return data
