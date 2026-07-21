###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Writers for tomviz state containers — the ``.tvh5`` format that
bundles a schema-v2 state JSON with per-port voxel data inside one
HDF5 file. Mirrors the C++ ``Tvh5Format::write`` so files produced
here can be loaded by the in-app pipeline as well as round-tripped
by :func:`tomviz_pipeline.state.load_state`."""

import copy
import json
import logging
from pathlib import Path

import h5py
import numpy as np

from tomviz_pipeline.core import SinkNode
from tomviz_pipeline.io.emd import _write_emd_node_into
from tomviz_pipeline.molecule import Molecule
from tomviz_pipeline.table import Table


logger = logging.getLogger('tomviz_pipeline')


# Mirrors VTK's vtkType.h. Used as the ``vtkDataType`` attribute on each
# column dataset so the reader can rebuild the matching vtkAbstractArray
# subclass.
VTK_STRING = 13


# numpy dtype kind+size -> VTK data type code, for annotating columns
# written from pure-Python Tables. The codes are VTK's stable ABI
# constants (vtkType.h) — using them here does not require vtk.
_VTK_TYPE_FOR_DTYPE = {
    ('i', 1): 15,  # VTK_SIGNED_CHAR
    ('u', 1): 3,   # VTK_UNSIGNED_CHAR
    ('i', 2): 4,   # VTK_SHORT
    ('u', 2): 5,   # VTK_UNSIGNED_SHORT
    ('i', 4): 6,   # VTK_INT
    ('u', 4): 7,   # VTK_UNSIGNED_INT
    ('i', 8): 16,  # VTK_LONG_LONG
    ('u', 8): 17,  # VTK_UNSIGNED_LONG_LONG
    ('f', 4): 10,  # VTK_FLOAT
    ('f', 8): 11,  # VTK_DOUBLE
}


def _is_table(payload) -> bool:
    """True for the library's Table, or (duck-typed) a vtkTable
    produced by a script that builds vtk objects directly. The vtk
    check is attribute-based so vtk is never imported speculatively."""
    return isinstance(payload, Table) or (
        hasattr(payload, 'GetNumberOfColumns')
        and hasattr(payload, 'GetNumberOfRows')
        and hasattr(payload, 'GetColumn'))


def _is_molecule(payload) -> bool:
    """True for the library's Molecule or (duck-typed) a vtkMolecule.
    Same rationale as :func:`_is_table`."""
    return isinstance(payload, Molecule) or (
        hasattr(payload, 'GetNumberOfAtoms')
        and hasattr(payload, 'GetNumberOfBonds')
        and hasattr(payload, 'GetAtomicNumberArray'))


def _write_table_into(group: 'h5py.Group', table) -> None:
    """Serialize a Table (or vtkTable) under ``group`` using the same
    layout as C++ ``Tvh5Format::writeTablePayload``: each column becomes
    a sub-dataset ``c0``, ``c1``, … with attributes ``name``,
    ``vtkDataType`` and ``numberOfComponents``. Numeric columns are
    written as their native numpy dtype; string columns become a
    JSON-encoded int8 blob (matching the C++ writer, which can't use
    HDF5 variable-length strings through h5cpp's writeData<>)."""
    if isinstance(table, Table):
        _write_pure_table_into(group, table)
        return

    from vtk.util.numpy_support import vtk_to_numpy

    num_columns = int(table.GetNumberOfColumns())
    num_rows = int(table.GetNumberOfRows())
    group.attrs['type'] = 'table'
    group.attrs['numColumns'] = np.int64(num_columns)
    group.attrs['numRows'] = np.int64(num_rows)

    for i in range(num_columns):
        column = table.GetColumn(i)
        if column is None:
            continue
        name = column.GetName() or ''
        dataset_name = f'c{i}'
        if hasattr(column, 'GetValue') and not hasattr(column, 'GetTuple1'):
            # vtkStringArray: serialize as JSON.
            values = [column.GetValue(j)
                      for j in range(column.GetNumberOfValues())]
            blob = json.dumps(values).encode('utf-8')
            ds = group.create_dataset(
                dataset_name, data=np.frombuffer(blob, dtype='i1'))
            vtk_data_type = VTK_STRING
            number_of_components = 1
        else:
            # Any vtkDataArray subclass.
            np_array = vtk_to_numpy(column)
            ds = group.create_dataset(dataset_name, data=np_array)
            vtk_data_type = int(column.GetDataType())
            number_of_components = int(column.GetNumberOfComponents())
        ds.attrs['name'] = name
        ds.attrs['vtkDataType'] = np.int32(vtk_data_type)
        ds.attrs['numberOfComponents'] = np.int32(number_of_components)


def _write_pure_table_into(group: 'h5py.Group', table: Table) -> None:
    """The :class:`tomviz_pipeline.table.Table` flavor of
    :func:`_write_table_into` — identical group layout, written straight
    from the numpy columns. The chart-axis attributes are additive
    extras (absent from vtk-backed writes; ignored by the C++ reader)."""
    group.attrs['type'] = 'table'
    group.attrs['numColumns'] = np.int64(table.num_columns)
    group.attrs['numRows'] = np.int64(table.num_rows)

    for i, (name, column) in enumerate(table.columns.items()):
        dataset_name = f'c{i}'
        if isinstance(column, list):
            # String column: JSON-encoded int8 blob, as the C++ writer
            # and the vtk branch above produce.
            blob = json.dumps([str(v) for v in column]).encode('utf-8')
            ds = group.create_dataset(
                dataset_name, data=np.frombuffer(blob, dtype='i1'))
            vtk_data_type = VTK_STRING
        else:
            ds = group.create_dataset(dataset_name, data=column)
            key = (column.dtype.kind, column.dtype.itemsize)
            vtk_data_type = _VTK_TYPE_FOR_DTYPE.get(key, 11)
        ds.attrs['name'] = name
        ds.attrs['vtkDataType'] = np.int32(vtk_data_type)
        ds.attrs['numberOfComponents'] = np.int32(1)

    if table.axes_labels is not None:
        group.attrs['axesLabels'] = json.dumps(
            [str(v) for v in table.axes_labels])
    if table.axes_log_scale is not None:
        group.attrs['axesLogScale'] = json.dumps(
            [bool(v) for v in table.axes_log_scale])


def _write_molecule_into(group: 'h5py.Group', molecule) -> None:
    """Serialize a Molecule (or vtkMolecule) under ``group`` using the
    same layout as C++ ``Tvh5Format::writeMoleculePayload``:

    - Group attrs ``type=molecule``, ``numAtoms``, ``numBonds``.
    - ``atomicNumbers`` — uint16 dataset of length numAtoms.
    - ``atomPositions`` — float32 dataset shaped (numAtoms, 3).
    - ``bondAtoms`` — int64 dataset shaped (numBonds, 2), each row is
      a ``(beginAtomId, endAtomId)`` pair.
    - ``bondOrders`` — uint16 dataset of length numBonds (omitted when
      there are no bonds; reader defaults to 1)."""
    if isinstance(molecule, Molecule):
        _write_pure_molecule_into(group, molecule)
        return

    from vtk.util.numpy_support import vtk_to_numpy

    num_atoms = int(molecule.GetNumberOfAtoms())
    num_bonds = int(molecule.GetNumberOfBonds())
    group.attrs['type'] = 'molecule'
    group.attrs['numAtoms'] = np.int64(num_atoms)
    group.attrs['numBonds'] = np.int64(num_bonds)

    if num_atoms > 0:
        atomic_numbers_arr = molecule.GetAtomicNumberArray()
        atomic_numbers = vtk_to_numpy(atomic_numbers_arr).astype(
            np.uint16, copy=False)
        group.create_dataset('atomicNumbers', data=atomic_numbers)
        positions = np.empty((num_atoms, 3), dtype=np.float32)
        for i in range(num_atoms):
            atom = molecule.GetAtom(i)
            pos = atom.GetPosition()
            positions[i, 0] = pos[0]
            positions[i, 1] = pos[1]
            positions[i, 2] = pos[2]
        group.create_dataset('atomPositions', data=positions)

    if num_bonds > 0:
        bond_atoms = np.empty((num_bonds, 2), dtype=np.int64)
        for i in range(num_bonds):
            bond = molecule.GetBond(i)
            bond_atoms[i, 0] = bond.GetBeginAtomId()
            bond_atoms[i, 1] = bond.GetEndAtomId()
        group.create_dataset('bondAtoms', data=bond_atoms)
        orders_arr = molecule.GetBondOrdersArray()
        if orders_arr is not None:
            orders = vtk_to_numpy(orders_arr).astype(np.uint16, copy=False)
            if orders.shape[0] == num_bonds:
                group.create_dataset('bondOrders', data=orders)


def _write_pure_molecule_into(group: 'h5py.Group',
                              molecule: Molecule) -> None:
    """The :class:`tomviz_pipeline.molecule.Molecule` flavor of
    :func:`_write_molecule_into` — identical group layout, written
    straight from the numpy arrays (their dtypes already match)."""
    group.attrs['type'] = 'molecule'
    group.attrs['numAtoms'] = np.int64(molecule.num_atoms)
    group.attrs['numBonds'] = np.int64(molecule.num_bonds)

    if molecule.num_atoms > 0:
        group.create_dataset('atomicNumbers',
                             data=molecule.atomic_numbers)
        group.create_dataset('atomPositions', data=molecule.positions)

    if molecule.num_bonds > 0:
        group.create_dataset('bondAtoms', data=molecule.bonds)
        group.create_dataset('bondOrders', data=molecule.bond_orders)


def write_state_tvh5(target_path, state_json: dict, pipeline) -> None:
    """Write ``state_json`` plus every populated, non-sink output port
    into ``target_path`` as a ``.tvh5`` HDF5 container.

    For every node N with output port P that carries a Dataset payload
    (volume data), the voxels are written under ``/data/<N>/<P>/`` in
    the same EMD layout as a stand-alone ``.emd`` file. ``Table`` ports
    are written column-by-column under ``/data/<N>/<P>/c<i>`` and
    ``Molecule`` ports as ``atomicNumbers`` / ``atomPositions`` /
    ``bondAtoms`` / ``bondOrders`` datasets, mirroring C++
    ``Tvh5Format::writeTablePayload`` and ``writeMoleculePayload``.
    Either way, a ``dataRef`` entry pointing at the group is stamped
    onto the matching port entry in the JSON before serialization.

    Other payload types (raw scalars, etc.) are skipped with a warning
    — they aren't persisted in the tvh5 container today. The caller
    that wants those leaves on disk should request
    ``output_format='state+port'`` so the per-port writers run
    alongside the tvh5 writer."""
    snapshot = copy.deepcopy(state_json)
    nodes_by_id = {entry['id']: entry for entry in
                   snapshot.get('pipeline', {}).get('nodes') or []
                   if 'id' in entry}

    target_path = Path(target_path)
    with h5py.File(target_path, 'w') as f:
        f.create_group('/data')
        for node in pipeline.nodes:
            if isinstance(node, SinkNode):
                continue
            entry = nodes_by_id.get(node.id)
            if entry is None:
                continue
            outputs = entry.setdefault('outputPorts', {})
            for port in node.output_ports():
                # materialize() (not data()) so a persistent-OnDisk
                # port whose payload was evicted to its cache file is
                # still embedded; the handle keeps it alive for the
                # duration of the write.
                handle = port.materialize()
                if handle is None:
                    continue
                payload = handle.data.payload
                is_volume = hasattr(payload, 'arrays')
                is_table = _is_table(payload)
                is_molecule = _is_molecule(payload)
                if not (is_volume or is_table or is_molecule):
                    logger.warning(
                        'tvh5 writer: skipping unsupported port %s.%s '
                        '(port type %r) — only volume, table and '
                        'molecule payloads are persisted.',
                        node.label or type(node).__name__, port.name,
                        port.port_type)
                    continue
                node_group_path = f'/data/{node.id}'
                port_group_path = f'{node_group_path}/{port.name}'
                if node_group_path not in f:
                    f.create_group(node_group_path)
                port_group = f.create_group(port_group_path)
                if is_volume:
                    _write_emd_node_into(port_group, payload)
                elif is_table:
                    _write_table_into(port_group, payload)
                else:
                    _write_molecule_into(port_group, payload)
                outputs.setdefault(port.name, {})['dataRef'] = {
                    'container': 'h5',
                    'path': port_group_path,
                }

        # Serialize the (now dataRef-stamped) JSON as an int8 (signed)
        # array at /tomviz_state. The C++ side reads with
        # H5ReadWrite::readData<char>, which strictly H5Tequal-checks
        # the storage type against H5T_STD_I8LE — using uint8 here
        # makes Tvh5Format::readState fall through to an empty state
        # and tomviz then mis-routes the file to LegacyStateLoader.
        # The bytes are identical either way; only the HDF5 type
        # label differs.
        state_bytes = json.dumps(snapshot).encode('utf-8')
        f.create_dataset('tomviz_state',
                         data=np.frombuffer(state_bytes, dtype='i1'))
