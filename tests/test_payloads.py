###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Tests for the pure-Python Table / Molecule payload classes, the
make_spreadsheet / make_molecule helpers, and their tvh5 group
(de)serialization — including cross-compat with vtk-built payloads."""

import numpy as np
import h5py
import pytest

from tomviz_pipeline.molecule import Molecule, element_symbol
from tomviz_pipeline.state.state_io import (
    _read_molecule_group,
    _read_table_group,
)
from tomviz_pipeline.state.state_writer import (
    _write_molecule_into,
    _write_table_into,
)
from tomviz_pipeline.table import Table
from tomviz_pipeline.utils import make_molecule, make_spreadsheet


# --- Table ----------------------------------------------------------------


def test_table_columns_and_rows():
    t = Table()
    t.set_column('a', np.arange(3, dtype=np.float32))
    t.set_column('b', ['x', 'y', 'z'])
    assert t.column_names == ['a', 'b']
    assert t.num_columns == 2
    assert t.num_rows == 3
    assert t.column('b') == ['x', 'y', 'z']


def test_table_rejects_mismatched_lengths():
    t = Table()
    t.set_column('a', np.arange(3))
    with pytest.raises(ValueError):
        t.set_column('b', np.arange(4))


def test_table_rejects_2d_columns():
    with pytest.raises(ValueError):
        Table({'a': np.zeros((2, 2))})


def test_make_spreadsheet_returns_table():
    data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    t = make_spreadsheet(['x', 'y'], data,
                         axes_labels=('X', 'Y'),
                         axes_log_scale=(False, True))
    assert isinstance(t, Table)
    assert t.column_names == ['x', 'y']
    # Columns are float32, matching the vtkFloatArray the in-app
    # implementation builds.
    assert t.column('x').dtype == np.float32
    np.testing.assert_allclose(t.column('y'), [10.0, 20.0, 30.0])
    assert t.axes_labels == ('X', 'Y')
    assert t.axes_log_scale == (False, True)


def test_make_spreadsheet_column_count_mismatch_returns_none():
    assert make_spreadsheet(['only'], np.zeros((2, 3))) is None


def test_table_tvh5_round_trip(tmp_path):
    t = Table()
    t.set_column('radius', np.array([1.5, 2.5], dtype=np.float32))
    t.set_column('count', np.array([7, 9], dtype=np.int64))
    t.set_column('label', ['first', 'second'])
    t.axes_labels = ('r', 'n')
    t.axes_log_scale = (False, True)

    path = tmp_path / 'table.h5'
    with h5py.File(path, 'w') as f:
        _write_table_into(f.create_group('t'), t)
    with h5py.File(path, 'r') as f:
        group = f['t']
        # Layout matches C++ Tvh5Format::writeTablePayload.
        assert group.attrs['type'] in ('table', b'table')
        assert int(group.attrs['numColumns']) == 3
        assert int(group.attrs['numRows']) == 2
        assert int(group['c0'].attrs['vtkDataType']) == 10  # VTK_FLOAT
        assert int(group['c1'].attrs['vtkDataType']) == 16  # LONG_LONG
        assert int(group['c2'].attrs['vtkDataType']) == 13  # VTK_STRING
        back = _read_table_group(group)

    assert isinstance(back, Table)
    assert back.column_names == ['radius', 'count', 'label']
    assert back.column('radius').dtype == np.float32
    np.testing.assert_allclose(back.column('radius'), [1.5, 2.5])
    assert back.column('count').dtype == np.int64
    np.testing.assert_array_equal(back.column('count'), [7, 9])
    assert back.column('label') == ['first', 'second']
    assert back.axes_labels == ('r', 'n')
    assert back.axes_log_scale == (False, True)


def test_vtk_table_written_group_reads_as_pure_table(tmp_path):
    """Cross-compat: a group serialized from a vtkTable (the dual-accept
    write path) loads back as an equivalent pure Table."""
    vtk = pytest.importorskip('vtk')

    t = vtk.vtkTable()
    radius = vtk.vtkFloatArray()
    radius.SetName('radius')
    radius.SetNumberOfComponents(1)
    radius.SetNumberOfTuples(2)
    radius.SetValue(0, 1.5)
    radius.SetValue(1, 2.5)
    t.AddColumn(radius)
    labels = vtk.vtkStringArray()
    labels.SetName('labels')
    labels.SetNumberOfValues(2)
    labels.SetValue(0, 'a')
    labels.SetValue(1, 'b')
    t.AddColumn(labels)

    path = tmp_path / 'table.h5'
    with h5py.File(path, 'w') as f:
        _write_table_into(f.create_group('t'), t)
    with h5py.File(path, 'r') as f:
        back = _read_table_group(f['t'])

    assert isinstance(back, Table)
    np.testing.assert_allclose(back.column('radius'), [1.5, 2.5])
    assert back.column('labels') == ['a', 'b']


# --- Molecule ---------------------------------------------------------------


def test_molecule_defaults_and_validation():
    m = Molecule([6, 1], [[0, 0, 0], [1, 0, 0]], bonds=[[0, 1]])
    assert m.num_atoms == 2
    assert m.num_bonds == 1
    # Bond orders default to 1.
    np.testing.assert_array_equal(m.bond_orders, [1])
    assert m.atomic_numbers.dtype == np.uint16
    assert m.positions.dtype == np.float32
    assert m.bonds.dtype == np.int64

    with pytest.raises(ValueError):
        Molecule([6], [[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError):
        Molecule([6, 1], [[0, 0, 0], [1, 0, 0]],
                 bonds=[[0, 1]], bond_orders=[1, 2])


def test_make_molecule_accepts_flat_positions():
    m = make_molecule([1, 8, 1],
                      [0.0, 0.0, 0.0, 0.96, 0.0, 0.0, 1.2, 0.93, 0.0],
                      bonds=[0, 1, 1, 2], bond_orders=[1, 2])
    assert isinstance(m, Molecule)
    assert m.num_atoms == 3
    np.testing.assert_allclose(m.positions[1], [0.96, 0.0, 0.0])
    np.testing.assert_array_equal(m.bonds, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(m.bond_orders, [1, 2])


def test_element_symbol():
    assert element_symbol(1) == 'H'
    assert element_symbol(6) == 'C'
    assert element_symbol(118) == 'Og'
    assert element_symbol(0) == 'X'
    assert element_symbol(999) == 'X'


def test_molecule_tvh5_round_trip(tmp_path):
    m = make_molecule([1, 8], [[0, 0, 0], [0.96, 0, 0]],
                      bonds=[[0, 1]], bond_orders=[2])

    path = tmp_path / 'mol.h5'
    with h5py.File(path, 'w') as f:
        _write_molecule_into(f.create_group('m'), m)
    with h5py.File(path, 'r') as f:
        group = f['m']
        # Layout matches C++ Tvh5Format::writeMoleculePayload.
        assert group.attrs['type'] in ('molecule', b'molecule')
        assert int(group.attrs['numAtoms']) == 2
        assert int(group.attrs['numBonds']) == 1
        assert group['atomicNumbers'].dtype == np.uint16
        assert group['atomPositions'].dtype == np.float32
        assert group['bondAtoms'].dtype == np.int64
        assert group['bondOrders'].dtype == np.uint16
        back = _read_molecule_group(group)

    assert isinstance(back, Molecule)
    np.testing.assert_array_equal(back.atomic_numbers, [1, 8])
    np.testing.assert_allclose(back.positions[1], [0.96, 0, 0])
    np.testing.assert_array_equal(back.bonds, [[0, 1]])
    np.testing.assert_array_equal(back.bond_orders, [2])
