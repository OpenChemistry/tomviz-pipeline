###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Tests for the per-port-type output writers (CSV for tables, XYZ for
molecules, plus the registry plumbing). The primary payloads are the
library's pure-Python Table / Molecule; the vtk dual-accept paths are
covered at the end and skip when vtk isn't installed."""

import csv

import numpy as np
import pytest

from tomviz_pipeline.molecule import Molecule
from tomviz_pipeline.table import Table
from tomviz_pipeline.writers import (
    register_writer,
    writer_for,
    write_molecule_xyz,
    write_table_csv,
)


def _build_table() -> Table:
    """A Table with a numeric column and a string column."""
    t = Table()
    t.set_column('radius', np.array([1.5, 2.0, 3.25], dtype=np.float32))
    t.set_column('label', ['small', 'medium', 'large'])
    return t


def _build_methane_molecule() -> Molecule:
    """Methane (CH4) with one bond per H — bonds aren't written by the
    XYZ writer but we add them to verify they're tolerated."""
    return Molecule(
        atomic_numbers=[6, 1, 1, 1, 1],
        positions=[[0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [-1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, -1.0, 0.0]],
        bonds=[[0, 1], [0, 2], [0, 3], [0, 4]],
    )


def test_label_maps_are_written_as_emd(tmp_path):
    """Segmentation results leave a pipeline on LabelMap ports; they are
    image data and get the EMD writer like any other volume."""
    from tomviz_pipeline import Dataset
    from tomviz_pipeline.io import load_dataset

    entry = writer_for('LabelMap')
    assert entry is not None
    extension, writer = entry
    assert extension == 'emd'

    labels = np.zeros((3, 4, 5), dtype=np.uint8, order='F')
    labels[1, 2, 3] = 7
    dataset = Dataset({'Labels': labels}, 'Labels')
    target = tmp_path / f'labels.{extension}'
    writer(dataset, target)
    assert np.array_equal(load_dataset(target).active_scalars, labels)


def test_table_csv_round_trip(tmp_path):
    target = tmp_path / 'out.csv'
    write_table_csv(_build_table(), target)
    with open(target, newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == ['radius', 'label']
    # Numeric values come out as Python floats (str-formatted by csv).
    assert rows[1] == ['1.5', 'small']
    assert rows[2] == ['2.0', 'medium']
    assert rows[3] == ['3.25', 'large']


def test_table_csv_rejects_non_table(tmp_path):
    with pytest.raises(TypeError):
        write_table_csv({'not': 'a table'}, tmp_path / 'x.csv')


def test_molecule_xyz_writes_methane(tmp_path):
    target = tmp_path / 'methane.xyz'
    write_molecule_xyz(_build_methane_molecule(), target)
    lines = target.read_text().splitlines()
    assert lines[0] == '5'
    # Comment line is freeform; just make sure it's there.
    assert lines[1]
    # Five atom lines starting with element symbols.
    symbols = [line.split()[0] for line in lines[2:7]]
    assert symbols == ['C', 'H', 'H', 'H', 'H']
    # First C is at the origin.
    parts = lines[2].split()
    assert float(parts[1]) == pytest.approx(0.0)
    assert float(parts[2]) == pytest.approx(0.0)
    assert float(parts[3]) == pytest.approx(0.0)


def test_molecule_xyz_rejects_non_molecule(tmp_path):
    with pytest.raises(TypeError):
        write_molecule_xyz({'not': 'a molecule'}, tmp_path / 'x.xyz')


def test_registry_lookup():
    # Built-in port types resolve to their declared writers.
    assert writer_for('Volume')[0] == 'emd'
    assert writer_for('TiltSeries')[0] == 'emd'
    assert writer_for('Table')[0] == 'csv'
    assert writer_for('Molecule')[0] == 'xyz'
    # Unknown types return None so the CLI can skip with a warning.
    assert writer_for('Bogus') is None


def test_register_writer_overrides(tmp_path):
    sentinel = []

    def custom(payload, target):
        sentinel.append((payload, target))
        target.write_text('ok')

    try:
        register_writer('Bogus', 'bin', custom)
        ext, w = writer_for('Bogus')
        assert ext == 'bin'
        target = tmp_path / 'out.bin'
        w('payload', target)
        assert target.read_text() == 'ok'
        assert sentinel == [('payload', target)]
    finally:
        # Clean up — the registry is process-global.
        from tomviz_pipeline.writers import _REGISTRY
        _REGISTRY.pop('Bogus', None)


# --- vtk dual-accept: payloads built by scripts that use vtk directly ---


def test_table_csv_accepts_vtk_table(tmp_path):
    vtk = pytest.importorskip('vtk')

    t = vtk.vtkTable()
    radius = vtk.vtkFloatArray()
    radius.SetName('radius')
    radius.SetNumberOfComponents(1)
    radius.SetNumberOfTuples(2)
    radius.SetValue(0, 1.5)
    radius.SetValue(1, 2.0)
    t.AddColumn(radius)

    target = tmp_path / 'out.csv'
    write_table_csv(t, target)
    with open(target, newline='') as f:
        rows = list(csv.reader(f))
    assert rows == [['radius'], ['1.5'], ['2.0']]


def test_molecule_xyz_accepts_vtk_molecule(tmp_path):
    vtk = pytest.importorskip('vtk')

    m = vtk.vtkMolecule()
    m.AppendAtom(6, 0.0, 0.0, 0.0)
    m.AppendAtom(1, 1.0, 0.0, 0.0)
    m.AppendBond(0, 1, 1)

    target = tmp_path / 'ch.xyz'
    write_molecule_xyz(m, target)
    lines = target.read_text().splitlines()
    assert lines[0] == '2'
    assert lines[2].split()[0] == 'C'
    assert lines[3].split()[0] == 'H'
