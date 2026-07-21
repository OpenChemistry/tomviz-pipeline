###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Pure-Python payload for ``Molecule`` output ports.

Mirrors the role vtkMolecule plays in the C++ application: atoms
(atomic number + position) and optional bonds (atom-index pairs with a
bond order). Dtypes match the tvh5 on-disk layout written by
``state_writer._write_molecule_into`` so serialization is a straight
dump. Applications convert to their native molecule type (e.g.
vtkMolecule) at the visualization boundary."""

from __future__ import annotations

import numpy as np

# Element symbols indexed by atomic number (0 = placeholder), matching
# vtkPeriodicTable so XYZ output is identical to the vtk-backed path.
ELEMENT_SYMBOLS = [
    'X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
    'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
    'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
    'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
    'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
    'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
    'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
    'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
    'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
]


def element_symbol(atomic_number: int) -> str:
    """Symbol for an atomic number, or 'X' when out of range."""
    if 0 < atomic_number < len(ELEMENT_SYMBOLS):
        return ELEMENT_SYMBOLS[atomic_number]
    return 'X'


class Molecule:
    """Atomic structure flowing through a ``Molecule`` port.

    - ``atomic_numbers`` — uint16 array of shape (num_atoms,)
    - ``positions``      — float32 array of shape (num_atoms, 3);
      flat length-3N input is reshaped
    - ``bonds``          — int64 array of shape (num_bonds, 2), each
      row a (begin_atom_index, end_atom_index) pair
    - ``bond_orders``    — uint16 array of shape (num_bonds,);
      defaults to all 1s when bonds are given without orders"""

    def __init__(self, atomic_numbers=None, positions=None,
                 bonds=None, bond_orders=None):
        self.atomic_numbers = np.asarray(
            atomic_numbers if atomic_numbers is not None else [],
            dtype=np.uint16).ravel()
        self.positions = np.asarray(
            positions if positions is not None else [],
            dtype=np.float32).reshape(-1, 3)
        if self.positions.shape[0] != self.atomic_numbers.shape[0]:
            raise ValueError(
                f'Molecule has {self.atomic_numbers.shape[0]} atomic '
                f'numbers but {self.positions.shape[0]} positions')

        self.bonds = np.asarray(
            bonds if bonds is not None else [],
            dtype=np.int64).reshape(-1, 2)
        if bond_orders is None:
            self.bond_orders = np.ones(self.bonds.shape[0],
                                       dtype=np.uint16)
        else:
            self.bond_orders = np.asarray(
                bond_orders, dtype=np.uint16).ravel()
        if self.bond_orders.shape[0] != self.bonds.shape[0]:
            raise ValueError(
                f'Molecule has {self.bonds.shape[0]} bonds but '
                f'{self.bond_orders.shape[0]} bond orders')

    @property
    def num_atoms(self) -> int:
        return int(self.atomic_numbers.shape[0])

    @property
    def num_bonds(self) -> int:
        return int(self.bonds.shape[0])

    def __repr__(self):
        return (f'Molecule(atoms={self.num_atoms}, '
                f'bonds={self.num_bonds})')
