###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Per-port-type output writers used by the CLI to persist leaf-node
outputs. Each writer is a callable `(payload, target_path) -> None`
paired with a file extension. The CLI's `_write_outputs` looks up the
writer by `port.port_type` and falls back to a "skip with warning"
behavior for unknown types."""

from typing import Callable

from tomviz_pipeline.writers.emd import write_emd
from tomviz_pipeline.writers.molecule_xyz import write_molecule_xyz
from tomviz_pipeline.writers.table_csv import write_table_csv


# (extension_without_dot, writer)
_REGISTRY: dict[str, tuple[str, Callable]] = {
    'ImageData': ('emd', write_emd),
    'Volume': ('emd', write_emd),
    'TiltSeries': ('emd', write_emd),
    'Image': ('emd', write_emd),
    # A label map is an image dataset whose voxels are labels; the
    # thresholding and segmentation transforms hand their results out
    # on ports of this type.
    'LabelMap': ('emd', write_emd),
    'Table': ('csv', write_table_csv),
    'Molecule': ('xyz', write_molecule_xyz),
}


def writer_for(port_type: str):
    """Return ``(extension, writer)`` for a port type, or ``None`` if no
    writer is registered."""
    return _REGISTRY.get(port_type)


def register_writer(port_type: str, extension: str,
                    writer: Callable) -> None:
    """Register a custom writer for a port type. Useful for downstream
    code that wants to plug in additional formats."""
    _REGISTRY[port_type] = (extension, writer)


__all__ = ['register_writer', 'writer_for', 'write_emd',
           'write_molecule_xyz', 'write_table_csv']
