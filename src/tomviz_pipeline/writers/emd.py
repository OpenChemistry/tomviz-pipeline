###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""EMD writer for ImageData/Volume/TiltSeries/LabelMap port payloads."""

from pathlib import Path

from tomviz_pipeline.io.emd import _write_emd


def write_emd(payload, target_path: Path) -> None:
    if not hasattr(payload, 'arrays'):
        raise TypeError(
            'EMD writer expected a tomviz Dataset, got '
            f'{type(payload).__name__}')
    _write_emd(target_path, payload)
