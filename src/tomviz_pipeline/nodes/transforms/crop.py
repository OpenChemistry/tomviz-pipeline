###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Crop — extracts a sub-volume defined by a 6-tuple VTK extent
[xmin, xmax, ymin, ymax, zmin, zmax] (inclusive). Mirrors C++
CropTransform, which uses vtkExtractVOI.

The default sentinel INT_MIN (saved when the user never set bounds)
means "use the whole volume" — same as the C++ widget code which
back-fills the input's full extent when bounds[0] is INT_MIN."""

from __future__ import annotations

import copy

from tomviz_pipeline.core import PortData, TransformNode


# The C++ side uses std::numeric_limits<int>::min() as the sentinel.
_SENTINEL = -(2 ** 31)


class CropTransform(TransformNode):
    type_name = 'transform.crop'

    def __init__(self):
        super().__init__()
        self.add_input('volume', 'ImageData')
        self.add_output('output', 'ImageData')
        self.label = 'Crop'
        self._parameters['bounds'] = [_SENTINEL] * 6

    def serialize(self) -> dict:
        data = super().serialize()
        data['bounds'] = list(self.parameter('bounds'))
        return data

    def deserialize(self, data: dict) -> bool:
        if not super().deserialize(data):
            return False
        bounds = data.get('bounds')
        if isinstance(bounds, list) and len(bounds) == 6:
            self._parameters['bounds'] = [int(b) for b in bounds]
        return True

    def transform(self, inputs):
        primary = inputs.get('volume')
        if primary is None:
            return {}
        dataset = copy.deepcopy(primary.payload)

        active = dataset.active_scalars
        if active is None or active.ndim < 3:
            return {'output': PortData(dataset, primary.port_type)}

        full = [0, active.shape[0] - 1,
                0, active.shape[1] - 1,
                0, active.shape[2] - 1]
        b = list(self.parameter('bounds'))
        for i in range(6):
            if b[i] == _SENTINEL:
                b[i] = full[i]
        # Clamp into the available extent.
        b[0] = max(b[0], full[0])
        b[1] = min(b[1], full[1])
        b[2] = max(b[2], full[2])
        b[3] = min(b[3], full[3])
        b[4] = max(b[4], full[4])
        b[5] = min(b[5], full[5])

        # VTK extent is inclusive; numpy slicing is exclusive on the
        # upper bound.
        sl = (slice(b[0], b[1] + 1),
              slice(b[2], b[3] + 1),
              slice(b[4], b[5] + 1))

        for name in list(dataset.scalars_names):
            arr = dataset.scalars(name)
            if arr.ndim >= 3:
                dataset.set_scalars(name, arr[sl].copy())

        return {'output': PortData(dataset, primary.port_type)}
