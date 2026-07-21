###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""SetTiltAngles — assigns tilt angles to the slice axis, turning a
volume into a TiltSeries. Mirrors C++ SetTiltAnglesTransform.

The serialized form is `{"angles": {"<index>": <value>, ...}}` — a sparse
map from slice index (string) to angle. We expand it into a dense
QVector<double>(numSlices) using the volume's z-extent, exactly like
the C++ side does."""

from __future__ import annotations

import copy

import numpy as np

from tomviz_pipeline.core import PortData, TransformNode


class SetTiltAnglesTransform(TransformNode):
    type_name = 'transform.setTiltAngles'

    def __init__(self):
        super().__init__()
        self.add_input('volume', 'ImageData')
        self.add_output('output', 'TiltSeries')
        self.label = 'Set Tilt Angles'
        # Schema form: sparse {"<index>": <angle>} map with string keys.
        self._parameters['angles'] = {}

    def serialize(self) -> dict:
        data = super().serialize()
        data['angles'] = dict(self.parameter('angles'))
        return data

    def deserialize(self, data: dict) -> bool:
        if not super().deserialize(data):
            return False
        angles_obj = data.get('angles', {}) or {}
        self._parameters['angles'] = dict(angles_obj)
        return True

    def transform(self, inputs):
        primary = inputs.get('volume')
        if primary is None:
            return {}
        dataset = copy.deepcopy(primary.payload)

        # The slice axis matches the C++ side's dimensions[2]: the
        # tilt-axis dimension of the active scalar array. After EMD load
        # this is the third axis of the (Fortran-ordered) array.
        active = dataset.active_scalars
        num_slices = active.shape[2] if active.ndim >= 3 else len(active)

        angles = np.zeros(num_slices, dtype=np.float64)
        for key, val in self.parameter('angles').items():
            idx = int(key)
            if 0 <= idx < num_slices:
                angles[idx] = float(val)

        dataset.tilt_angles = angles
        if dataset.tilt_axis is None:
            dataset.tilt_axis = 2
        return {'output': PortData(dataset, 'TiltSeries')}
