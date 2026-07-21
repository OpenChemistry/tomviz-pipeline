###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Cylindrical Crop — masks voxels outside a cylinder with arbitrary axis.

Parameters mirror those of the legacy CylindricalCrop.py operator so that
the same JSON state can drive both the legacy and new pipeline paths."""

import copy

import numpy as np

from tomviz_pipeline.core import PortData, TransformNode


class CylindricalCropTransform(TransformNode):
    type_name = 'transform.cylindricalCrop'

    _DEFAULTS = {
        'center_x': -1.0,
        'center_y': -1.0,
        'center_z': -1.0,
        'axis_x': 0.0,
        'axis_y': 0.0,
        'axis_z': 1.0,
        'radius': -1.0,
        'fill_value': 0.0,
    }

    def __init__(self):
        super().__init__()
        self.add_input('volume', 'ImageData')
        self.add_output('output', 'ImageData')
        self.label = 'Cylindrical Crop'
        self._parameters.update(self._DEFAULTS)

    def serialize(self) -> dict:
        data = super().serialize()
        data.update(self._parameters)
        return data

    def deserialize(self, data: dict) -> bool:
        if not super().deserialize(data):
            return False
        for name, default in self._DEFAULTS.items():
            self._parameters[name] = float(data.get(name, default))
        return True

    def transform(self, inputs):
        primary = inputs.get('volume')
        if primary is None:
            return {}
        dataset = copy.deepcopy(primary.payload)

        active = dataset.active_scalars
        if active is None or active.ndim < 3:
            return {'output': PortData(dataset, primary.port_type)}

        nx, ny, nz = active.shape

        p = self.parameter
        cx = p('center_x') if p('center_x') >= 0 else (nx - 1) / 2.0
        cy = p('center_y') if p('center_y') >= 0 else (ny - 1) / 2.0
        cz = p('center_z') if p('center_z') >= 0 else (nz - 1) / 2.0
        r = p('radius') if p('radius') > 0 else min(nx, ny) / 2.0

        axis = np.array([p('axis_x'), p('axis_y'), p('axis_z')],
                        dtype=np.float64)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-12:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / axis_len

        center = np.array([cx, cy, cz], dtype=np.float64)

        X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                              indexing='ij')

        dx = X - center[0]
        dy = Y - center[1]
        dz = Z - center[2]

        proj = dx * axis[0] + dy * axis[1] + dz * axis[2]

        perp_dist_sq = ((dx - proj * axis[0]) ** 2 +
                        (dy - proj * axis[1]) ** 2 +
                        (dz - proj * axis[2]) ** 2)

        mask = perp_dist_sq > r ** 2

        for name in list(dataset.scalars_names):
            arr = dataset.scalars(name)
            if arr.ndim >= 3:
                result = arr.copy()
                result[mask] = self.parameter('fill_value')
                dataset.set_scalars(name, result)

        return {'output': PortData(dataset, primary.port_type)}
