###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Threshold — produces a binary uint8 label map from the active scalar
array, with 1 inside [minValue, maxValue] and 0 outside. Mirrors
C++ ThresholdTransform.

Note: the C++ class does not currently serialize its min/max values
(no override of serialize/deserialize beyond TransformNode), so a
state-file-loaded ThresholdTransform always uses defaults until it is
edited interactively. We keep the same defaults here."""

import numpy as np

from tomviz_pipeline.core import PortData, TransformNode
from tomviz_pipeline.dataset import Dataset


class ThresholdTransform(TransformNode):
    type_name = 'transform.threshold'

    def __init__(self):
        super().__init__()
        self.add_input('volume', 'ImageData')
        self.add_output('mask', 'LabelMap')
        self.label = 'Threshold'
        self._parameters.update(minValue=-1e30, maxValue=0.0)

    def serialize(self) -> dict:
        data = super().serialize()
        data.update(self._parameters)
        return data

    def deserialize(self, data: dict) -> bool:
        if not super().deserialize(data):
            return False
        if 'minValue' in data:
            self._parameters['minValue'] = float(data['minValue'])
        if 'maxValue' in data:
            self._parameters['maxValue'] = float(data['maxValue'])
        return True

    def transform(self, inputs):
        primary = inputs.get('volume')
        if primary is None:
            return {}
        src = primary.payload
        active = src.active_scalars
        if active is None:
            return {}
        min_value = self.parameter('minValue')
        max_value = self.parameter('maxValue')
        mask = np.where(
            (active >= min_value) & (active <= max_value),
            np.uint8(1), np.uint8(0))

        out = Dataset({'Mask': mask}, 'Mask')
        out.spacing = src.spacing
        out.metadata = dict(src.metadata) if src.metadata else {}
        return {'mask': PortData(out, 'LabelMap')}
