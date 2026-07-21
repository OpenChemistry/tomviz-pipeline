###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""ConvertToFloat — casts every scalar array to float32. Mirrors C++
ConvertToFloatTransform."""

import copy

import numpy as np

from tomviz_pipeline.core import PortData, TransformNode


class ConvertToFloatTransform(TransformNode):
    type_name = 'transform.convertToFloat'

    def __init__(self):
        super().__init__()
        self.add_input('volume', 'ImageData')
        self.add_output('output', 'ImageData')
        self.label = 'Convert to Float'

    def transform(self, inputs):
        primary = inputs.get('volume')
        if primary is None:
            return {}
        dataset = copy.deepcopy(primary.payload)
        for name in list(dataset.scalars_names):
            arr = dataset.scalars(name)
            if arr.dtype != np.float32:
                dataset.set_scalars(name, arr.astype(np.float32))
        return {'output': PortData(dataset, primary.port_type)}
