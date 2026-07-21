###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Per-transform parity tests for the built-in transforms. Each test
exercises the Python implementation against the documented semantics of
the C++ transforms they mirror (tomviz repo, tomviz/pipeline/
transforms/*.cxx), plus LegacyPythonTransform running real operator
scripts (inlined below so the suite is self-contained)."""

import json

import numpy as np
import pytest

from tomviz_pipeline import PortData, register_builtins
from tomviz_pipeline.dataset import Dataset
from tomviz_pipeline.nodes.transforms.convert_to_float import (
    ConvertToFloatTransform,
)
from tomviz_pipeline.nodes.transforms.convert_to_volume import (
    ConvertToVolumeTransform,
)
from tomviz_pipeline.nodes.transforms.crop import CropTransform
from tomviz_pipeline.nodes.transforms.legacy_python import (
    LegacyPythonTransform,
)
from tomviz_pipeline.nodes.transforms.set_tilt_angles import (
    SetTiltAnglesTransform,
)
from tomviz_pipeline.nodes.transforms.threshold import ThresholdTransform
from tomviz_pipeline.nodes.transforms.transpose import TransposeDataTransform


@pytest.fixture(autouse=True)
def _builtins():
    register_builtins()


def _make_dataset(arr, name='ImageScalars'):
    ds = Dataset({name: arr}, name)
    ds.spacing = (1.0, 1.0, 1.0)
    return ds


def _run(transform, dataset, port_type='ImageData'):
    inputs = {'volume': PortData(dataset, port_type)}
    return transform.transform(inputs)


# ---- ConvertToFloat -----------------------------------------------------


def test_convert_to_float_casts_int_array():
    arr = np.arange(24, dtype=np.uint8).reshape((2, 3, 4))
    out = _run(ConvertToFloatTransform(), _make_dataset(arr))
    out_arr = out['output'].payload.active_scalars
    assert out_arr.dtype == np.float32
    np.testing.assert_array_equal(out_arr, arr.astype(np.float32))


# ---- ConvertToVolume ----------------------------------------------------


def test_convert_to_volume_strips_tilt_angles():
    arr = np.zeros((2, 2, 2), dtype=np.float32)
    ds = _make_dataset(arr)
    ds.tilt_angles = np.array([0.0, 1.0])
    ds.tilt_axis = 2
    out = _run(ConvertToVolumeTransform(), ds, port_type='TiltSeries')
    out_ds = out['output'].payload
    assert out_ds.tilt_angles is None
    assert out_ds.tilt_axis is None
    assert out['output'].port_type == 'Volume'


# ---- SetTiltAngles ------------------------------------------------------


def test_set_tilt_angles_expands_sparse_map():
    arr = np.zeros((2, 2, 4), dtype=np.float32)
    ds = _make_dataset(arr)
    t = SetTiltAnglesTransform()
    # As written by the C++ side: string keys, sparse coverage.
    t.deserialize({'angles': {'0': -10.0, '3': 20.0}})
    out = _run(t, ds)
    out_ds = out['output'].payload
    np.testing.assert_array_equal(out_ds.tilt_angles,
                                  np.array([-10.0, 0.0, 0.0, 20.0]))
    assert out['output'].port_type == 'TiltSeries'


# ---- TransposeData ------------------------------------------------------


def test_transpose_data_swaps_i_and_k_axes():
    arr = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    out = _run(TransposeDataTransform(), _make_dataset(arr))
    out_arr = out['output'].payload.active_scalars
    np.testing.assert_array_equal(out_arr, np.transpose(arr, (2, 1, 0)))


# ---- Crop ---------------------------------------------------------------


def test_crop_default_sentinel_returns_full_volume():
    arr = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    t = CropTransform()  # bounds left at INT_MIN sentinel
    out = _run(t, _make_dataset(arr))
    np.testing.assert_array_equal(out['output'].payload.active_scalars, arr)


def test_crop_inclusive_bounds():
    arr = np.arange(60, dtype=np.float32).reshape((3, 4, 5))
    t = CropTransform()
    # VTK extent is inclusive on both ends.
    t.deserialize({'bounds': [0, 1, 1, 2, 2, 4]})
    out = _run(t, _make_dataset(arr))
    np.testing.assert_array_equal(
        out['output'].payload.active_scalars,
        arr[0:2, 1:3, 2:5])


def test_crop_clamps_to_full_extent():
    arr = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    t = CropTransform()
    t.deserialize({'bounds': [-5, 10, -2, 100, 0, 99]})
    out = _run(t, _make_dataset(arr))
    np.testing.assert_array_equal(out['output'].payload.active_scalars, arr)


# ---- Threshold ----------------------------------------------------------


def test_threshold_produces_binary_mask():
    arr = np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.float32)
    ds = _make_dataset(arr)
    t = ThresholdTransform()
    t.deserialize({'minValue': 1.5, 'maxValue': 3.5})
    out = _run(t, ds)
    mask = out['mask'].payload.active_scalars
    expected = ((arr >= 1.5) & (arr <= 3.5)).astype(np.uint8)
    np.testing.assert_array_equal(mask, expected)
    # The port is typed LabelMap, so the mask has to be integral for its
    # two states to be enumerable as labels.
    assert mask.dtype == np.uint8


# ---- LegacyPythonTransform ----------------------------------------------
# Real operator scripts, inlined verbatim from the tomviz repo
# (tomviz/python/AddConstant.{py,json} and CylindricalCrop.{py,json}) so
# LegacyPythonTransform is exercised against genuine operator code.

ADD_CONSTANT_DESCRIPTION = json.dumps({
    'name': 'AddConstant',
    'label': 'Add Constant',
    'description': 'Add a constant value to each voxel in the dataset.',
    'parameters': [
        {
            'name': 'constant',
            'label': 'Constant',
            'description': 'Constant factor added to each voxel.',
            'type': 'double',
            'default': 0.0,
            'precision': 3,
        },
    ],
    'inputType': 'ImageData',
    'outputType': 'ImageData',
})

ADD_CONSTANT_SCRIPT = '''
def transform(dataset, constant=0.0):
    """Add a constant to the data set"""

    import numpy as np

    scalars = dataset.active_scalars
    if scalars is None:
        raise RuntimeError("No scalars found!")

    # Ensure we start with a float
    constant = float(constant)

    # Try to be a little smart so that we don't always just produce a
    # double-precision output
    newMin = np.min(scalars) + constant
    newMax = np.max(scalars) + constant
    if (constant).is_integer() and newMin.is_integer() and newMax.is_integer():
        # Let ints be ints!
        constant = int(constant)
        newMin = int(newMin)
        newMax = int(newMax)
    newMin = np.min_scalar_type(newMin).type(newMin)
    newMax = np.min_scalar_type(newMax).type(newMax)
    for dtype in [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32,
                  np.uint64, np.int64, np.float32, np.float64]:
        if np.can_cast(newMin, dtype) and np.can_cast(newMax, dtype):
            constant = np.array([constant], dtype=dtype)
            break

    # numpy should cast to an appropriate output type to avoid overflow
    dataset.active_scalars = scalars + constant
'''


def test_legacy_python_transform_runs_real_operator():
    """LegacyPythonTransform must be able to load a JSON-described
    Python operator, execute its `transform()`, and return the mutated
    dataset on its primary output port."""
    t = LegacyPythonTransform()
    t.deserialize({'description': ADD_CONSTANT_DESCRIPTION,
                   'script': ADD_CONSTANT_SCRIPT,
                   'arguments': {'constant': 7.0}})

    arr = np.zeros((2, 2, 3), dtype=np.float32)
    ds = Dataset({'ImageScalars': arr}, 'ImageScalars')
    ds.spacing = (1.0, 1.0, 1.0)
    result = t.transform({'volume': PortData(ds, 'ImageData')})
    out_arr = result[t._primary_output_name].payload.active_scalars
    np.testing.assert_array_equal(out_arr, np.full_like(arr, 7.0))


# ---- CylindricalCrop (legacy operator) ------------------------------------

CYLINDRICAL_CROP_DESCRIPTION = json.dumps({
    'name': 'CylindricalCrop',
    'label': 'Cylindrical Crop',
    'description': 'Crop a volume to a cylindrical region.',
    'parameters': [
        {'name': 'center_x', 'type': 'double', 'default': -1.0},
        {'name': 'center_y', 'type': 'double', 'default': -1.0},
        {'name': 'center_z', 'type': 'double', 'default': -1.0},
        {'name': 'axis_x', 'type': 'double', 'default': 0.0},
        {'name': 'axis_y', 'type': 'double', 'default': 0.0},
        {'name': 'axis_z', 'type': 'double', 'default': 1.0},
        {'name': 'radius', 'type': 'double', 'default': -1.0},
        {'name': 'fill_value', 'type': 'double', 'default': 0.0},
    ],
    'inputType': 'ImageData',
    'outputType': 'ImageData',
})

CYLINDRICAL_CROP_SCRIPT = '''
def transform(dataset, center_x=-1.0, center_y=-1.0, center_z=-1.0,
              axis_x=0.0, axis_y=0.0, axis_z=1.0,
              radius=-1.0, fill_value=0.0):
    """Crop a volume to a cylindrical region.

    Array convention: shape = (nx, ny, nz), axis 0 = X, 1 = Y, 2 = Z.
    """
    import numpy as np

    array = dataset.active_scalars
    if array is None or array.ndim < 3:
        return

    nx, ny, nz = array.shape

    if center_x < 0:
        center_x = (nx - 1) / 2.0
    if center_y < 0:
        center_y = (ny - 1) / 2.0
    if center_z < 0:
        center_z = (nz - 1) / 2.0

    axis = np.array([axis_x, axis_y, axis_z], dtype=np.float64)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-12:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = axis / axis_len

    if radius <= 0:
        radius = min(nx, ny) / 2.0

    center = np.array([center_x, center_y, center_z], dtype=np.float64)

    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                          indexing='ij')

    dx = X - center[0]
    dy = Y - center[1]
    dz = Z - center[2]

    proj = dx * axis[0] + dy * axis[1] + dz * axis[2]

    perp_dist_sq = (dx - proj * axis[0]) ** 2 + \\
                   (dy - proj * axis[1]) ** 2 + \\
                   (dz - proj * axis[2]) ** 2

    mask = perp_dist_sq > radius ** 2

    result = array.copy()
    result[mask] = fill_value

    dataset.active_scalars = result
'''


def _run_cylindrical_crop(arr, **kwargs):
    t = LegacyPythonTransform()
    t.deserialize({'description': CYLINDRICAL_CROP_DESCRIPTION,
                   'script': CYLINDRICAL_CROP_SCRIPT,
                   'arguments': kwargs})

    ds = Dataset({'ImageScalars': arr.copy()}, 'ImageScalars')
    ds.spacing = (1.0, 1.0, 1.0)
    result = t.transform({'volume': PortData(ds, 'ImageData')})
    return result[t._primary_output_name].payload.active_scalars


def test_cylindrical_crop_defaults_preserve_center():
    """With default params (center=volume center, radius=min(nx,ny)/2),
    the center voxel should always be preserved."""
    arr = np.ones((10, 10, 10), dtype=np.float32)
    out = _run_cylindrical_crop(arr)
    assert out[5, 5, 5] == 1.0


def test_cylindrical_crop_zeros_corners():
    """Corners of a cube should be outside the default cylinder."""
    arr = np.ones((10, 10, 10), dtype=np.float32)
    out = _run_cylindrical_crop(arr)
    assert out[0, 0, 5] == 0.0
    assert out[9, 9, 5] == 0.0
    assert out[0, 9, 5] == 0.0
    assert out[9, 0, 5] == 0.0


def test_cylindrical_crop_custom_fill_value():
    arr = np.ones((10, 10, 10), dtype=np.float32)
    out = _run_cylindrical_crop(arr, fill_value=-999.0)
    assert out[0, 0, 0] == -999.0


def test_cylindrical_crop_small_radius():
    """A very small radius should zero out almost everything."""
    arr = np.ones((10, 10, 10), dtype=np.float32)
    out = _run_cylindrical_crop(arr, radius=0.5)
    kept = np.count_nonzero(out)
    assert kept < arr.size * 0.05  # less than 5% preserved


def test_cylindrical_crop_large_radius_preserves_all():
    """A very large radius should keep everything."""
    arr = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
    out = _run_cylindrical_crop(arr, radius=100.0)
    np.testing.assert_array_equal(out, arr)


def test_cylindrical_crop_custom_axis():
    """With axis along X, the crop should be cylindrical around X."""
    arr = np.ones((10, 10, 10), dtype=np.float32)
    out = _run_cylindrical_crop(
        arr, axis_x=1.0, axis_y=0.0, axis_z=0.0, radius=2.0,
    )
    # Center of YZ face should be preserved
    assert out[0, 5, 5] == 1.0
    # Corner of YZ face should be zeroed
    assert out[0, 0, 0] == 0.0
