###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""The numpy-backed Dataset passed to operators, plus the LegacyDataset
upgrade used by v1 (``tomviz.operators``-style) transforms. Merges the
old tomviz ``dataset`` ABC and its ``external_dataset`` implementation
into a single concrete class."""

from __future__ import annotations

import collections.abc
from typing import Callable, Optional

import numpy as np

ARRAY_TYPES = (collections.abc.Sequence, np.ndarray)


class Dataset:
    """The standard object that is passed to operators within tomviz.

    It provides a unified interface for accessing and manipulating tilt
    image stacks and volumetric data, including scalar arrays, spacing
    information, tilt series metadata (such as tilt angles), and
    calibration data (dark/white fields).

    This object will always be automatically provided as the first
    argument in the `transform()` function within operators. For
    example, for the `Rotate` operator:

    .. code-block:: python

        def transform(dataset, rotation_angle=90.0, rotation_axis=0):
            # ...

            array = dataset.active_scalars

            # ...
    """

    def __init__(self, arrays=None, active=None):
        # Holds the map of scalars name => array
        if arrays is None:
            arrays = {}
        self.arrays = arrays
        self.tilt_angles = None
        self.tilt_axis = None
        self.scan_ids = None
        # The currently active scalar
        self.active_name = active
        # If we weren't given the active array, set the first as the active
        # array.
        if active is None and len(arrays.keys()):
            self.active_name = next(iter(arrays.keys()))

        self._spacing = None

        # Dark and white backgrounds
        self.dark = None
        self.white = None

        # Filename and metadata
        self.file_name = None
        self.metadata = {}

    @property
    def active_scalars(self) -> np.ndarray:
        """The currently active scalar array data as a numpy array.

        The shape and dtype depend on the specific dataset.
        """
        return self.scalars(self.active_name)

    @active_scalars.setter
    def active_scalars(self, array: np.ndarray):
        """Set the currently active scalar array data."""
        self.set_scalars(self.active_name, array)

    @property
    def num_scalars(self) -> int:
        """The total number of scalar arrays stored in this dataset."""
        return len(self.arrays)

    @property
    def scalars_names(self) -> list[str]:
        """List containing the name of each scalar array in the
        dataset."""
        return list(self.arrays.keys())

    def scalars(self, name: Optional[str] = None) -> np.ndarray:
        """Get a scalar array by name.

        :param name: The name of the scalar array to retrieve. If None,
                     returns the active scalar array.
        :return: The requested scalar array data.
        :raises KeyError: If the specified name does not exist in the
                          dataset.
        """
        if name is None:
            name = self.active_name
        return self.arrays[name]

    def set_scalars(self, name: str, array: np.ndarray):
        """Add or update a scalar array in the dataset.

        :param name: The name to assign to this scalar array. If a
                     field with this name already exists, it will be
                     overwritten.
        :param array: The scalar array data to store.
        """
        self.arrays[name] = array
        # Mirror internal_dataset.Dataset: the first scalar added to an
        # otherwise-empty dataset becomes the active one.
        if self._active_name is None:
            self._active_name = name

    @property
    def spacing(self) -> Optional[tuple]:
        """Voxel spacing in physical units for (x, y, z) dimensions.

        Units depend on the dataset but are typically in nanometers or
        similar physical units.
        """
        return self._spacing

    @spacing.setter
    def spacing(self, v):
        """Set the voxel spacing in physical units for (x, y, z)
        dimensions."""
        if not isinstance(v, ARRAY_TYPES):
            raise Exception('Spacing must be an iterable type')
        if not len(v) == 3:
            raise Exception('Length of spacing must be 3')

        self._spacing = v

    @property
    def active_name(self) -> str:
        """The name of the currently active scalar array."""
        return self._active_name

    @active_name.setter
    def active_name(self, v: str):
        """Set which scalar array is the active one (by name)."""
        self._active_name = v

    @property
    def tilt_axis(self) -> Optional[int]:
        """The axis index around which tilting occurs in a tomographic
        tilt series.

        - 0 = x-axis
        - 1 = y-axis
        - 2 = z-axis
        - None = not applicable or not set
        """
        return self._tilt_axis

    @tilt_axis.setter
    def tilt_axis(self, v: Optional[int]):
        self._tilt_axis = v

    @property
    def tilt_angles(self) -> Optional[np.ndarray]:
        """Array of tilt angles for tomographic tilt series.

        Tilt angles are typically in degrees for each projection in a
        tilt series. Returns None if this is not a tilt series dataset
        or if tilt angles have not been set.
        """
        return self._tilt_angles

    @tilt_angles.setter
    def tilt_angles(self, v: Optional[np.ndarray]):
        """Set the tilt angles for tomographic tilt series.

        Provide None to clear tilt angles.
        """
        self._tilt_angles = v

    @property
    def file_name(self) -> Optional[str]:
        """The original filename this dataset was loaded from.

        Returns None if the dataset was not loaded from a file or if
        the filename is not available.
        """
        return self._file_name

    @file_name.setter
    def file_name(self, v: Optional[str]):
        self._file_name = v

    @property
    def metadata(self) -> dict:
        """Dictionary containing arbitrary metadata associated with the
        dataset.

        Can include acquisition parameters, instrument settings,
        timestamps, etc.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, v: dict):
        self._metadata = v

    @property
    def dark(self) -> Optional[np.ndarray]:
        """Dark field calibration data.

        Dark field images are captured with no illumination and
        represent the baseline signal level of the detector (electronic
        noise, thermal noise, etc.). Returns None if not available.
        """
        return self._dark

    @dark.setter
    def dark(self, v: Optional[np.ndarray]):
        self._dark = v

    @property
    def white(self) -> Optional[np.ndarray]:
        """White field (flat field) calibration data.

        White field images are captured with uniform illumination and
        no sample, representing the detector response and illumination
        variations. Used for flat-field correction. Returns None if not
        available.
        """
        return self._white

    @white.setter
    def white(self, v: Optional[np.ndarray]):
        self._white = v

    @property
    def scan_ids(self) -> Optional[np.ndarray]:
        """Array of scan IDs associated with each projection in a tilt
        series.

        Returns None if scan IDs have not been set.
        """
        return self._scan_ids

    @scan_ids.setter
    def scan_ids(self, v: Optional[np.ndarray]):
        """Set the scan IDs for projections in a tilt series.

        Provide None to clear scan IDs.
        """
        self._scan_ids = v

    def remove_scalars(self, name: str):
        """Remove a scalar array from the dataset.

        :param name: The name of the scalar array to remove.
        :raises KeyError: If the specified name does not exist in the
                          dataset.
        """
        if name not in self.arrays:
            raise KeyError(f"No scalar array named '{name}'")
        del self.arrays[name]
        if self.active_name == name and self.arrays:
            self.active_name = next(iter(self.arrays.keys()))

    def rename_active(self, new_name: str):
        self.arrays[new_name] = self.arrays.pop(self.active_name)
        # Keep the pointer in sync with the dict key.
        self._active_name = new_name

    def empty_copy(self) -> 'Dataset':
        """Return a new Dataset of the same concrete type with the
        same metadata (spacing, tilt angles, dimensions, file_name,
        ...) but no scalar arrays. Useful as a starting point for
        building a derived dataset that shares geometry with this one.
        """
        # Shallow-copy the instance dict so metadata (spacing, tilt
        # angles, file_name, ...) carries over, then drop the arrays.
        # type(self) preserves the concrete subclass (Dataset vs
        # LegacyDataset) so the upgrade is sticky.
        new_ds = type(self).__new__(type(self))
        new_ds.__dict__.update(self.__dict__)
        new_ds.arrays = {}
        return new_ds

    def apply_to_each_scalar_array(
            self, fn: Callable[[np.ndarray], Optional[np.ndarray]]
    ) -> 'Dataset':
        """Return a new Dataset with each scalar array replaced by
        ``fn(array)``. The original dataset is not modified.
        Active-scalar selection is preserved when possible.

        ``fn`` may either return a new numpy array or mutate the input
        in place and return it; either form works. Returning
        ``None`` excludes that array from the output — useful for
        filtering arrays out of the dataset alongside the per-array
        transformation.

        If the original active-scalar array is filtered out (``fn``
        returned ``None`` for it), the first remaining array becomes
        the new active scalar. If every array is filtered out, the
        result is an empty dataset.

        Useful when an operator's per-array logic should run uniformly
        across every scalar array in the dataset — saves writing the
        explicit ``for name in dataset.scalars_names: …`` loop.

        Example:

        .. code-block:: python

            class AddConstant(tomviz.nodes.TransformNode):
                def transform(self, inputs, constant=0.0):
                    ds = inputs["volume"]
                    return {
                        "volume": ds.apply_to_each_scalar_array(
                            lambda a: a + constant)
                    }
        """
        result = self.empty_copy()
        active = self.active_name
        # Snapshot names so iteration is stable if fn does anything
        # surprising to the source dataset.
        names = list(self.scalars_names)
        for name in names:
            new_arr = fn(self.scalars(name))
            if new_arr is None:
                continue
            result.set_scalars(name, new_arr)
        if active is not None and active in result.scalars_names:
            result.active_name = active
        elif result.scalars_names:
            # The original active was filtered out — fall back to the
            # first remaining scalar so the output's active_name
            # references an array that actually exists.
            result.active_name = result.scalars_names[0]
        return result


class LegacyDataset(Dataset):
    """Adds the v1 ``create_child_dataset`` API expected by
    ``tomviz.operators``-derived recon-style operators. The CLI's
    LegacyPythonTransform upgrades each input to a LegacyDataset
    before invoking a v1 operator; v2 nodes see the base
    :class:`Dataset` instead, so the legacy affordance never leaks
    into a v2 author's namespace."""

    @classmethod
    def from_dataset(cls, base: Dataset) -> 'LegacyDataset':
        """Build a LegacyDataset that shares state with @a base.
        Bypasses ``__init__`` and copies the instance dict, which is
        safe here because LegacyDataset adds no new instance state.
        Used by the CLI's LegacyPythonTransform to upgrade the
        deep-copied input on its way into a v1 operator."""
        legacy = cls.__new__(cls)
        legacy.__dict__.update(base.__dict__)
        return legacy

    def create_child_dataset(self):
        # Empty child — matches the in-app internal_dataset.LegacyDataset.
        # Inheriting the parent's scalars would leave the un-written
        # ones at the input shape while the active one takes the
        # output shape, breaking downstream readers.
        child = LegacyDataset({}, active=self.active_name)
        if self.spacing is not None:
            s = self.spacing
            if self.tilt_angles is not None:
                child.spacing = [s[0], s[1], s[0]]
            else:
                child.spacing = list(s)
        return child
