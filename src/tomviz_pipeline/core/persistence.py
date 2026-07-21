###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Output-port persistence: enums, the pipeline-wide default policy, and
the disk cache used by OnDisk ports. Mirrors the C++ PersistenceMode /
PipelineSettings / PortDataDiskCache trio.

A port is either *transient* (`persistent == False` — its payload lives
only while some handle is held) or *persistent*, in which case
`persistence_mode` picks the storage medium:

- InMemory: the port pins a strong reference forever (legacy behavior).
- OnDisk: the strong reference is released once no consumer holds one;
  the payload is serialized to a temporary cache file and lazily
  reloaded on the next materialize().

The disk serializer is pluggable via set_port_data_serializer(); the
default uses pickle so the core stays dependency-free. Payloads that
fail to serialize are simply dropped — the execution planner notices the
missing output and re-runs the producer, matching the C++ behavior for
non-serializable payloads.
"""

from __future__ import annotations

import enum
import logging
import os
import pickle
import tempfile
from typing import Callable, Optional

from .events import Signal

logger = logging.getLogger('tomviz_pipeline')


class PersistenceMode(enum.Enum):
    """Storage medium for a *persistent* port's payload. Irrelevant while
    a port is transient."""

    InMemory = 'memory'
    OnDisk = 'disk'


def persistence_mode_to_string(mode: PersistenceMode) -> str:
    return 'disk' if mode == PersistenceMode.OnDisk else 'memory'


def persistence_mode_from_string(value: str) -> PersistenceMode:
    if value == 'disk':
        return PersistenceMode.OnDisk
    return PersistenceMode.InMemory


class DataLocation(enum.Enum):
    """Where a port's payload currently resides."""

    Nowhere = 'none'
    InMemory = 'memory'
    OnDisk = 'disk'


class TransformPersistenceDefault(enum.Enum):
    """Pipeline-wide default applied to transform outputs at add_output
    time. Encodes the (persistent, mode) pair as one choice."""

    InMemory = 'memory'
    OnDisk = 'disk'
    Transient = 'transient'


class PipelineSettings:
    """Process-wide pipeline settings (the C++ version is QSettings-backed;
    here it is plain in-process state an application configures at
    startup).

    transform_persistence_default is consulted by
    TransformNode.add_output only — changing it never retouches existing
    ports. The library default is InMemory (data stays available after a
    run, like the original Python runtime); the tomviz application uses
    OnDisk to keep large intermediate volumes out of RAM.

    Signals:
      transform_persistence_default_changed(TransformPersistenceDefault)
    """

    _instance: Optional['PipelineSettings'] = None

    def __init__(self):
        self._transform_persistence_default = \
            TransformPersistenceDefault.InMemory
        self.transform_persistence_default_changed = Signal(
            'transform_persistence_default_changed')

    @classmethod
    def instance(cls) -> 'PipelineSettings':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def transform_persistence_default(self) -> TransformPersistenceDefault:
        return self._transform_persistence_default

    @transform_persistence_default.setter
    def transform_persistence_default(self,
                                      value: TransformPersistenceDefault):
        if value == self._transform_persistence_default:
            return
        self._transform_persistence_default = value
        self.transform_persistence_default_changed.emit(value)


# ---- disk cache ----------------------------------------------------------

def port_cache_dir() -> str:
    """Directory for OnDisk port cache files: $TOMVIZ_PORT_CACHE_DIR if
    set and non-empty, else the system temp dir."""
    env = os.environ.get('TOMVIZ_PORT_CACHE_DIR', '')
    return env if env else tempfile.gettempdir()


def create_cache_file() -> str:
    """Create (and keep) an empty uniquely-named cache file, returning
    its path. The caller owns deletion."""
    fd, path = tempfile.mkstemp(prefix='tomviz_port_', suffix='.portdata',
                                dir=port_cache_dir())
    os.close(fd)
    return path


def _pickle_write(port_data, path: str) -> bool:
    try:
        with open(path, 'wb') as f:
            pickle.dump((port_data.payload, port_data.port_type), f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception:
        logger.exception('Failed to serialize port data to %s', path)
        return False


def _pickle_read(path: str):
    from .node import PortData
    try:
        with open(path, 'rb') as f:
            payload, port_type = pickle.load(f)
        return PortData(payload, port_type)
    except Exception:
        logger.exception('Failed to deserialize port data from %s', path)
        return None


_write_port_data: Callable = _pickle_write
_read_port_data: Callable = _pickle_read


def set_port_data_serializer(writer: Callable, reader: Callable):
    """Install a custom OnDisk spill serializer.

    writer(port_data, path) -> bool; reader(path) -> PortData | None.
    Cache files are private to the process, so the format is free —
    the default is pickle."""
    global _write_port_data, _read_port_data
    _write_port_data = writer
    _read_port_data = reader


def write_port_data_to_file(port_data, path: str) -> bool:
    return _write_port_data(port_data, path)


def read_port_data_from_file(path: str):
    return _read_port_data(path)
