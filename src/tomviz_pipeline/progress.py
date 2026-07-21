###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Progress reporters for the pipeline CLI. Three modes are supported, all
preserving the JSON wire format used by the legacy CLI so the future C++
external pipeline executor can consume them unchanged:

  - tqdm: console progress bar (interactive use)
  - socket: connect to a Unix domain socket and write JSON lines
  - files: write each JSON message to a numbered file in a directory
"""

from __future__ import annotations

import abc
import json
import os
import socket
import stat

from tqdm import tqdm


class ProgressBase(object):
    def started(self, op=None):
        self._operator_index = op

    def finished(self, op=None):
        pass

    def control_channel(self):
        """Return a ControlChannel for the parent → child direction,
        or None if this progress mode has no return path (e.g. tqdm).
        OperatorWrapper polls the channel lazily to detect cancel /
        complete requests."""
        return None


class ControlChannel(object):
    """Subprocess-side reader for control messages from the parent.
    Poll() is called from OperatorWrapper getters; implementations
    update wrapper._canceled / wrapper._completed in place based on
    whatever signals are pending in their transport. Cheap enough to
    invoke per-getter — operators check these flags inside loops."""

    def poll(self, wrapper):
        raise NotImplementedError


class FilesControlChannel(ControlChannel):
    """Watches for cancel.flag / complete.flag inside the progress
    directory. Mirrors FilesProgressReader.sendSignal on the parent
    side."""

    def __init__(self, progress_dir):
        self._cancel_flag = os.path.join(progress_dir, 'cancel.flag')
        self._complete_flag = os.path.join(progress_dir, 'complete.flag')

    def poll(self, wrapper):
        if not wrapper._canceled and os.path.exists(self._cancel_flag):
            wrapper._canceled = True
        if not wrapper._completed and os.path.exists(self._complete_flag):
            wrapper._completed = True


class SocketControlChannel(ControlChannel):
    """Reads JSON-line control messages from the duplex socket the
    progress channel is already using. Mirrors LocalSocketProgress
    Reader.sendSignal on the parent side."""

    def __init__(self, sock):
        self._sock = sock
        self._sock.setblocking(False)
        self._buffer = b''

    def poll(self, wrapper):
        try:
            while True:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                self._buffer += chunk
        except (BlockingIOError, InterruptedError):
            pass
        except OSError:
            # Socket closed or some other I/O error — give up silently;
            # the parent's exit will tear us down anyway.
            return

        while b'\n' in self._buffer:
            line, _, rest = self._buffer.partition(b'\n')
            self._buffer = rest
            if not line:
                continue
            try:
                msg = json.loads(line.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            t = msg.get('type')
            if t == 'cancel':
                wrapper._canceled = True
            elif t == 'complete':
                wrapper._completed = True


class TqdmProgress(ProgressBase):
    def __init__(self):
        # Initialize maximum/value to 0 (matching the in-app
        # _operator_wrapper defaults) so operators can safely use
        # ``self.progress.value += 1`` before any explicit assignment.
        self._maximum = 0
        self._value = 0
        self._message = None
        self._progress_bar = None

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, value):
        self._progress_bar = tqdm(total=value)
        self._maximum = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self._progress_bar is not None:
            self._progress_bar.update(value - self._progress_bar.n)
        self._value = value

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, msg):
        if self._progress_bar is not None:
            self._progress_bar.set_description(msg)
        self._message = msg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._progress_bar is not None:
            self._progress_bar.close()
        return False

    def finished(self, op=None):
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None


class JsonProgress(ProgressBase, metaclass=abc.ABCMeta):
    """Base class for the socket and files progress modes. Subclasses
    implement write() (emit one JSON message)."""

    # Set by the transform wrapper before the operator runs so a bare
    # `self.progress.data = X` knows which output port to populate.
    _primary_port = None
    # Path of the most recent intermediate-update tvh5 file. Held so we
    # can delete it when the next update arrives, keeping disk usage
    # bounded.
    _last_intermediate_path = None
    _intermediate_seq = 0

    @abc.abstractmethod
    def write(self, data):
        ...

    def set_operator_index(self, index):
        self._operator_index = index

    def set_primary_port(self, name):
        """Configure the port name a bare-value `progress.data = X`
        update routes to. Per-node — call before invoking the
        operator."""
        self._primary_port = name

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, value):
        self.write({
            'type': 'progress.maximum',
            'operator': self._operator_index,
            'value': value,
        })
        self._maximum = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.write({
            'type': 'progress.step',
            'operator': self._operator_index,
            'value': value,
        })
        self._value = value

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, msg):
        self.write({
            'type': 'progress.message',
            'operator': self._operator_index,
            'value': msg,
        })
        self._message = msg

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        # Two accepted forms:
        #   self.progress.data = X            (single-payload, routes
        #                                      to the configured
        #                                      primary output port)
        #   self.progress.data = {name: X}    (explicit per-port map)
        if isinstance(value, dict):
            port_data = value
        else:
            if self._primary_port is None:
                # No port to route to — drop silently rather than raise
                # (operators shouldn't crash if previews aren't wired).
                return
            port_data = {self._primary_port: value}
        self.write_intermediate(port_data)
        self._data = value

    def write_intermediate(self, port_data: dict):
        """Write a multi-port intermediate update as a small `.tvh5`
        file in the temp working directory and emit a `progress.data`
        message carrying its basename. Deletes the previous
        intermediate file first so disk usage stays bounded."""
        if self._last_intermediate_path:
            try:
                os.unlink(self._last_intermediate_path)
            except OSError:
                pass
            self._last_intermediate_path = None
        path = self._write_intermediate_tvh5(port_data)
        self._last_intermediate_path = path
        self.write({
            'type': 'progress.data',
            'operator': self._operator_index,
            'value': os.path.basename(path),
        })

    def _write_intermediate_tvh5(self, port_data: dict) -> str:
        from tomviz_pipeline.core import (
            Pipeline, PortData, SourceNode,
        )
        from tomviz_pipeline.state import write_state_tvh5

        # Build a one-node pipeline mirroring the parent-side target,
        # populate its output ports with the intermediate payloads.
        # write_state_tvh5 walks the pipeline + state JSON together,
        # stamping dataRefs onto matching outputPort entries.
        pipeline = Pipeline()
        src = SourceNode()
        pipeline.add_node(src)
        node_id = self._operator_index if self._operator_index is not None \
            else 1
        pipeline.set_node_id(src, node_id)

        output_ports_json = {}
        for port_name, payload in port_data.items():
            port_type = _infer_port_type(payload)
            port = src.add_output(port_name, port_type)
            port.set_data(PortData(payload, port_type))
            output_ports_json[port_name] = {
                'type': port_type,
                'persistent': True,
            }

        state_json = {
            'schemaVersion': 2,
            'pipeline': {
                'nodes': [{
                    'id': node_id,
                    'type': 'source.generic',
                    'outputPorts': output_ports_json,
                }],
                'links': [],
            },
        }

        target_dir = os.path.dirname(self._path)
        path = os.path.join(
            target_dir,
            'intermediate-%d.tvh5' % self._intermediate_seq)
        self._intermediate_seq += 1
        write_state_tvh5(path, state_json, pipeline)
        return path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def started(self, op=None):
        super().started(op)
        msg = {'type': 'started'}
        if op is not None:
            msg['operator'] = op
        self.write(msg)

    def finished(self, op=None):
        super().started(op)
        msg = {'type': 'finished'}
        if op is not None:
            msg['operator'] = op
        self.write(msg)


def _infer_port_type(payload) -> str:
    """Classify an intermediate payload into the schema-v2 port type
    string that downstream consumers (Tvh5Format::populatePayloadData)
    use to dispatch the right reader."""
    from tomviz_pipeline.molecule import Molecule
    from tomviz_pipeline.table import Table

    if isinstance(payload, Table):
        return 'Table'
    if isinstance(payload, Molecule):
        return 'Molecule'
    if hasattr(payload, 'arrays') or hasattr(payload, 'active_scalars'):
        return 'ImageData'
    # Duck-typed vtk fallbacks for scripts that build vtk objects
    # directly — attribute checks so vtk is never imported here.
    if (hasattr(payload, 'GetNumberOfColumns')
            and hasattr(payload, 'GetNumberOfRows')):
        return 'Table'
    if (hasattr(payload, 'GetNumberOfAtoms')
            and hasattr(payload, 'GetAtomicNumberArray')):
        return 'Molecule'
    return 'ImageData'


class LocalSocketProgress(JsonProgress):
    """Connect to a Unix domain socket and write JSON-newline messages
    consumed by the C++ Qt-side QLocalServer."""

    def __init__(self, socket_path):
        # 0 default so ``self.progress.value += 1`` works before any
        # explicit assignment (mirrors the in-app wrapper defaults).
        self._maximum = 0
        self._value = 0
        self._message = None
        self._connection = None
        self._path = socket_path
        self._sequence_number = 0

        try:
            mode = os.stat(self._path).st_mode
            if stat.S_ISSOCK(mode):
                self._uds_connect()
            else:
                raise Exception('Invalid progress path type.')
        except OSError as e:
            raise Exception(
                f"Progress path doesn't exist: {self._path}") from e

    def _uds_connect(self):
        self._connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._connection.connect(self._path)

    def write(self, data):
        encoded = ('%s\n' % json.dumps(data)).encode('utf8')
        self._connection.sendall(encoded)

    def control_channel(self):
        if self._connection is None:
            return None
        return SocketControlChannel(self._connection)

    def __exit__(self, *exc):
        if self._connection is not None:
            self._connection.close()
        return False


class FilesProgress(JsonProgress):
    """Write each JSON message as a separate numbered file in a directory
    that is being watched by the consumer (legacy on Mac/Windows)."""

    def __init__(self, path):
        self._maximum = 0
        self._value = 0
        self._message = None
        self._path = path
        self._sequence_number = 0

    def write(self, data):
        file_path = os.path.join(self._path,
                                 'progress%d' % self._sequence_number)
        self._sequence_number += 1
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def control_channel(self):
        return FilesControlChannel(self._path)


def make_progress(method: str, path: str | None = None):
    if method == 'tqdm':
        return TqdmProgress()
    if method == 'socket':
        if not path:
            raise Exception("'socket' progress mode requires a socket path")
        return LocalSocketProgress(path)
    if method == 'files':
        if not path:
            raise Exception("'files' progress mode requires a directory path")
        return FilesProgress(path)
    raise Exception(f'Unrecognized progress method: {method}')
