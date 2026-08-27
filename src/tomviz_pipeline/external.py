###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""ExternalNodeExecutor: run a single node in a subprocess under a
different Python environment. Parent-side port of the C++
ExternalNodeExecutor / ProgressReader pair.

Protocol (unchanged from C++ so either side can drive the same child):

1. Build a *shim* pipeline: a synthetic ``source.generic`` node (id 1)
   carrying the target's input-port payloads, plus a factory-fresh clone
   of the target node (id 2) with its ``executor`` block stripped so the
   child doesn't recurse. Write it as ``shim.tvh5`` in a temp dir.
2. Spawn ``<env>/bin/tomviz-pipeline -s shim.tvh5 -o out
   --output-format state -p socket|files -u <progress path>`` with
   PYTHONHOME/PYTHONPATH/TOMVIZ_APPLICATION scrubbed from the
   environment.
3. Stream JSON-line progress messages back (Unix-domain socket on POSIX,
   a polled directory of numbered files elsewhere) and forward them to
   the node's progress API. cancel/complete travel the other way on the
   same channel (a JSON line, or a ``<signal>.flag`` file) — soft
   signals the child observes at its next poll; the child is never
   killed, matching in-process cooperative-cancel semantics.
4. On a zero exit code, read ``out/output_state.tvh5`` and copy the
   clone's populated output-port payloads onto the real node.

The node's runtime state bag (``user_state``, ``self.state`` in kernels)
rides a JSON sidecar in both directions (``--node-state`` in,
``out/node_state.json`` back), and the periodic-execution poll
(``should_auto_execute``) spawns the same CLI in ``--check-auto-execute``
mode so the hook runs with the imports the child environment has.
Parameter values a kernel changes through ``self.set_parameter`` come
back in ``out/node_parameters.json`` (written by the child only when
something changed; same ``{"nodes": {"<id>": {...}}}`` shape) and are
installed on the real node through ``Node.apply_parameter_updates``.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

from .core import NodeExecutor, NodeExecutorFactory, SourceNode
from .core.events import Signal
from .core.state import pipeline_to_state_dict

logger = logging.getLogger('tomviz_pipeline')

_SHIM_SOURCE_ID = 1
_SHIM_TARGET_ID = 2

# Hard timeout for the should_auto_execute subprocess. Polls fire on a
# timer, so unlike execute() a hung check would stall auto-execution
# forever; kill it and answer False instead.
_CHECK_TIMEOUT_SECONDS = 10 * 60


def _use_socket_progress() -> bool:
    return os.name == 'posix'


def _write_node_state_file(node, node_id: int, path: Path) -> bool:
    """Write ``node``'s user-state bag as the ``--node-state`` sidecar
    the CLI reads: ``{"nodes": {"<node_id>": {...}}}``."""
    try:
        path.write_text(json.dumps(
            {'nodes': {str(node_id): dict(node.user_state)}}))
    except (OSError, TypeError, ValueError):
        logger.warning('ExternalNodeExecutor: failed to write node-state '
                       'file %s', path, exc_info=True)
        return False
    return True


def _apply_node_state_file(node, node_id: int, path: Path):
    """Install the entry for ``node_id`` from a ``node_state.json`` the
    CLI wrote back as ``node``'s user-state bag. A missing file or entry
    is a no-op (the run failed before writing it, or the env's package
    predates the sidecar)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    entry = (data.get('nodes') or {}).get(str(node_id))
    if isinstance(entry, dict):
        node.user_state = entry


def _apply_node_parameters_file(node, node_id: int, path: Path):
    """Install the parameter changes the child's kernel made — the entry
    for ``node_id`` in the ``node_parameters.json`` the CLI wrote — on
    ``node``, the quiet way (``Node.apply_parameter_updates``: no
    staleness, no re-execution). A missing file or entry is a no-op: the
    kernel changed nothing, or the env's package predates the file."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    entry = (data.get('nodes') or {}).get(str(node_id))
    if isinstance(entry, dict) and entry:
        node.apply_parameter_updates(entry)


class ProgressReader:
    """Parses the child's JSON-line progress stream and re-emits it as
    signals. Per-node messages carry an ``operator`` field (the node id
    in the *subprocess* pipeline); pipeline-level messages omit it.

    Signals:
      node_started(id), node_finished(id), node_error(id, message),
      node_progress_maximum(id, value), node_progress_step(id, value),
      node_progress_message(id, message), node_progress_data(id, path),
      pipeline_started(), pipeline_finished()
    """

    def __init__(self, path: str):
        self.path = str(path)
        self.node_started = Signal('node_started')
        self.node_finished = Signal('node_finished')
        self.node_error = Signal('node_error')
        self.node_progress_maximum = Signal('node_progress_maximum')
        self.node_progress_step = Signal('node_progress_step')
        self.node_progress_message = Signal('node_progress_message')
        self.node_progress_data = Signal('node_progress_data')
        self.pipeline_started = Signal('pipeline_started')
        self.pipeline_finished = Signal('pipeline_finished')

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def send_signal(self, signal_name: str):
        """Send a parent→child control signal ('cancel' or 'complete')."""
        raise NotImplementedError

    def handle_message(self, message: str):
        message = message.strip()
        if not message:
            return
        try:
            obj = json.loads(message)
        except ValueError:
            logger.error("Invalid progress message '%s'", message)
            return
        if not isinstance(obj, dict):
            logger.error("Invalid progress message '%s'", message)
            return
        msg_type = obj.get('type', '')

        if 'operator' in obj:
            node_id = int(obj['operator'])
            if msg_type == 'started':
                self.node_started.emit(node_id)
            elif msg_type == 'finished':
                self.node_finished.emit(node_id)
            elif msg_type == 'error':
                self.node_error.emit(node_id, obj.get('error', ''))
            elif msg_type == 'progress.maximum':
                self.node_progress_maximum.emit(node_id,
                                                int(obj.get('value', 0)))
            elif msg_type == 'progress.step':
                self.node_progress_step.emit(node_id,
                                             int(obj.get('value', 0)))
            elif msg_type == 'progress.message':
                self.node_progress_message.emit(node_id,
                                                obj.get('value', ''))
            elif msg_type == 'progress.data':
                self.node_progress_data.emit(node_id, obj.get('value', ''))
            else:
                logger.error('Unrecognized message type: %s', msg_type)
            return

        if msg_type == 'started':
            self.pipeline_started.emit()
        elif msg_type == 'finished':
            self.pipeline_finished.emit()
        else:
            logger.error('Unrecognized message type: %s', msg_type)


class FilesProgressReader(ProgressReader):
    """Watches a directory for numbered ``progress*`` files written by
    the child (FilesProgress), reading and deleting each in name order.
    Control signals are ``<name>.flag`` files dropped in the same
    directory — the ``.flag`` suffix keeps them out of the progress
    glob."""

    _POLL_INTERVAL = 0.2

    def __init__(self, path: str):
        super().__init__(path)
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, name='tomviz-pipeline-progress',
            daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        # Final sweep so messages written just before the child exited
        # are not lost.
        self._drain()

    def send_signal(self, signal_name: str):
        flag = Path(self.path) / f'{signal_name}.flag'
        try:
            flag.touch()
        except OSError:
            logger.warning('ProgressReader: failed to write signal flag %s',
                           flag)

    def _poll_loop(self):
        while not self._stop.wait(self._POLL_INTERVAL):
            self._drain()

    def _drain(self):
        try:
            names = sorted(p for p in os.listdir(self.path)
                           if p.startswith('progress'))
        except OSError:
            return
        for name in names:
            path = Path(self.path) / name
            try:
                message = path.read_text().splitlines()[0] \
                    if path.stat().st_size else ''
            except (OSError, IndexError):
                continue
            if not message:
                # Race with the writer — retry on the next tick.
                continue
            self.handle_message(message)
            try:
                path.unlink()
            except OSError:
                pass


class LocalSocketProgressReader(ProgressReader):
    """Unix-domain socket server; the child connects and writes JSON
    lines. Control signals are JSON lines written back on the same
    connection."""

    def __init__(self, path: str):
        super().__init__(path)
        self._server: Optional[socket.socket] = None
        self._connection: Optional[socket.socket] = None
        self._conn_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if os.path.exists(self.path):
            os.unlink(self.path)
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.path)
        self._server.listen(1)
        self._server.settimeout(0.2)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._serve, name='tomviz-pipeline-progress',
            daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._conn_lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except OSError:
                    pass
                self._connection = None
        if self._server is not None:
            self._server.close()
            self._server = None
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def send_signal(self, signal_name: str):
        with self._conn_lock:
            conn = self._connection
        if conn is None:
            logger.warning(
                'ProgressReader: socket not connected; signal %s dropped',
                signal_name)
            return
        line = json.dumps({'type': signal_name}) + '\n'
        try:
            conn.sendall(line.encode('utf-8'))
        except OSError:
            logger.warning('ProgressReader: failed to send signal %s',
                           signal_name)

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._conn_lock:
                self._connection = conn
            conn.settimeout(0.2)
            buffer = b''
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    self.handle_message(line.decode('utf-8'))
            return


class ExternalNodeExecutor(NodeExecutor):
    """Run a node's execute() in a subprocess using the tomviz-pipeline
    CLI installed in another Python environment (``env_path`` points at
    the environment root, e.g. a conda env or virtualenv).

    Serialized per node as ``{"type": "external", "envPath": ...}`` —
    identical to the C++ implementation, so state files round-trip.

    Signals:
      intermediate_data(node, {port_name: PortData}) — live preview
        payloads decoded from the child's progress.data messages. The
        node's real output ports are only touched by the final result.
    """

    type_name = 'external'

    def __init__(self, env_path: str = ''):
        self.env_path = env_path
        self.intermediate_data = Signal('intermediate_data')
        self._reader: Optional[ProgressReader] = None
        self._outputs_finalized = threading.Event()

    # ---- serialization ---------------------------------------------------

    def serialize(self) -> dict:
        return {'type': self.type_name, 'envPath': self.env_path}

    def deserialize(self, data: dict) -> bool:
        self.env_path = data.get('envPath', '')
        return True

    # ---- helpers ---------------------------------------------------------

    def find_cli_executable(self) -> Optional[str]:
        if not self.env_path:
            return None
        if os.name == 'nt':
            candidate = Path(self.env_path) / 'Scripts' / \
                'tomviz-pipeline.exe'
        else:
            candidate = Path(self.env_path) / 'bin' / 'tomviz-pipeline'
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        return None

    def _make_clone(self, node):
        """A factory-fresh clone of ``node`` for the child pipeline, or
        ``None`` (after logging) when the node can't cross the process
        boundary."""
        from .core import NodeFactory

        if not node.type_name:
            logger.warning(
                'ExternalNodeExecutor: target node has no registered '
                'type; cannot externalize.')
            return None
        clone = NodeFactory.create(node.type_name)
        if clone is None:
            logger.warning(
                "ExternalNodeExecutor: NodeFactory could not create '%s'",
                node.type_name)
            return None
        clone_entry = node.serialize()
        # A python-hosted kernel crosses the process boundary only as a
        # script (serialize() re-expresses a bound class via source
        # capture). If that failed, stop here — before spawning —
        # instead of letting the child die with a confusing error.
        from tomviz_pipeline.nodes.python_node import PythonNode
        if isinstance(node, PythonNode) and not clone_entry.get('script'):
            logger.warning(
                "ExternalNodeExecutor: node '%s' has no script and its "
                'kernel class (if any) has no retrievable source; '
                'provide the kernel as a script string or .py path to '
                'run it externally.', node.label or node.type_name)
            return None
        # Strip the executor block so the subprocess doesn't recurse into
        # another ExternalNodeExecutor. The legacy `tomviz_pipeline_env`
        # description key gets the same treatment — the child would
        # otherwise re-promote it and spawn a grandchild. Auto-execute
        # timers are the parent's business, not the child's.
        clone_entry.pop('executor', None)
        clone_entry.pop('autoExecute', None)
        description = clone_entry.get('description')
        if description and 'tomviz_pipeline_env' in description:
            try:
                desc_obj = json.loads(description)
                desc_obj.pop('tomviz_pipeline_env', None)
                clone_entry['description'] = json.dumps(desc_obj)
            except ValueError:
                pass
        clone.deserialize(clone_entry)
        clone.node_executor = None
        return clone

    def _write_shim_tvh5(self, node, tmp_dir: Path) -> Optional[str]:
        from .core import Pipeline
        from .state import write_state_tvh5

        shim = Pipeline()
        shim_source = SourceNode()
        shim_source.label = 'ExternalShimSource'
        shim.add_node(shim_source)
        shim.set_node_id(shim_source, _SHIM_SOURCE_ID)

        for port in node.input_ports():
            if port.link is None or not port.has_data():
                logger.warning(
                    "ExternalNodeExecutor: input port '%s' has no data; "
                    'cannot run externally.', port.name)
                return None
            payload = port.data()
            out = shim_source.add_output(port.name, payload.port_type)
            out.set_data(payload)

        clone = self._make_clone(node)
        if clone is None:
            return None
        shim.add_node(clone)
        shim.set_node_id(clone, _SHIM_TARGET_ID)

        for port in node.input_ports():
            src = shim_source.output_port(port.name)
            dst = clone.input_port(port.name)
            if src is None or dst is None:
                logger.warning(
                    "ExternalNodeExecutor: missing port pairing for '%s'",
                    port.name)
                return None
            shim.create_link(src, dst)

        shim_path = tmp_dir / 'shim.tvh5'
        state_json = pipeline_to_state_dict(shim)
        write_state_tvh5(shim_path, state_json, shim)
        return str(shim_path)

    def _write_check_state(self, node, tmp_dir: Path) -> Optional[str]:
        """Minimal state file for the auto-execute poll: just a clone of
        ``node`` (no inputs, no payloads — the hook receives neither),
        so polling never serializes input volumes to disk."""
        from .core import Pipeline

        clone = self._make_clone(node)
        if clone is None:
            return None
        check = Pipeline()
        check.add_node(clone)
        check.set_node_id(clone, _SHIM_TARGET_ID)

        check_path = tmp_dir / 'check.tvsm'
        check_path.write_text(json.dumps(pipeline_to_state_dict(check)))
        return str(check_path)

    def _decode_outputs(self, node, tvh5_path) -> dict:
        """Read the clone's populated output ports out of a child-written
        tvh5 and return {port_name: PortData} limited to ports that also
        exist on the real node."""
        from .state import load_state

        if not Path(tvh5_path).is_file():
            return {}
        try:
            result = load_state(tvh5_path)
        except Exception:
            logger.exception('ExternalNodeExecutor: failed to read %s',
                             tvh5_path)
            return {}
        clone = result.node_by_id(_SHIM_TARGET_ID)
        if clone is None:
            return {}
        outputs = {}
        for port in clone.output_ports():
            if port.has_data() and node.output_port(port.name) is not None:
                outputs[port.name] = port.data()
        return outputs

    # ---- NodeExecutor interface -------------------------------------------

    def execute(self, node) -> bool:
        from .nodes import register_builtins
        register_builtins()

        node.reset_progress()
        self._outputs_finalized.clear()

        cli = self.find_cli_executable()
        if cli is None:
            logger.warning(
                'ExternalNodeExecutor: tomviz-pipeline not found in env %s',
                self.env_path)
            return False

        with tempfile.TemporaryDirectory(
                prefix='tomviz-pipeline-') as tmp_name:
            tmp_dir = Path(tmp_name)
            return self._execute_in_dir(node, cli, tmp_dir)

    def _execute_in_dir(self, node, cli: str, tmp_dir: Path) -> bool:
        shim_path = self._write_shim_tvh5(node, tmp_dir)
        if shim_path is None:
            return False

        out_dir = tmp_dir / 'out'
        out_dir.mkdir(parents=True, exist_ok=True)
        output_state_path = out_dir / 'output_state.tvh5'

        if _use_socket_progress():
            progress_path = tmp_dir / 'progress.sock'
            reader: ProgressReader = LocalSocketProgressReader(
                str(progress_path))
            progress_method = 'socket'
        else:
            progress_path = tmp_dir / 'progress'
            reader = FilesProgressReader(str(progress_path))
            progress_method = 'files'

        target_id = _SHIM_TARGET_ID

        def on_maximum(node_id, value):
            if node_id == target_id:
                node.set_total_progress_steps(value)

        def on_step(node_id, value):
            if node_id == target_id:
                node.set_progress_step(value)

        def on_message(node_id, message):
            if node_id == target_id:
                node.set_progress_message(message)

        def on_error(node_id, error):
            if node_id == target_id:
                logger.warning(
                    'ExternalNodeExecutor: node %d reported error: %s',
                    node_id, error)

        def on_data(node_id, filename):
            if node_id != target_id or not filename:
                return
            # Drop tail-end intermediates delivered after the final
            # result is installed.
            if self._outputs_finalized.is_set():
                return
            intermediate_path = tmp_dir / filename
            updates = self._decode_outputs(node, intermediate_path)
            if updates:
                self.intermediate_data.emit(node, updates)

        reader.node_progress_maximum.connect(on_maximum)
        reader.node_progress_step.connect(on_step)
        reader.node_progress_message.connect(on_message)
        reader.node_error.connect(on_error)
        reader.node_progress_data.connect(on_data)

        reader.start()
        self._reader = reader

        env = dict(os.environ)
        env.pop('TOMVIZ_APPLICATION', None)
        env.pop('PYTHONHOME', None)
        env.pop('PYTHONPATH', None)
        env['PYTHONUNBUFFERED'] = 'ON'

        args = [
            cli,
            '-s', shim_path,
            '-o', str(out_dir),
            '--output-format', 'state',
            '-p', progress_method,
            '-u', str(progress_path),
        ]

        # Round-trip the node's user-state bag through the child. Only
        # when the node actually uses the feature — the flag is unknown
        # to older packages, and passing it unconditionally would break
        # every existing external env.
        pass_node_state = node.auto_execute_enabled or bool(node.user_state)
        if pass_node_state:
            node_state_in = tmp_dir / 'node_state_in.json'
            if _write_node_state_file(node, target_id, node_state_in):
                args += ['--node-state', str(node_state_in)]

        try:
            process = subprocess.Popen(
                args, env=env, cwd=str(tmp_dir),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            logger.exception('ExternalNodeExecutor: failed to start %s', cli)
            reader.stop()
            self._reader = None
            return False

        stdout, stderr = process.communicate()
        exit_code = process.returncode

        reader.stop()
        self._reader = None

        if stdout:
            logger.debug('%s', stdout.decode('utf-8', 'replace').strip())
        if stderr:
            logger.debug('%s', stderr.decode('utf-8', 'replace').strip())

        # Pick up state mutations regardless of how the run ended: the
        # CLI writes node_state.json after its runs, so a failed node
        # execution can still have recorded state.
        if pass_node_state:
            _apply_node_state_file(node, target_id,
                                   out_dir / 'node_state.json')
        _apply_node_parameters_file(node, target_id,
                                    out_dir / 'node_parameters.json')

        if node.is_cancel_requested():
            return False

        if exit_code != 0:
            logger.warning(
                'ExternalNodeExecutor: subprocess failed (exit=%d).\n%s',
                exit_code, stderr.decode('utf-8', 'replace').strip())
            return False

        outputs = self._decode_outputs(node, output_state_path)
        self._outputs_finalized.set()
        if not outputs:
            logger.warning(
                'ExternalNodeExecutor: no output ports were populated '
                'from %s', output_state_path)
            return False
        for name, data in outputs.items():
            node.output_port(name).set_data(data)
        return True

    def should_auto_execute(self, node) -> bool:
        """Auto-execute poll: spawn ``tomviz-pipeline --check-auto-execute``
        in the configured env so the node's should_auto_execute hook runs
        with the same imports execute() would have. Blocks (bounded by a
        kill timeout). Any failure — missing CLI, an env whose package
        predates the check mode, a timeout — answers False."""
        from .nodes import register_builtins
        register_builtins()

        cli = self.find_cli_executable()
        if cli is None:
            logger.warning(
                'ExternalNodeExecutor: cannot poll should_auto_execute — '
                'tomviz-pipeline not found in env %s', self.env_path)
            return False

        with tempfile.TemporaryDirectory(
                prefix='tomviz-pipeline-') as tmp_name:
            return self._check_in_dir(node, cli, Path(tmp_name))

    def _check_in_dir(self, node, cli: str, tmp_dir: Path) -> bool:
        state_path = self._write_check_state(node, tmp_dir)
        if state_path is None:
            return False

        target_id = _SHIM_TARGET_ID
        node_state_in = tmp_dir / 'node_state_in.json'
        if not _write_node_state_file(node, target_id, node_state_in):
            return False

        out_dir = tmp_dir / 'out'
        out_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env.pop('TOMVIZ_APPLICATION', None)
        env.pop('PYTHONHOME', None)
        env.pop('PYTHONPATH', None)
        env['PYTHONUNBUFFERED'] = 'ON'

        args = [
            cli,
            '-s', state_path,
            '-o', str(out_dir),
            '--check-auto-execute', str(target_id),
            '--node-state', str(node_state_in),
        ]

        try:
            completed = subprocess.run(
                args, env=env, cwd=str(tmp_dir),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=_CHECK_TIMEOUT_SECONDS)
        except OSError:
            logger.exception('ExternalNodeExecutor: failed to start %s', cli)
            return False
        except subprocess.TimeoutExpired:
            logger.warning('ExternalNodeExecutor: should_auto_execute poll '
                           'timed out; the subprocess was killed.')
            return False

        if completed.stderr:
            logger.debug('%s',
                         completed.stderr.decode('utf-8', 'replace').strip())

        # State mutations made by the hook count even when it answered
        # "no"; so do parameter changes.
        _apply_node_state_file(node, target_id, out_dir / 'node_state.json')
        _apply_node_parameters_file(node, target_id,
                                    out_dir / 'node_parameters.json')

        if completed.returncode != 0:
            logger.warning(
                'ExternalNodeExecutor: should_auto_execute poll failed '
                '(exit=%d). If the environment\'s tomviz-pipeline package '
                'predates --check-auto-execute, update it.\n%s',
                completed.returncode,
                completed.stderr.decode('utf-8', 'replace').strip())
            return False

        try:
            verdict = json.loads((out_dir / 'auto_execute.json').read_text())
        except (OSError, ValueError):
            logger.warning('ExternalNodeExecutor: should_auto_execute poll '
                           'wrote no auto_execute.json.')
            return False
        return bool(verdict.get('shouldExecute', False))

    def cancel(self, node):
        self._send_control_signal('cancel')

    def complete(self, node):
        self._send_control_signal('complete')

    def _send_control_signal(self, signal_name: str):
        # Soft signal — the child's operator polls and observes it at its
        # next checkpoint; the subprocess is never killed (matches the
        # in-process cooperative-cancel semantics).
        reader = self._reader
        if reader is not None:
            reader.send_signal(signal_name)


def register_external_executor():
    """Register the 'external' executor type so state files carrying
    per-node executor blocks round-trip. Idempotent."""
    NodeExecutorFactory.register(ExternalNodeExecutor.type_name,
                                 ExternalNodeExecutor)


def promote_description_env_path(node, entry: dict, env_path: str):
    """Compatibility shim for the pre-extraction `tomviz_pipeline_env`
    key: old operator JSON descriptions could request execution in a
    specific Python environment. Promote it to an ExternalNodeExecutor
    at deserialize time — unless the node entry carries an explicit
    `executor` block, which is the modern mechanism and always wins."""
    if not env_path:
        return
    if entry.get('executor'):
        return
    node.node_executor = ExternalNodeExecutor(env_path)
