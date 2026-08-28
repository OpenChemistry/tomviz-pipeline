###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Unified pipeline runner — execute a state-file pipeline once, or
many times against a sequence of input file sets.

The pipeline graph is built once (state-file parse, JSON description
parse, link resolution, …) and reused across runs; between runs we
clear all output-port data and reset every node's state to ``New`` so
the executor re-executes the whole graph against the new inputs.

Output layout
-------------

When ``run`` produces exactly one run, leaf outputs land directly
under ``output_dir`` (no per-run subdirectory)::

    output_dir/<id>_<label>__<port>.<ext>

When ``run`` produces two or more runs, each gets its own
subdirectory under ``output_dir``::

    output_dir/<prefix>_<N>/<id>_<label>__<port>.<ext>

``<prefix>`` defaults to ``"run"`` (overridable via
``run_dir_prefix``) and ``<N>`` is zero-padded by the total run
count.

API
---

For state files run as-is (no overrides)::

    run(state_path, output_dir)

For multi-source pipelines, ``inputs`` is a dict mapping source node
id → path-or-glob-or-list::

    run(state_path, output_dir, inputs={
        1: sorted(glob.glob("data/run*.emd")),  # 5 paths
        3: "reference.emd",                      # broadcast to 5
    })

For single-source pipelines, ``inputs`` may simply be a path, glob, or
list — the unique ``ReaderSourceNode`` is auto-picked::

    run(state_path, output_dir, inputs="data/*.emd")
    run(state_path, output_dir, inputs=["a.emd", "b.emd", "c.emd"])

Per-source values are normalized as follows:
  - ``str`` containing glob magic → ``sorted(glob.glob(...))`` (must
    match at least one file)
  - ``str`` without glob magic → length-1 list (eligible for broadcast)
  - ``list[str]`` → each element is treated like a string above and the
    results are concatenated

After normalization, length-1 lists are broadcast to the longest
non-broadcast length ``N``. All non-broadcast lists must equal ``N``."""

from __future__ import annotations

import copy
import glob as _glob
import json
import logging
from pathlib import Path
from typing import Union

from tomviz_pipeline.core import (
    DefaultExecutor,
    InternalNodeExecutor,
    NodeState,
    Pipeline,
    SourceNode,
)
from tomviz_pipeline.leaf_writer import write_leaf_outputs
from tomviz_pipeline.nodes.sources.reader import ReaderSourceNode
from tomviz_pipeline.progress import make_progress
from tomviz_pipeline.state import (
    load_state,
    read_state_json,
    write_state_tvh5,
)


_OUTPUT_FORMATS = ('port', 'state', 'state+port')


logger = logging.getLogger('tomviz_pipeline')


PerSourceValue = Union[str, Path, list]
InputsArg = Union[PerSourceValue, dict, None]


def run(
    state_path,
    output_dir,
    inputs: InputsArg = None,
    run_dir_prefix: str = 'run',
    output_format: str = 'port',
    progress_method: str = 'tqdm',
    progress_path: str | None = None,
    node_state_file: str | None = None,
) -> list[Path]:
    """Run a pipeline once or many times.

    Parameters
    ----------
    state_path:
        Path to the schema-v2 ``.tvsm`` or ``.tvh5`` state file.
    output_dir:
        Base directory. With one run, outputs land directly under
        ``output_dir``; with 2+ runs, each lands under
        ``output_dir/<run_dir_prefix>_<N>/`` with N zero-padded.
    inputs:
        ``None`` (default) → execute the pipeline once with whatever
        ``fileNames`` are pinned in the state file.
        ``dict[node_id, str | list[str]]`` → multi-source batch, one
        entry per source to override; see module docstring for
        normalization and broadcast rules.
        ``str | list[str]`` → single-source convenience; the pipeline
        must contain exactly one ``ReaderSourceNode``.
    run_dir_prefix:
        Customize the per-run directory name when there are 2+ runs.
        Defaults to ``"run"``, producing ``run_0``, ``run_1``, … When
        there is exactly one run, this parameter is ignored and
        outputs go straight into ``output_dir``.
    output_format:
        ``'port'`` (default) — write each unconsumed leaf output port
        to a typed file (EMD/CSV/XYZ).
        ``'state'`` — write a single ``output_state.tvh5`` HDF5 file
        per run that bundles the schema-v2 state JSON together with
        the volume payloads of every populated, non-sink output port
        (tables and molecules are dropped, with a warning).
        ``'state+port'`` — both: tvh5 plus the typed per-port files.
    progress_method, progress_path:
        Forwarded to :func:`tomviz_pipeline.progress.make_progress`.
    node_state_file:
        Optional JSON file ``{"nodes": {"<id>": {...}}}`` carrying
        per-node runtime state bags. Installed as each node's
        ``user_state`` (surfaced to schema-v2 kernels as
        ``self.state``) before running; after all runs, the updated
        bags are written back to ``<output_dir>/node_state.json``.
        The external node executor uses this to round-trip node state
        across the process boundary.

    Returns
    -------
    list[Path]
        The per-run output directories, in run order.
    """
    if output_format not in _OUTPUT_FORMATS:
        raise ValueError(
            f"output_format must be one of {_OUTPUT_FORMATS}; got "
            f"{output_format!r}")
    write_ports = output_format in ('port', 'state+port')
    write_state = output_format in ('state', 'state+port')

    pipeline = load_state(state_path)
    _apply_node_states(pipeline, node_state_file)
    parameter_updates = _track_parameter_updates(pipeline)

    target_nodes, expanded, runs_count = _resolve_overrides(pipeline, inputs)

    # Always load the raw state JSON so each run directory is
    # self-contained: state.tvsm goes alongside the outputs whether or
    # not overrides were applied. Overrides patch fileNames on the
    # snapshot per run; with no overrides the snapshot is a verbatim
    # copy of the input state.
    raw_state = read_state_json(state_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # With exactly one run, outputs land directly under output_dir; no
    # run subdirectory is created and run_dir_prefix is unused. With
    # 2+ runs we fall back to the per-run subdir layout so files don't
    # collide.
    width = max(1, len(str(runs_count - 1))) if runs_count > 1 else 0
    written_dirs: list[Path] = []
    for index in range(runs_count):
        if runs_count == 1:
            run_dir = output_dir
            run_name = ''
        else:
            run_name = f'{run_dir_prefix}_{index:0{width}d}'
            run_dir = output_dir / run_name

        per_run_files = {sid: expanded[sid][index] for sid in target_nodes}
        log_target = run_name or str(run_dir)
        if per_run_files:
            logger.info(
                '[run] %s/%d %s -> %s', log_target or 'run', runs_count,
                ', '.join(f'{sid}={Path(p).name}'
                          for sid, p in per_run_files.items()),
                run_dir)
        else:
            logger.info('[run] %s/%d -> %s',
                        log_target or 'run', runs_count, run_dir)

        # Iteration 0 starts from the freshly loaded pipeline (load_state
        # already gave us clean state plus any tvh5-embedded payloads).
        # Subsequent iterations need a reset so override file_names get
        # re-read by their source nodes.
        if index > 0:
            _reset_pipeline(pipeline)
        for sid, path in per_run_files.items():
            target_nodes[sid].file_names = [str(path)]

        progress = make_progress(progress_method, progress_path)
        with progress as progress_obj:
            progress_obj.started()
            # Headless runs must never park on breakpoints saved in the
            # state file, so breakpoint honoring is disabled.
            ok = DefaultExecutor(
                pipeline, progress=progress_obj,
                honor_breakpoints=False).execute()
            progress_obj.finished()

        if not ok:
            logger.error('[run] %s failed', run_name)

        run_dir.mkdir(parents=True, exist_ok=True)

        # Patched JSON for this run. Used for both the input snapshot
        # (state.tvsm) and the output bundle (output_state.tvh5) so we
        # only deep-copy and patch once per run. With no overrides
        # this is a verbatim copy of the input state.
        run_state = _patch_state(raw_state, per_run_files)

        _write_state_snapshot(run_dir, run_state)
        if write_state:
            write_state_tvh5(run_dir / 'output_state.tvh5',
                             run_state, pipeline)
        if write_ports:
            write_leaf_outputs(pipeline, run_dir)

        written_dirs.append(run_dir)

    # Only when the caller opted into state round-tripping: nothing
    # reads the file otherwise.
    if node_state_file is not None:
        _write_node_states(pipeline, output_dir)
    _write_node_parameters(parameter_updates, output_dir)

    return written_dirs


def check_auto_execute(state_path, output_dir, node_id: int,
                       node_state_file: str | None = None) -> bool:
    """Evaluate one node's ``should_auto_execute`` hook without
    executing the pipeline.

    The question goes through the node's executor (as an application's
    periodic poll would), so a node carrying an ``executor`` block is
    asked in its own environment. Writes
    ``<output_dir>/auto_execute.json`` (``{"shouldExecute": bool}``)
    with the verdict, and ``<output_dir>/node_state.json`` with every
    node's updated ``user_state`` bag so the caller can round-trip
    state mutations the hook made (plus ``node_parameters.json`` when
    the hook changed parameters, see ``_write_node_parameters``). An
    unknown node id answers ``False``."""
    pipeline = load_state(state_path)
    _apply_node_states(pipeline, node_state_file)
    parameter_updates = _track_parameter_updates(pipeline)

    should = False
    node = pipeline.node_by_id(node_id)
    if node is None:
        logger.error('check-auto-execute: no node with id %s', node_id)
    else:
        executor = node.node_executor or InternalNodeExecutor.instance()
        should = bool(executor.should_auto_execute(node))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_node_states(pipeline, out_dir)
    _write_node_parameters(parameter_updates, out_dir)
    with open(out_dir / 'auto_execute.json', 'w', encoding='utf-8') as f:
        json.dump({'shouldExecute': should}, f)
    return should


def _apply_node_states(pipeline: Pipeline, node_state_file) -> None:
    """Read a ``{"nodes": {"<id>": {...}}}`` JSON file and install each
    entry as the matching node's runtime ``user_state`` bag."""
    if not node_state_file:
        return
    try:
        with open(node_state_file, encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.exception('Failed to read node-state file %s',
                         node_state_file)
        return
    for sid, state in (data.get('nodes') or {}).items():
        try:
            node = pipeline.node_by_id(int(sid))
        except (TypeError, ValueError):
            continue
        if node is not None and isinstance(state, dict):
            node.user_state = dict(state)


def _write_node_states(pipeline: Pipeline, out_dir) -> None:
    """Write every node's non-empty ``user_state`` to
    ``<out_dir>/node_state.json`` (the shape ``_apply_node_states``
    reads). A bag that isn't JSON-serializable is dropped with an
    error rather than failing the run."""
    states = {}
    for node in pipeline.nodes:
        state = getattr(node, 'user_state', None)
        if not state:
            continue
        try:
            # allow_nan=False: NaN/inf would serialize as non-strict
            # JSON that the parent's RFC-strict parser rejects wholesale.
            json.dumps(state, allow_nan=False)
        except (TypeError, ValueError):
            logger.error(
                'Node %s user state is not JSON-serializable; dropping '
                'it from node_state.json. Keep self.state values to '
                'finite bool/int/float/str, lists, and dicts.', node.id)
            continue
        states[str(node.id)] = state
    with open(Path(out_dir) / 'node_state.json', 'w', encoding='utf-8') as f:
        json.dump({'nodes': states}, f)


def _track_parameter_updates(pipeline: Pipeline) -> dict:
    """Return a ``{node_id: {name: value}}`` dict that accumulates every
    ``parameters_updated`` emission from the pipeline's nodes — the
    parameter changes kernels make through ``self.set_parameter``. Later
    updates overwrite earlier ones, so across a multi-run batch the
    dict holds the final values."""
    updates: dict[int, dict] = {}

    def on_updated(node, changed):
        updates.setdefault(node.id, {}).update(changed)

    for node in pipeline.nodes:
        node.parameters_updated.connect(on_updated)
    return updates


def _write_node_parameters(updates: dict, out_dir) -> None:
    """Write the parameter changes kernels made (see
    ``_track_parameter_updates``) to ``<out_dir>/node_parameters.json``
    in the ``{"nodes": {"<id>": {name: value}}}`` shape the external
    node executor reads back. Written only when something changed, so
    parents that don't know the file never see it and runs that don't
    use the feature leave no trace."""
    if not updates:
        return
    nodes = {}
    for node_id, changed in updates.items():
        try:
            json.dumps(changed, allow_nan=False)
        except (TypeError, ValueError):
            # Defensive: set_parameter coerces to JSON types, but any
            # host code may call apply_parameter_updates directly.
            logger.error(
                'Node %s parameter updates are not JSON-serializable; '
                'dropping them from node_parameters.json.', node_id)
            continue
        nodes[str(node_id)] = changed
    if not nodes:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'node_parameters.json', 'w', encoding='utf-8') as f:
        json.dump({'nodes': nodes}, f)


def _patch_state(raw_state: dict, per_run_files: dict) -> dict:
    """Deep-copy ``raw_state`` and replace ``fileNames`` on every
    source node listed in ``per_run_files``. Returns the patched
    snapshot."""
    snapshot = copy.deepcopy(raw_state)
    nodes = snapshot.get('pipeline', {}).get('nodes') or []
    for entry in nodes:
        node_id = entry.get('id')
        if node_id in per_run_files:
            entry['fileNames'] = [str(per_run_files[node_id])]
    return snapshot


def _write_state_snapshot(run_dir: Path, run_state: dict) -> None:
    """Write the patched state JSON to ``run_dir/state.tvsm``. The
    snapshot is itself a valid input to :func:`run`."""
    target = run_dir / 'state.tvsm'
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(run_state, f, indent=2)


# ---- normalization --------------------------------------------------


def _resolve_overrides(pipeline, inputs):
    """Return ``(target_nodes, expanded, runs_count)``.

    * ``target_nodes`` — dict[node_id, ReaderSourceNode] of sources we
      will mutate per run. Empty when ``inputs is None``.
    * ``expanded`` — dict[node_id, list[str]] aligned to ``runs_count``
      after broadcasting. Empty when ``inputs is None``.
    * ``runs_count`` — number of runs to execute.
    """
    if inputs is None:
        return {}, {}, 1

    by_source = _normalize_inputs_arg(pipeline, inputs)
    expanded = {sid: _expand_value(v) for sid, v in by_source.items()}
    runs_count = _resolve_run_count(expanded)

    target_nodes: dict[int, ReaderSourceNode] = {}
    for sid in sorted(expanded):
        node = pipeline.node_by_id(sid)
        if node is None:
            raise ValueError(f'No node with id {sid} in pipeline')
        if not isinstance(node, ReaderSourceNode):
            raise ValueError(
                f'Node {sid} is a {type(node).__name__}, not a '
                f'ReaderSourceNode; cannot override its file paths.')
        target_nodes[sid] = node
        if len(expanded[sid]) == 1 and runs_count > 1:
            expanded[sid] = expanded[sid] * runs_count

    return target_nodes, expanded, runs_count


def _normalize_inputs_arg(pipeline: Pipeline, inputs) -> dict:
    """Coerce the user's ``inputs`` parameter to ``dict[int, value]``,
    where ``value`` is the user's per-source ``str | list[str]``."""
    if isinstance(inputs, dict):
        if not inputs:
            raise ValueError('inputs dict must not be empty')
        for k in inputs:
            if not isinstance(k, int):
                raise TypeError(
                    f'inputs dict keys must be node ids (int); '
                    f'got {type(k).__name__} {k!r}')
        return dict(inputs)

    if isinstance(inputs, (str, Path, list, tuple)):
        readers = [n for n in pipeline.nodes
                   if isinstance(n, ReaderSourceNode)]
        if len(readers) != 1:
            raise ValueError(
                f'Single-source convenience inputs require exactly one '
                f'ReaderSourceNode in the pipeline; found {len(readers)}.'
                f' Pass a dict {{node_id: paths}} for multi-source batch.')
        return {readers[0].id: inputs}

    raise TypeError(
        f'inputs must be a dict, str, Path, list, or None; '
        f'got {type(inputs).__name__}')


def _expand_value(value) -> list[str]:
    """Normalize a per-source value to a concrete list of file paths.
    Glob magic gets expanded sorted. A bare list stays as-is, but its
    elements are themselves passed through glob expansion so a list
    can mix literal paths and glob patterns freely."""
    if isinstance(value, (str, Path)):
        return _expand_one(str(value))

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError('Per-source path list must not be empty')
        out: list[str] = []
        for item in value:
            if not isinstance(item, (str, Path)):
                raise TypeError(
                    f'List entries must be str or Path; got '
                    f'{type(item).__name__}')
            out.extend(_expand_one(str(item)))
        return out

    raise TypeError(
        f'Per-source value must be str, Path, or list; got '
        f'{type(value).__name__}')


def _expand_one(item: str) -> list[str]:
    if _glob.has_magic(item):
        matched = sorted(_glob.glob(item))
        if not matched:
            raise ValueError(f'Glob {item!r} matched no files')
        return matched
    if not Path(item).is_file():
        raise ValueError(f'Path {item!r} does not exist or is not a file')
    return [item]


def _resolve_run_count(expanded: dict) -> int:
    """Pick the run count N: the longest non-length-1 list. Lists with
    length 1 are eligible for broadcast. All lists with length > 1 must
    agree."""
    long_lists = {sid: len(v) for sid, v in expanded.items() if len(v) > 1}
    if not long_lists:
        return 1
    counts = set(long_lists.values())
    if len(counts) > 1:
        details = ', '.join(f'{sid}={n}' for sid, n in
                            sorted(long_lists.items()))
        raise ValueError(
            f'Inconsistent input list lengths across sources: {details}')
    return next(iter(counts))


def _reset_pipeline(pipeline: Pipeline) -> None:
    """Clear output-port payloads and reset state on every node so the
    executor will re-run the whole graph on the next call.

    Source nodes whose data has no replay mechanism (e.g. source.generic
    populated from tvh5 dataRefs at load time — no file_names, no
    execute() that produces data) are skipped: wiping their outputs
    would leave them stuck on subsequent iterations. ReaderSourceNode
    (re-reads file_names on each execute) and transforms (re-execute
    from upstream) reset normally.
    """
    for node in pipeline.nodes:
        if isinstance(node, SourceNode) and not getattr(
                node, 'file_names', None):
            continue
        node.state = NodeState.New
        for port in node.output_ports():
            port.set_data(None)
