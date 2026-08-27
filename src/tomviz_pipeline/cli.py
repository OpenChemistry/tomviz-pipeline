###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""tomviz-pipeline — execute a schema-v2 graph state file headlessly.

Sinks are ignored; the leaves of the graph (after sinks are stripped)
have their unconnected output ports written to disk via the registered
writers (EMD for ImageData, CSV for Tables, XYZ for Molecules, …). The
future C++ external pipeline executor will invoke this same entry
point with `--progress socket|files` so the in-app pipeline can
offload graph execution to a separate Python environment.

Output layout depends on the number of runs:

  * Exactly one run: outputs land directly under ``--output-dir``::

        <output-dir>/<id>_<label>__<port>.<ext>

  * Two or more runs: each lands in its own subdirectory::

        <output-dir>/<prefix>_<N>/<id>_<label>__<port>.<ext>

``<prefix>`` defaults to ``"run"`` (override with ``--run-prefix``)
and ``<N>`` is zero-padded by the total run count.

Run modes
---------

* No ``--input``: execute the state file as-is, one run::

    tomviz-pipeline -s state.tvsm -o out

* Bare ``--input`` (single source — exactly one ``--input``, no
  ``NODE_ID:`` prefix; pipeline must contain exactly one
  ReaderSourceNode)::

    tomviz-pipeline -s state.tvsm -o out --input a.emd
    tomviz-pipeline -s state.tvsm -o out --input 'data/*.emd'
    tomviz-pipeline -s state.tvsm -o out --input a.emd,b.emd,c.emd

* Prefixed ``--input`` (multi-source — every ``--input`` carries a
  ``NODE_ID:`` prefix). A length-1 list against another source's
  longer list broadcasts; non-broadcast lists must all match in
  length. Repeats for the same node id concatenate::

    tomviz-pipeline -s state.tvsm -o out \\
        --input '1:data/*.emd' --input '3:reference.emd'
    tomviz-pipeline -s state.tvsm -o out \\
        --input 1:a.emd --input 1:b.emd --input 3:ref.emd

Each ``--input`` value (after the optional prefix) is comma-separated;
each segment is either a literal path or a glob pattern. Mixing bare
and prefixed values, or supplying multiple bare values, is rejected."""

from __future__ import annotations

import glob as _glob
import logging
import re
import sys
from pathlib import Path

import click

from tomviz_pipeline import __version__
from tomviz_pipeline.runner import check_auto_execute, run


LOG_FORMAT = '[%(asctime)s] %(levelname)s: %(message)s'

logger = logging.getLogger('tomviz_pipeline')

_PREFIX_RE = re.compile(r'^(\d+):(.*)$')


def _configure_logging():
    if logger.hasHandlers():
        return
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)


@click.command()
@click.version_option(__version__, '--version', prog_name='tomviz-pipeline')
@click.option('-s', '--state-file', 'state_file',
              required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to the .tvsm or .tvh5 state file to execute.')
@click.option('-o', '--output-dir', 'output_dir',
              required=True, type=click.Path(file_okay=False),
              help='Directory under which per-run subdirectories are '
                   'created.')
@click.option('-p', '--progress', 'progress_method',
              type=click.Choice(['tqdm', 'socket', 'files']),
              default='tqdm', show_default=True,
              help='How to report operator progress.')
@click.option('-u', '--progress-path', 'progress_path',
              type=click.Path(), default=None,
              help='Socket path (socket mode) or directory (files mode).')
@click.option('-i', '--input', 'inputs', type=str, multiple=True,
              help='Input override(s). Format: [NODE_ID:]VALUE where '
                   'VALUE is one or more comma-separated paths or '
                   'globs. Without a NODE_ID prefix, the pipeline must '
                   'have exactly one ReaderSourceNode and only one '
                   'bare --input is permitted; pass multiple paths via '
                   'a comma-separated list or glob inside the value. '
                   'With NODE_ID prefixes, every --input must be '
                   'prefixed; repeats for the same node id concatenate.')
@click.option('--run-prefix', 'run_prefix', type=str, default='run',
              show_default=True,
              help='Prefix for per-run subdirectories when there are '
                   '2+ runs. The full name is "<prefix>_<N>" with N '
                   'zero-padded. Ignored for single-run invocations '
                   '(outputs go straight into --output-dir).')
@click.option('--output-format', 'output_format',
              type=click.Choice(['port', 'state', 'state+port']),
              default='port', show_default=True,
              help='What to write per run. "port" -> typed per-port '
                   'files (EMD/CSV/XYZ). "state" -> a single '
                   'output_state.tvh5 bundling the schema-v2 state '
                   'with embedded volume payloads. "state+port" -> '
                   'both.')
@click.option('--node-state', 'node_state_file',
              type=click.Path(exists=True, dir_okay=False), default=None,
              help='JSON file carrying per-node runtime state '
                   '({"nodes": {"<id>": {...}}}). Installed as each '
                   'node\'s state bag (self.state in kernels) before '
                   'running; the updated bags are written back to '
                   '<output-dir>/node_state.json.')
@click.option('--check-auto-execute', 'check_node_id', type=int,
              default=None, metavar='NODE_ID',
              help='Do not execute the pipeline. Instead evaluate node '
                   'NODE_ID\'s should_auto_execute hook and write the '
                   'verdict to <output-dir>/auto_execute.json (plus '
                   'the updated state bags to node_state.json).')
def main(state_file, output_dir, progress_method, progress_path,
         inputs, run_prefix, output_format, node_state_file,
         check_node_id):
    """Execute a tomviz pipeline state file headlessly."""
    _configure_logging()

    state_path = Path(state_file)
    out_dir = Path(output_dir)

    if check_node_id is not None:
        try:
            should = check_auto_execute(
                state_path, out_dir, check_node_id,
                node_state_file=node_state_file)
        except (ValueError, TypeError) as e:
            click.echo(f'Error: {e}', err=True)
            sys.exit(2)
        logger.info('should_auto_execute(node %d) -> %s',
                    check_node_id, should)
        return

    inputs_arg = _build_inputs_arg(list(inputs)) if inputs else None

    try:
        run_dirs = run(
            state_path=state_path,
            output_dir=out_dir,
            inputs=inputs_arg,
            run_dir_prefix=run_prefix,
            output_format=output_format,
            progress_method=progress_method,
            progress_path=progress_path,
            node_state_file=node_state_file,
        )
    except (ValueError, TypeError) as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(2)

    logger.info('Wrote %d run directory(ies) under %s',
                len(run_dirs), out_dir)


def _build_inputs_arg(raw_inputs: list[str]):
    """Parse the raw ``--input`` strings into the value to forward to
    :func:`tomviz_pipeline.run`. The result is either a flat list of
    paths (bare mode -> single-source convenience) or a
    ``dict[node_id, list[paths]]`` (prefixed mode)."""
    parsed = [_parse_input(v) for v in raw_inputs]
    bare = [paths for sid, paths in parsed if sid is None]
    prefixed = [(sid, paths) for sid, paths in parsed if sid is not None]

    if bare and prefixed:
        raise click.UsageError(
            'Cannot mix bare and "NODE_ID:" prefixed --input values; '
            'pick one mode.')

    if bare:
        if len(bare) > 1:
            raise click.UsageError(
                'Only one bare --input is allowed (single-source '
                'mode). Use comma-separated paths or a glob inside '
                'the value to specify multiple inputs, or use the '
                '"NODE_ID:" prefix for multi-source batch.')
        # Single-source convenience: pass the path list straight
        # through; run() will auto-pick the unique reader.
        return bare[0]

    by_id: dict[int, list[str]] = {}
    for sid, paths in prefixed:
        by_id.setdefault(sid, []).extend(paths)
    return by_id


def _parse_input(value: str):
    """Parse a single ``--input`` value into ``(node_id, paths)``.

    ``node_id`` is ``None`` when the value has no ``\\d+:`` prefix.
    Comma-separated segments inside the value are each globbed/
    validated, then concatenated into the returned ``paths`` list."""
    m = _PREFIX_RE.match(value)
    if m:
        node_id: int | None = int(m.group(1))
        rest = m.group(2)
    else:
        node_id = None
        rest = value

    if not rest:
        raise click.BadParameter(
            f'--input value {value!r} has no path part')

    paths: list[str] = []
    for segment in rest.split(','):
        segment = segment.strip()
        if not segment:
            raise click.BadParameter(
                f'--input value {value!r}: empty segment')
        if _glob.has_magic(segment):
            matched = sorted(_glob.glob(segment))
            if not matched:
                raise click.BadParameter(
                    f'--input pattern {segment!r} matched no files')
            paths.extend(matched)
        else:
            if not Path(segment).is_file():
                raise click.BadParameter(
                    f'--input path {segment!r} does not exist '
                    f'or is not a file')
            paths.append(segment)
    return node_id, paths


if __name__ == '__main__':
    main()
