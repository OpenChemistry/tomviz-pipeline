###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Tests for tomviz_pipeline.runner.run — execute a state-file pipeline
either with no overrides (one run) or against a sequence of input
file sets (single-source convenience or dict[node_id, paths]) — plus
the matching CLI surface (``-i [NODE_ID:]VALUE`` with
``--run-prefix``)."""

import json

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from tomviz_pipeline import Dataset, register_builtins, run
from tomviz_pipeline.cli import main as cli_main
from tomviz_pipeline.io.emd import _write_emd, load_dataset


@pytest.fixture(autouse=True)
def _builtins():
    register_builtins()


def _write_simple_emd(path, arr):
    ds = Dataset({'ImageScalars': np.asfortranarray(arr)}, 'ImageScalars')
    ds.spacing = (1.0, 1.0, 1.0)
    _write_emd(path, ds)


def _make_inputs(tmp_path, names_and_values):
    """Materialize one tiny EMD per name with the given fill value."""
    inputs = []
    for name, value in names_and_values:
        path = tmp_path / f'{name}.emd'
        _write_simple_emd(path, np.full((2, 2, 3), value, dtype=np.uint8))
        inputs.append(path)
    return inputs


def _single_source_state(tmp_path, placeholder_path):
    """A Reader → ConvertToFloat graph; the Reader gets its fileNames
    overridden per run."""
    state = {
        'schemaVersion': 2,
        'pipeline': {
            'nextNodeId': 3,
            'nodes': [
                {'id': 1, 'type': 'source.reader', 'label': 'Reader',
                 'fileNames': [str(placeholder_path)]},
                {'id': 2, 'type': 'transform.convertToFloat',
                 'label': 'CvtFloat'},
            ],
            'links': [
                {'from': {'node': 1, 'port': 'volume'},
                 'to':   {'node': 2, 'port': 'volume'}},
            ],
        },
    }
    state_path = tmp_path / 'state.tvsm'
    state_path.write_text(json.dumps(state))
    return state_path


def _two_source_state(tmp_path, placeholder_a, placeholder_b):
    """Reader1 → CvtA, Reader2 → CvtB. Both transforms are leaves."""
    state = {
        'schemaVersion': 2,
        'pipeline': {
            'nodes': [
                {'id': 1, 'type': 'source.reader', 'label': 'Reader1',
                 'fileNames': [str(placeholder_a)]},
                {'id': 2, 'type': 'transform.convertToFloat',
                 'label': 'CvtA'},
                {'id': 3, 'type': 'source.reader', 'label': 'Reader2',
                 'fileNames': [str(placeholder_b)]},
                {'id': 4, 'type': 'transform.convertToFloat',
                 'label': 'CvtB'},
            ],
            'links': [
                {'from': {'node': 1, 'port': 'volume'},
                 'to':   {'node': 2, 'port': 'volume'}},
                {'from': {'node': 3, 'port': 'volume'},
                 'to':   {'node': 4, 'port': 'volume'}},
            ],
        },
    }
    state_path = tmp_path / 'state.tvsm'
    state_path.write_text(json.dumps(state))
    return state_path


# ---- Python API: single-source convenience -----------------------------


def test_run_single_source_list_creates_run_dirs(tmp_path):
    inputs = _make_inputs(tmp_path,
                          [('alpha', 1), ('bravo', 5), ('charlie', 9)])
    state_path = _single_source_state(tmp_path, inputs[0])
    out_dir = tmp_path / 'out'

    run_dirs = run(state_path, out_dir,
                   inputs=[str(p) for p in inputs])

    assert [d.name for d in run_dirs] == ['run_0', 'run_1', 'run_2']
    for run_dir, expected in zip(run_dirs, [1.0, 5.0, 9.0]):
        emds = list(run_dir.glob('*.emd'))
        assert len(emds) == 1
        assert np.all(load_dataset(emds[0]).active_scalars == expected)


def test_run_single_source_glob_string(tmp_path):
    inputs = _make_inputs(tmp_path, [('a', 1), ('b', 2), ('c', 3)])
    state_path = _single_source_state(tmp_path, inputs[0])
    out_dir = tmp_path / 'out'

    run_dirs = run(state_path, out_dir, inputs=f'{tmp_path}/*.emd')

    # Glob matches sort alphabetically: a, b, c -> run_0, run_1, run_2.
    assert [d.name for d in run_dirs] == ['run_0', 'run_1', 'run_2']


def test_run_single_source_single_path_string(tmp_path):
    inputs = _make_inputs(tmp_path, [('only', 4)])
    state_path = _single_source_state(tmp_path, inputs[0])
    out_dir = tmp_path / 'out'

    run_dirs = run(state_path, out_dir, inputs=str(inputs[0]))

    # Single run → outputs go directly into out_dir (no subdir).
    assert run_dirs == [out_dir]
    assert list(out_dir.glob('*.emd'))


def test_run_single_source_run_naming_zero_pads(tmp_path):
    """Twelve runs should pad run names to width 2 (run_00..run_11)."""
    inputs = _make_inputs(tmp_path,
                          [(f'in{i:02d}', i + 1) for i in range(12)])
    state_path = _single_source_state(tmp_path, inputs[0])
    out_dir = tmp_path / 'out'

    run_dirs = run(state_path, out_dir,
                   inputs=[str(p) for p in inputs])

    assert run_dirs[0].name == 'run_00'
    assert run_dirs[-1].name == 'run_11'


def test_run_non_dict_with_multiple_readers_raises(tmp_path):
    inputs = _make_inputs(tmp_path, [('a', 1)])
    state_path = _two_source_state(tmp_path, inputs[0], inputs[0])
    with pytest.raises(ValueError, match='exactly one ReaderSourceNode'):
        run(state_path, tmp_path / 'out', inputs=[str(inputs[0])])


# ---- Python API: dict form ---------------------------------------------


def test_run_dict_multi_source(tmp_path):
    a1, a2, b1, b2 = _make_inputs(
        tmp_path, [('a1', 1), ('a2', 2), ('b1', 10), ('b2', 20)])
    state_path = _two_source_state(tmp_path, a1, b1)

    run_dirs = run(
        state_path,
        tmp_path / 'out',
        inputs={1: [str(a1), str(a2)], 3: [str(b1), str(b2)]},
    )
    assert [d.name for d in run_dirs] == ['run_0', 'run_1']

    for run_dir, expected_a, expected_b in (
            (run_dirs[0], 1.0, 10.0),
            (run_dirs[1], 2.0, 20.0)):
        emds = sorted(run_dir.glob('*.emd'))
        assert len(emds) == 2
        a_file = next(e for e in emds if e.name.startswith('2_'))
        b_file = next(e for e in emds if e.name.startswith('4_'))
        assert np.all(load_dataset(a_file).active_scalars == expected_a)
        assert np.all(load_dataset(b_file).active_scalars == expected_b)


def test_run_dict_broadcast_shorter_to_longer(tmp_path):
    """When one source has a length-1 list and another has N paths,
    the length-1 list broadcasts to N runs."""
    a1, a2, ref = _make_inputs(
        tmp_path, [('a1', 1), ('a2', 2), ('ref', 99)])
    state_path = _two_source_state(tmp_path, a1, ref)

    run_dirs = run(
        state_path,
        tmp_path / 'out',
        inputs={1: [str(a1), str(a2)], 3: str(ref)},   # source 3 broadcast
    )
    assert len(run_dirs) == 2
    for run_dir in run_dirs:
        emds = sorted(run_dir.glob('*.emd'))
        # CvtB output (node 4) should always equal 99 (the ref).
        b_file = next(e for e in emds if e.name.startswith('4_'))
        assert np.all(load_dataset(b_file).active_scalars == 99.0)


def test_run_dict_glob_in_value(tmp_path):
    """A glob string as a per-source value gets expanded."""
    _make_inputs(tmp_path,
                 [('a1', 1), ('a2', 2), ('b1', 10), ('b2', 20)])
    a1 = tmp_path / 'a1.emd'
    b1 = tmp_path / 'b1.emd'
    state_path = _two_source_state(tmp_path, a1, b1)

    run_dirs = run(
        state_path,
        tmp_path / 'out',
        inputs={1: f'{tmp_path}/a*.emd', 3: f'{tmp_path}/b*.emd'},
    )
    assert len(run_dirs) == 2


def test_run_dict_all_broadcast_yields_one_run(tmp_path):
    a, b = _make_inputs(tmp_path, [('a', 1), ('b', 10)])
    state_path = _two_source_state(tmp_path, a, b)

    out_dir = tmp_path / 'out'
    run_dirs = run(
        state_path,
        out_dir,
        inputs={1: str(a), 3: str(b)},
    )
    # All sources broadcast → 1 run → flat output.
    assert run_dirs == [out_dir]


def test_run_dict_inconsistent_lengths_raises(tmp_path):
    a1, a2, b1, b2, b3 = _make_inputs(
        tmp_path, [('a1', 1), ('a2', 2),
                   ('b1', 10), ('b2', 20), ('b3', 30)])
    state_path = _two_source_state(tmp_path, a1, b1)

    with pytest.raises(ValueError, match='Inconsistent input list lengths'):
        run(
            state_path,
            tmp_path / 'out',
            inputs={1: [str(a1), str(a2)],
                    3: [str(b1), str(b2), str(b3)]},
        )


def test_run_dict_unknown_node_id_raises(tmp_path):
    inputs = _make_inputs(tmp_path, [('a', 1)])
    state_path = _single_source_state(tmp_path, inputs[0])
    with pytest.raises(ValueError, match='No node with id 99'):
        run(state_path, tmp_path / 'out',
            inputs={99: str(inputs[0])})


def test_run_dict_non_reader_node_id_raises(tmp_path):
    inputs = _make_inputs(tmp_path, [('a', 1)])
    state_path = _single_source_state(tmp_path, inputs[0])
    with pytest.raises(ValueError, match='ReaderSourceNode'):
        # Node 2 is the ConvertToFloat transform.
        run(state_path, tmp_path / 'out',
            inputs={2: str(inputs[0])})


def test_run_dict_string_keys_raise(tmp_path):
    inputs = _make_inputs(tmp_path, [('a', 1)])
    state_path = _single_source_state(tmp_path, inputs[0])
    with pytest.raises(TypeError, match='node ids'):
        run(state_path, tmp_path / 'out',
            inputs={'1': str(inputs[0])})


def test_run_empty_dict_raises(tmp_path):
    inputs = _make_inputs(tmp_path, [('a', 1)])
    state_path = _single_source_state(tmp_path, inputs[0])
    with pytest.raises(ValueError, match='must not be empty'):
        run(state_path, tmp_path / 'out', inputs={})


def test_run_dict_glob_no_match_raises(tmp_path):
    inputs = _make_inputs(tmp_path, [('a', 1)])
    state_path = _single_source_state(tmp_path, inputs[0])
    with pytest.raises(ValueError, match='matched no files'):
        run(
            state_path,
            tmp_path / 'out',
            inputs={1: f'{tmp_path}/zzz_*.emd'},
        )


# ---- CLI: bare mode (single source) -------------------------------------


def test_cli_bare_input_single_path(tmp_path):
    inputs = _make_inputs(tmp_path, [('a', 3)])
    state_path = _single_source_state(tmp_path, inputs[0])
    out_dir = tmp_path / 'out'

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(out_dir),
        '-i', str(inputs[0]),
    ])
    assert result.exit_code == 0, result.output
    # Single run → flat output, no run subdir.
    assert list(out_dir.glob('*.emd'))
    assert not (out_dir / 'run_0').exists()


def test_cli_bare_input_comma_list(tmp_path):
    a, b, c = _make_inputs(tmp_path, [('a', 1), ('b', 2), ('c', 3)])
    state_path = _single_source_state(tmp_path, a)
    out_dir = tmp_path / 'out'

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(out_dir),
        '-i', f'{a},{b},{c}',
    ])
    assert result.exit_code == 0, result.output
    assert sorted(d.name for d in out_dir.iterdir()) == [
        'run_0', 'run_1', 'run_2']


def test_cli_bare_input_glob(tmp_path):
    a, b, c = _make_inputs(tmp_path, [('a', 1), ('b', 2), ('c', 3)])
    state_path = _single_source_state(tmp_path, a)
    out_dir = tmp_path / 'out'

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(out_dir),
        '-i', f'{tmp_path}/*.emd',
    ])
    assert result.exit_code == 0, result.output
    assert sorted(d.name for d in out_dir.iterdir()) == [
        'run_0', 'run_1', 'run_2']


def test_cli_multiple_bare_inputs_rejected(tmp_path):
    a, b = _make_inputs(tmp_path, [('a', 1), ('b', 2)])
    state_path = _single_source_state(tmp_path, a)

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(tmp_path / 'out'),
        '-i', str(a),
        '-i', str(b),
    ])
    assert result.exit_code != 0
    assert 'Only one bare --input is allowed' in result.output


def test_cli_bare_input_with_multi_reader_state_rejected(tmp_path):
    a, b = _make_inputs(tmp_path, [('a', 1), ('b', 1)])
    state_path = _two_source_state(tmp_path, a, b)

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(tmp_path / 'out'),
        '-i', str(a),
    ])
    # run_many raises ValueError; the CLI surfaces it as exit 2.
    assert result.exit_code != 0


# ---- CLI: prefixed mode (multi source) ----------------------------------


def test_cli_prefixed_multi_source(tmp_path):
    a1, a2, b1, b2 = _make_inputs(
        tmp_path, [('a1', 1), ('a2', 2), ('b1', 10), ('b2', 20)])
    state_path = _two_source_state(tmp_path, a1, b1)
    out_dir = tmp_path / 'out'

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(out_dir),
        '-i', f'1:{a1},{a2}',
        '-i', f'3:{b1},{b2}',
    ])
    assert result.exit_code == 0, result.output
    assert sorted(d.name for d in out_dir.iterdir()) == ['run_0', 'run_1']


def test_cli_prefixed_glob_with_broadcast(tmp_path):
    """Source 1 globs to several files, source 3 broadcasts a single
    reference. The result is one run per source-1 match."""
    a1, a2, a3, ref = _make_inputs(
        tmp_path, [('a1', 1), ('a2', 2), ('a3', 3), ('ref', 99)])
    state_path = _two_source_state(tmp_path, a1, ref)
    out_dir = tmp_path / 'out'

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(out_dir),
        '-i', f'1:{tmp_path}/a*.emd',
        '-i', f'3:{ref}',
    ])
    assert result.exit_code == 0, result.output
    assert sorted(d.name for d in out_dir.iterdir()) == [
        'run_0', 'run_1', 'run_2']
    # Every run's CvtB (node 4) should equal the broadcast reference.
    for run_dir in out_dir.iterdir():
        b_file = next(run_dir.glob('4_*.emd'))
        assert np.all(load_dataset(b_file).active_scalars == 99.0)


def test_cli_prefixed_repeated_for_same_id_concatenates(tmp_path):
    a1, a2, ref = _make_inputs(
        tmp_path, [('a1', 1), ('a2', 2), ('ref', 99)])
    state_path = _two_source_state(tmp_path, a1, ref)
    out_dir = tmp_path / 'out'

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(out_dir),
        '-i', f'1:{a1}',
        '-i', f'1:{a2}',
        '-i', f'3:{ref}',
    ])
    assert result.exit_code == 0, result.output
    assert sorted(d.name for d in out_dir.iterdir()) == ['run_0', 'run_1']


def test_cli_mixing_bare_and_prefixed_rejected(tmp_path):
    a, b, ref = _make_inputs(tmp_path, [('a', 1), ('b', 2), ('ref', 99)])
    state_path = _two_source_state(tmp_path, a, ref)

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(tmp_path / 'out'),
        '-i', str(a),
        '-i', f'3:{ref}',
    ])
    assert result.exit_code != 0
    assert 'Cannot mix bare and' in result.output


def test_cli_prefixed_glob_no_match_errors(tmp_path):
    a = _make_inputs(tmp_path, [('a', 1)])[0]
    state_path = _single_source_state(tmp_path, a)
    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(tmp_path / 'out'),
        '-i', f'1:{tmp_path}/zzz_*.emd',
    ])
    assert result.exit_code != 0
    assert 'matched no files' in result.output


def test_cli_prefixed_inconsistent_lengths_errors(tmp_path):
    a1, a2, b1, b2, b3 = _make_inputs(
        tmp_path, [('a1', 1), ('a2', 2),
                   ('b1', 10), ('b2', 20), ('b3', 30)])
    state_path = _two_source_state(tmp_path, a1, b1)

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(tmp_path / 'out'),
        '-i', f'1:{a1},{a2}',
        '-i', f'3:{b1},{b2},{b3}',
    ])
    assert result.exit_code != 0
    assert 'Inconsistent input list lengths' in result.output


def test_cli_input_path_does_not_exist_errors(tmp_path):
    a = _make_inputs(tmp_path, [('a', 1)])[0]
    state_path = _single_source_state(tmp_path, a)
    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(tmp_path / 'out'),
        '-i', str(tmp_path / 'never_existed.emd'),
    ])
    assert result.exit_code != 0
    assert 'does not exist' in result.output


# ---- no-override mode (single run) --------------------------------------


def test_run_no_inputs_executes_state_as_is(tmp_path):
    """``run(state, out)`` with ``inputs=None`` runs once using the
    pinned fileNames in the state file and writes outputs straight
    into ``output_dir`` (single run = flat layout)."""
    inputs = _make_inputs(tmp_path, [('pinned', 7)])
    state_path = _single_source_state(tmp_path, inputs[0])
    out_dir = tmp_path / 'out'

    run_dirs = run(state_path, out_dir)
    assert run_dirs == [out_dir]
    emds = list(out_dir.glob('*.emd'))
    assert len(emds) == 1
    assert np.all(load_dataset(emds[0]).active_scalars == 7.0)


def test_cli_no_inputs_writes_flat(tmp_path):
    """No --input → single run → flat output directly in --output-dir."""
    inputs = _make_inputs(tmp_path, [('pinned', 7)])
    state_path = _single_source_state(tmp_path, inputs[0])
    out_dir = tmp_path / 'out'

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(out_dir),
    ])
    assert result.exit_code == 0, result.output
    emds = list(out_dir.glob('*.emd'))
    assert len(emds) == 1
    # And no run_N subdir was created.
    assert not (out_dir / 'run_0').exists()


# ---- --run-prefix / run_dir_prefix --------------------------------------


def test_run_custom_run_dir_prefix(tmp_path):
    inputs = _make_inputs(tmp_path, [('a', 1), ('b', 2)])
    state_path = _single_source_state(tmp_path, inputs[0])

    run_dirs = run(state_path, tmp_path / 'out',
                   inputs=[str(p) for p in inputs],
                   run_dir_prefix='sample')
    assert [d.name for d in run_dirs] == ['sample_0', 'sample_1']


def test_cli_run_prefix_flag(tmp_path):
    inputs = _make_inputs(tmp_path, [('a', 1), ('b', 2)])
    state_path = _single_source_state(tmp_path, inputs[0])
    out_dir = tmp_path / 'out'

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(out_dir),
        '--run-prefix', 'sample',
        '-i', f'{inputs[0]},{inputs[1]}',
    ])
    assert result.exit_code == 0, result.output
    assert sorted(d.name for d in out_dir.iterdir()) == [
        'sample_0', 'sample_1']


# ---- per-run state snapshot ---------------------------------------------


def test_run_writes_state_snapshot_when_overrides_applied(tmp_path):
    """A run with overrides should drop a state.tvsm into each run's
    output directory; fileNames in the snapshot reflect that run's
    inputs, not the placeholder pinned in the original state file."""
    a, b = _make_inputs(tmp_path, [('a', 1), ('b', 2)])
    state_path = _single_source_state(tmp_path, a)

    run_dirs = run(state_path, tmp_path / 'out',
                   inputs=[str(a), str(b)])

    for run_dir, expected_path in zip(run_dirs, [a, b]):
        snapshot_path = run_dir / 'state.tvsm'
        assert snapshot_path.is_file(), (
            f'Expected state snapshot at {snapshot_path}')
        snapshot = json.loads(snapshot_path.read_text())
        nodes = snapshot['pipeline']['nodes']
        reader = next(n for n in nodes if n['type'] == 'source.reader')
        assert reader['fileNames'] == [str(expected_path)]


def test_run_without_overrides_still_writes_snapshot(tmp_path):
    """state.tvsm is always written so the output directory is
    self-contained. With no overrides it's a verbatim copy of the
    input state."""
    a = _make_inputs(tmp_path, [('a', 1)])[0]
    state_path = _single_source_state(tmp_path, a)

    run_dirs = run(state_path, tmp_path / 'out')
    snapshot = run_dirs[0] / 'state.tvsm'
    assert snapshot.is_file()
    # Same logical content as the input — fileNames unchanged since
    # there were no overrides to apply.
    original = json.loads(state_path.read_text())
    written = json.loads(snapshot.read_text())
    assert original == written


def test_run_state_snapshot_is_reproducible(tmp_path):
    """Re-running a snapshot's state.tvsm must produce byte-identical
    leaf outputs to the original run."""
    a = _make_inputs(tmp_path, [('a', 1)])[0]
    state_path = _single_source_state(tmp_path, a)

    original_runs = run(state_path, tmp_path / 'orig', inputs=[str(a)])
    snapshot = original_runs[0] / 'state.tvsm'
    assert snapshot.is_file()

    reproduced = run(snapshot, tmp_path / 'reproduced')
    assert len(reproduced) == 1

    # Compare the produced .emd files (excluding the snapshots).
    orig_emds = sorted(original_runs[0].glob('*.emd'))
    repro_emds = sorted(reproduced[0].glob('*.emd'))
    assert [e.name for e in orig_emds] == [e.name for e in repro_emds]
    for orig_file, repro_file in zip(orig_emds, repro_emds):
        np.testing.assert_array_equal(
            load_dataset(orig_file).active_scalars,
            load_dataset(repro_file).active_scalars)


def test_run_snapshot_dict_form_patches_each_source(tmp_path):
    """Multi-source dict form patches each overridden source's
    fileNames in the snapshot, leaving non-overridden nodes alone."""
    a1, a2, b1, b2 = _make_inputs(
        tmp_path, [('a1', 1), ('a2', 2), ('b1', 10), ('b2', 20)])
    state_path = _two_source_state(tmp_path, a1, b1)

    run_dirs = run(
        state_path,
        tmp_path / 'out',
        inputs={1: [str(a1), str(a2)], 3: [str(b1), str(b2)]},
    )
    for run_dir, paths in zip(run_dirs, [(a1, b1), (a2, b2)]):
        snapshot = json.loads((run_dir / 'state.tvsm').read_text())
        nodes = {n['id']: n for n in snapshot['pipeline']['nodes']}
        assert nodes[1]['fileNames'] == [str(paths[0])]
        assert nodes[3]['fileNames'] == [str(paths[1])]


def test_cli_writes_state_snapshot(tmp_path):
    inputs = _make_inputs(tmp_path, [('a', 1), ('b', 2)])
    state_path = _single_source_state(tmp_path, inputs[0])
    out_dir = tmp_path / 'out'

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(out_dir),
        '-i', f'{inputs[0]},{inputs[1]}',
    ])
    assert result.exit_code == 0, result.output
    assert (out_dir / 'run_0' / 'state.tvsm').is_file()
    assert (out_dir / 'run_1' / 'state.tvsm').is_file()


# ---- output_format ------------------------------------------------------


def test_run_output_format_port_default(tmp_path):
    """Default output_format='port': port files written, no tvh5."""
    a = _make_inputs(tmp_path, [('a', 1)])[0]
    state_path = _single_source_state(tmp_path, a)

    run_dirs = run(state_path, tmp_path / 'out', inputs=[str(a)])
    assert list(run_dirs[0].glob('*.emd'))
    assert not (run_dirs[0] / 'output_state.tvh5').exists()


def test_run_output_format_state_only_writes_tvh5(tmp_path):
    """output_format='state' produces output_state.tvh5 and no per-port
    files."""
    a = _make_inputs(tmp_path, [('a', 5)])[0]
    state_path = _single_source_state(tmp_path, a)

    run_dirs = run(state_path, tmp_path / 'out', inputs=[str(a)],
                   output_format='state')

    tvh5 = run_dirs[0] / 'output_state.tvh5'
    assert tvh5.is_file()
    # No standalone EMD beside the tvh5.
    assert not list(run_dirs[0].glob('*.emd'))
    # Sanity check the HDF5 layout.
    with h5py.File(tvh5, 'r') as f:
        assert 'tomviz_state' in f
        assert '/data' in f
        # ConvertToFloat (node 2) wrote its 'output' port into the
        # tvh5 — verify the EMD layout under the port group. The port
        # group IS the EMD node (as in C++ EmdFormat::writeNode), so
        # 'data'/'dim1' sit directly under it.
        assert '/data/2/output/data' in f
        assert '/data/2/output/dim1' in f


def test_run_output_format_state_plus_port_writes_both(tmp_path):
    a = _make_inputs(tmp_path, [('a', 1)])[0]
    state_path = _single_source_state(tmp_path, a)

    run_dirs = run(state_path, tmp_path / 'out', inputs=[str(a)],
                   output_format='state+port')

    assert (run_dirs[0] / 'output_state.tvh5').is_file()
    assert list(run_dirs[0].glob('*.emd'))


def test_run_output_format_invalid_raises(tmp_path):
    a = _make_inputs(tmp_path, [('a', 1)])[0]
    state_path = _single_source_state(tmp_path, a)
    with pytest.raises(ValueError, match='output_format must be'):
        run(state_path, tmp_path / 'out', inputs=[str(a)],
            output_format='bogus')


def test_run_output_format_state_works_without_overrides(tmp_path):
    """output_format is independent of overrides — a no-input run still
    bundles its outputs into a tvh5."""
    a = _make_inputs(tmp_path, [('a', 1)])[0]
    state_path = _single_source_state(tmp_path, a)

    run_dirs = run(state_path, tmp_path / 'out',
                   output_format='state')
    assert (run_dirs[0] / 'output_state.tvh5').is_file()
    # state.tvsm is always written for self-contained output.
    assert (run_dirs[0] / 'state.tvsm').is_file()


def test_tvh5_port_group_layout_matches_emdformat_readnode(tmp_path):
    """Regression: C++ Tvh5Format::populatePayloadData calls
    EmdFormat::readNode(reader, /data/<id>/<port>, ...), which expects
    ``<port>/data`` to be the volume *dataset* and ``<port>/dim1/2/3``
    to be dim datasets. Writing the standalone-.emd layout
    ``<port>/data/tomography/data`` makes the C++ side log
    'failed to read /data/<id>/<port>' and abandon the load."""
    a = _make_inputs(tmp_path, [('a', 1)])[0]
    state_path = _single_source_state(tmp_path, a)

    run_dirs = run(state_path, tmp_path / 'out',
                   output_format='state')
    tvh5 = run_dirs[0] / 'output_state.tvh5'

    with h5py.File(tvh5, 'r') as f:
        for port_group_path in (f'/data/{n}/{p}'
                                for n in f['/data']
                                for p in f[f'/data/{n}']):
            port_group = f[port_group_path]
            assert 'data' in port_group, (
                f'{port_group_path}/data missing')
            assert isinstance(port_group['data'], h5py.Dataset), (
                f'{port_group_path}/data must be a Dataset, not a '
                f'Group; got {type(port_group["data"]).__name__}')
            for dim in ('dim1', 'dim2', 'dim3'):
                assert isinstance(port_group[dim], h5py.Dataset), (
                    f'{port_group_path}/{dim} must be a Dataset')


def test_tvh5_emd_dims_are_float32(tmp_path):
    """Regression: the C++ EmdFormat reader reads dim1/dim2/dim3 via
    readData<float>, which strictly requires H5T_IEEE_F32LE storage.
    Numpy's defaults (int64 from range, float64 from arange-with-
    float-step) make tomviz reject the file at attribute/dim read
    time. Coerce to float32 in the writer."""
    a = _make_inputs(tmp_path, [('a', 1)])[0]
    state_path = _single_source_state(tmp_path, a)

    run_dirs = run(state_path, tmp_path / 'out',
                   output_format='state')
    tvh5 = run_dirs[0] / 'output_state.tvh5'

    with h5py.File(tvh5, 'r') as f:
        # Walk to one of the embedded EMD payloads. Per-port groups
        # follow the C++ EmdFormat::writeNode layout: data/dim* sit
        # directly under the port group, NOT nested in /data/tomography
        # (that outer wrapper only exists in standalone .emd files).
        port_groups = [k for k in f['/data'].keys()]
        assert port_groups, 'expected at least one port payload'
        port = list(f[f'/data/{port_groups[0]}'].keys())[0]
        node_group = f[f'/data/{port_groups[0]}/{port}']
        for dim in ('dim1', 'dim2', 'dim3'):
            assert node_group[dim].dtype == np.float32, (
                f'{dim} must be float32 for C++ EmdFormat readData<float> '
                f'compatibility; got {node_group[dim].dtype}')


def test_tvh5_tomviz_state_is_signed_int8(tmp_path):
    """Regression: the C++ Tvh5Format::readState reads /tomviz_state
    via H5ReadWrite::readData<char>, which maps to H5T_STD_I8LE and
    rejects mismatched storage types via H5Tequal. Writing the JSON
    blob as uint8 makes tomviz fall back to LegacyStateLoader and
    fail to open the file. The bytes are identical; only the HDF5
    type label matters."""
    a = _make_inputs(tmp_path, [('a', 1)])[0]
    state_path = _single_source_state(tmp_path, a)

    run_dirs = run(state_path, tmp_path / 'out',
                   output_format='state')
    tvh5 = run_dirs[0] / 'output_state.tvh5'

    with h5py.File(tvh5, 'r') as f:
        ds = f['tomviz_state']
        assert ds.dtype == np.int8, (
            f'tomviz_state must be int8 for C++ Tvh5Format::readState '
            f'compatibility; got {ds.dtype}')


def test_run_output_format_state_round_trip(tmp_path):
    """Write a tvh5, load it back, verify the dataRefs are honored
    and downstream nodes read the embedded payloads. The reloaded
    pipeline's source node should be marked Current (data populated
    from dataRef), and re-running should produce identical leaf
    outputs to the original run."""
    a = _make_inputs(tmp_path, [('a', 7)])[0]
    state_path = _single_source_state(tmp_path, a)

    original = run(state_path, tmp_path / 'orig', inputs=[str(a)],
                   output_format='state+port')
    tvh5 = original[0] / 'output_state.tvh5'

    # Re-run from the tvh5 (no overrides; the reader's data is
    # already populated via dataRef).
    reproduced = run(tvh5, tmp_path / 'reproduced')

    orig_emds = sorted(original[0].glob('*.emd'))
    repro_emds = sorted(reproduced[0].glob('*.emd'))
    assert [e.name for e in orig_emds] == [e.name for e in repro_emds]
    for orig_file, repro_file in zip(orig_emds, repro_emds):
        np.testing.assert_array_equal(
            load_dataset(orig_file).active_scalars,
            load_dataset(repro_file).active_scalars)


def test_run_output_format_state_persists_table_payloads(tmp_path):
    """A LegacyPythonTransform that returns a vtkTable result is
    persisted column-by-column under /data/<id>/<port>/c<i>, mirroring
    C++ Tvh5Format::writeTablePayload. The dataRef on the port entry
    must point at the group, and on reload (via load_state) the port
    must carry an equivalent vtkTable."""
    pytest.importorskip('vtk')
    from tomviz_pipeline.state.state_io import load_state

    arr = (np.arange(24, dtype=np.uint8) + 1).reshape((2, 3, 4))
    input_emd = tmp_path / 'input.emd'
    _write_simple_emd(input_emd, arr)

    script = '''
import vtk

def transform(dataset):
    t = vtk.vtkTable()
    radius = vtk.vtkFloatArray()
    radius.SetName("radius")
    radius.SetNumberOfTuples(2)
    radius.SetValue(0, 1.0)
    radius.SetValue(1, 2.5)
    t.AddColumn(radius)
    labels = vtk.vtkStringArray()
    labels.SetName("labels")
    labels.SetNumberOfValues(2)
    labels.SetValue(0, "first")
    labels.SetValue(1, "second")
    t.AddColumn(labels)
    return {"stats": t}
'''
    description = json.dumps({
        'name': 'TableEmitter',
        'label': 'Table',
        'apply_to_each_array': False,
        'results': [{'name': 'stats', 'type': 'table'}],
    })
    state = {
        'schemaVersion': 2,
        'pipeline': {
            'nodes': [
                {'id': 1, 'type': 'source.reader', 'label': 'Reader',
                 'fileNames': [str(input_emd)]},
                {'id': 2, 'type': 'transform.legacyPython',
                 'label': 'TableOp',
                 'description': description, 'script': script},
            ],
            'links': [
                {'from': {'node': 1, 'port': 'volume'},
                 'to':   {'node': 2, 'port': 'volume'}},
            ],
        },
    }
    state_path = tmp_path / 'state.tvsm'
    state_path.write_text(json.dumps(state))

    run_dirs = run(state_path, tmp_path / 'out', output_format='state')
    tvh5 = run_dirs[0] / 'output_state.tvh5'
    assert tvh5.is_file()

    with h5py.File(tvh5, 'r') as f:
        port_group = f['/data/2/stats']
        assert port_group.attrs['type'] == 'table' or \
            port_group.attrs['type'] == b'table'
        assert int(port_group.attrs['numColumns']) == 2
        assert int(port_group.attrs['numRows']) == 2
        # Numeric column round-trips dtype + values exactly.
        c0 = port_group['c0']
        assert int(c0.attrs['vtkDataType']) == 10  # VTK_FLOAT
        np.testing.assert_allclose(c0[()], np.array([1.0, 2.5],
                                                    dtype=np.float32))
        # String column is JSON-encoded int8.
        c1 = port_group['c1']
        assert int(c1.attrs['vtkDataType']) == 13  # VTK_STRING
        text = c1[()].tobytes().decode('utf-8')
        assert json.loads(text) == ['first', 'second']

    # Round-trip the tvh5 back through the loader and check the
    # transform's stats port carries an equivalent pure-Python Table
    # (loading never constructs vtk objects).
    from tomviz_pipeline.table import Table
    pipeline = load_state(tvh5)
    transform_node = next(n for n in pipeline.nodes if n.id == 2)
    stats_port = transform_node.output_port('stats')
    table = stats_port.data().payload
    assert isinstance(table, Table)
    assert table.num_columns == 2
    assert table.num_rows == 2
    assert table.column_names == ['radius', 'labels']
    radius = table.column('radius')
    assert radius.dtype == np.float32
    np.testing.assert_allclose(radius, [1.0, 2.5])
    assert table.column('labels') == ['first', 'second']


def test_run_output_format_state_persists_molecule_payloads(tmp_path):
    """A LegacyPythonTransform that returns a vtkMolecule result is
    persisted as atomic-numbers / positions / bond datasets under
    /data/<id>/<port>/. On reload (via load_state) the port carries an
    equivalent vtkMolecule. Mirrors the table case below."""
    pytest.importorskip('vtk')
    from tomviz_pipeline.state.state_io import load_state

    arr = (np.arange(24, dtype=np.uint8) + 1).reshape((2, 3, 4))
    input_emd = tmp_path / 'input.emd'
    _write_simple_emd(input_emd, arr)

    script = '''
import vtk

def transform(dataset):
    m = vtk.vtkMolecule()
    m.AppendAtom(1, 0.0, 0.0, 0.0)
    m.AppendAtom(8, 0.96, 0.0, 0.0)
    m.AppendAtom(1, 1.20, 0.93, 0.0)
    m.AppendBond(0, 1, 1)
    m.AppendBond(1, 2, 2)
    return {"mol": m}
'''
    description = json.dumps({
        'name': 'MoleculeEmitter',
        'label': 'Molecule',
        'apply_to_each_array': False,
        'results': [{'name': 'mol', 'type': 'molecule'}],
    })
    state = {
        'schemaVersion': 2,
        'pipeline': {
            'nodes': [
                {'id': 1, 'type': 'source.reader', 'label': 'Reader',
                 'fileNames': [str(input_emd)]},
                {'id': 2, 'type': 'transform.legacyPython',
                 'label': 'MolOp',
                 'description': description, 'script': script},
            ],
            'links': [
                {'from': {'node': 1, 'port': 'volume'},
                 'to':   {'node': 2, 'port': 'volume'}},
            ],
        },
    }
    state_path = tmp_path / 'state.tvsm'
    state_path.write_text(json.dumps(state))

    run_dirs = run(state_path, tmp_path / 'out', output_format='state')
    tvh5 = run_dirs[0] / 'output_state.tvh5'
    assert tvh5.is_file()

    with h5py.File(tvh5, 'r') as f:
        port_group = f['/data/2/mol']
        assert (port_group.attrs['type'] in ('molecule', b'molecule'))
        assert int(port_group.attrs['numAtoms']) == 3
        assert int(port_group.attrs['numBonds']) == 2
        np.testing.assert_array_equal(
            np.asarray(port_group['atomicNumbers'][()]),
            np.array([1, 8, 1], dtype=np.uint16))
        np.testing.assert_allclose(
            np.asarray(port_group['atomPositions'][()]).reshape(-1, 3),
            np.array([[0.0, 0.0, 0.0],
                      [0.96, 0.0, 0.0],
                      [1.20, 0.93, 0.0]], dtype=np.float32))
        np.testing.assert_array_equal(
            np.asarray(port_group['bondAtoms'][()]).reshape(-1, 2),
            np.array([[0, 1], [1, 2]], dtype=np.int64))
        np.testing.assert_array_equal(
            np.asarray(port_group['bondOrders'][()]),
            np.array([1, 2], dtype=np.uint16))

    # On reload the port carries an equivalent pure-Python Molecule
    # (loading never constructs vtk objects).
    from tomviz_pipeline.molecule import Molecule
    pipeline = load_state(tvh5)
    transform_node = next(n for n in pipeline.nodes if n.id == 2)
    mol_port = transform_node.output_port('mol')
    molecule = mol_port.data().payload
    assert isinstance(molecule, Molecule)
    assert molecule.num_atoms == 3
    assert molecule.num_bonds == 2
    np.testing.assert_array_equal(molecule.atomic_numbers, [1, 8, 1])
    np.testing.assert_allclose(
        molecule.positions,
        np.array([[0.0, 0.0, 0.0],
                  [0.96, 0.0, 0.0],
                  [1.20, 0.93, 0.0]], dtype=np.float32))
    np.testing.assert_array_equal(molecule.bonds, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(molecule.bond_orders, [1, 2])


def test_cli_output_format_flag(tmp_path):
    a = _make_inputs(tmp_path, [('a', 1)])[0]
    state_path = _single_source_state(tmp_path, a)
    out_dir = tmp_path / 'out'

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '-s', str(state_path),
        '-o', str(out_dir),
        '-i', str(a),
        '--output-format', 'state+port',
    ])
    assert result.exit_code == 0, result.output
    # Single run → flat layout under out_dir.
    assert (out_dir / 'output_state.tvh5').is_file()
    assert list(out_dir.glob('*.emd'))
