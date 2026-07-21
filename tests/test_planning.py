###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Tests for incremental execution planning and graph mutation semantics:
execution_plan pruning, mark_stale cascades, transient-output eviction and
re-planning, targeted execution, cycle rejection, link replacement, node/
link removal, failure cascades, and dict-level state round-trips."""

import pytest

from tomviz_pipeline.core import (
    DefaultExecutor,
    NodeExecState,
    NodeExecutor,
    NodeExecutorFactory,
    NodeFactory,
    NodeState,
    Pipeline,
    PortData,
    SourceNode,
    TransformNode,
    pipeline_from_state_dict,
    pipeline_to_state_dict,
)


class _Source(SourceNode):
    type_name = 'test.planning.source'

    def __init__(self, value=1.0, persistent=True):
        super().__init__()
        self.value = value
        self.run_count = 0
        self.add_output('out', 'ImageData', persistent=persistent)

    def execute(self):
        self.run_count += 1
        self.output_port('out').set_data(
            PortData(self.value, 'ImageData'))
        return True


class _Double(TransformNode):
    type_name = 'test.planning.double'

    def __init__(self):
        super().__init__()
        self.run_count = 0
        self.add_input('in', 'ImageData')
        self.add_output('out', 'ImageData')

    def transform(self, inputs):
        self.run_count += 1
        return {'out': PortData(inputs['in'].payload * 2, 'ImageData')}


class _Merge(TransformNode):
    type_name = 'test.planning.merge'

    def __init__(self):
        super().__init__()
        self.add_input('a', 'ImageData')
        self.add_input('b', 'ImageData')
        self.add_output('out', 'ImageData')

    def transform(self, inputs):
        total = inputs['a'].payload + inputs['b'].payload
        return {'out': PortData(total, 'ImageData')}


class _Fails(TransformNode):
    type_name = 'test.planning.fails'

    def __init__(self):
        super().__init__()
        self.add_input('in', 'ImageData')
        self.add_output('out', 'ImageData')

    def transform(self, inputs):
        return None


class _RecordingExecutor(NodeExecutor):
    """Minimal in-process NodeExecutor with a type_name so it serializes
    into the node's `executor` block."""

    type_name = 'test.planning.exec'

    def execute(self, node):
        return node.execute()


@pytest.fixture(autouse=True)
def _register_test_types():
    NodeFactory.register(_Source.type_name, _Source)
    NodeFactory.register(_Double.type_name, _Double)
    NodeExecutorFactory.register(_RecordingExecutor.type_name,
                                 _RecordingExecutor)


def _chain(n_transforms=2, persistent=True, value=1.0):
    """src -> t1 -> ... -> tn. Returns (pipeline, [src, t1, ..., tn])."""
    p = Pipeline()
    src = _Source(value=value, persistent=persistent)
    p.add_node(src)
    nodes = [src]
    for _ in range(n_transforms):
        t = _Double()
        p.add_node(t)
        p.create_link(nodes[-1].output_port('out'), t.input_port('in'))
        nodes.append(t)
    return p, nodes


# ---- execution_plan ---------------------------------------------------------


def test_full_plan_on_fresh_graph():
    p, (src, t1, t2) = _chain()
    plan = p.execution_plan()
    assert plan == [src, t1, t2]


def test_current_nodes_pruned_from_plan():
    p, (src, t1, t2) = _chain()
    future = p.execute()
    assert future.succeeded()
    assert all(n.state == NodeState.Current for n in (src, t1, t2))
    # Everything is up to date: nothing to plan.
    assert p.execution_plan() == []


def test_mark_stale_cascade_reincludes_downstream_chain():
    p, (src, t1, t2) = _chain()
    p.execute()
    t1.mark_stale()
    assert t1.state == NodeState.Stale
    assert t2.state == NodeState.Stale
    assert src.state == NodeState.Current
    # The plan re-includes the stale chain but not the Current source
    # (its output data is still live).
    assert p.execution_plan() == [t1, t2]


def test_plan_is_topo_ordered():
    p = Pipeline()
    src = _Source()
    a = _Double()
    b = _Double()
    merge = _Merge()
    # Add in scrambled order; planning must still be topological.
    for n in (merge, b, src, a):
        p.add_node(n)
    p.create_link(src.output_port('out'), a.input_port('in'))
    p.create_link(src.output_port('out'), b.input_port('in'))
    p.create_link(a.output_port('out'), merge.input_port('a'))
    p.create_link(b.output_port('out'), merge.input_port('b'))

    plan = p.execution_plan()
    assert sorted(plan, key=id) == sorted([src, a, b, merge], key=id)
    assert plan.index(src) < plan.index(a) < plan.index(merge)
    assert plan.index(src) < plan.index(b) < plan.index(merge)

    future = p.execute()
    assert future.succeeded()
    assert merge.output_port('out').data().payload == 4.0


# ---- transient (non-persistent) outputs -------------------------------------


def test_transient_output_evicted_after_plan_execution():
    p, (src, t) = _chain(n_transforms=1, persistent=False, value=3.0)
    future = p.execute()
    assert future.succeeded()
    assert src.state == NodeState.Current
    assert t.state == NodeState.Current
    # The consumer got its data before the plan ended...
    assert t.output_port('out').data().payload == 6.0
    # ...but the transient producer output was released afterwards.
    assert not src.output_port('out').has_data()


def test_next_plan_reincludes_current_producer_with_evicted_output():
    p, (src, t) = _chain(n_transforms=1, persistent=False, value=3.0)
    p.execute()
    assert src.run_count == 1

    t.mark_stale()
    # src is Current but its output was evicted: the planner must bring
    # it back so the consumer has data to read.
    plan = p.execution_plan()
    assert plan == [src, t]

    future = p.execute()
    assert future.succeeded()
    assert src.run_count == 2
    assert t.state == NodeState.Current
    assert t.output_port('out').data().payload == 6.0
    # Evicted again at the end of the new plan.
    assert not src.output_port('out').has_data()


def test_legacy_default_executor_execute_does_not_evict():
    p, (src, t) = _chain(n_transforms=1, persistent=False, value=3.0)
    ok = DefaultExecutor(p).execute()
    assert ok is True
    # The legacy full-order walk leaves transient outputs in place (leaf
    # writers still need them).
    assert src.output_port('out').has_data()
    assert t.output_port('out').data().payload == 6.0


# ---- targeted execution ------------------------------------------------------


def test_execute_target_runs_only_needed_subgraph():
    p, (src, a, b) = _chain()
    c = _Double()  # second branch off the source
    p.add_node(c)
    p.create_link(src.output_port('out'), c.input_port('in'))

    future = p.execute(b)
    assert future.succeeded()
    assert src.state == NodeState.Current
    assert a.state == NodeState.Current
    assert b.state == NodeState.Current
    # The sibling branch was not needed and did not run.
    assert c.state == NodeState.New
    assert c.run_count == 0


def test_execute_upstream_of_target_does_not_run_target():
    p, (src, a, b) = _chain()
    future = p.execute_upstream_of(b)
    assert future.succeeded()
    assert src.state == NodeState.Current
    assert a.state == NodeState.Current
    assert a.output_port('out').has_data()
    # The target itself was left untouched.
    assert b.state == NodeState.New
    assert b.run_count == 0
    assert not b.output_port('out').has_data()


# ---- graph mutation ----------------------------------------------------------


def test_would_create_cycle_and_create_link_raises():
    p = Pipeline()
    a = _Double()
    b = _Double()
    p.add_node(a)
    p.add_node(b)
    p.create_link(a.output_port('out'), b.input_port('in'))

    assert p.would_create_cycle(b.output_port('out'),
                                a.input_port('in')) is True
    # Self-loop.
    assert p.would_create_cycle(a.output_port('out'),
                                a.input_port('in')) is True
    assert p.would_create_cycle(a.output_port('out'),
                                b.input_port('in')) is False

    with pytest.raises(ValueError):
        p.create_link(b.output_port('out'), a.input_port('in'))
    # The offending link was not added.
    assert len(p.links) == 1


def test_second_link_to_same_input_replaces_first():
    p = Pipeline()
    src1 = _Source(value=1.0)
    src2 = _Source(value=2.0)
    t = _Double()
    for n in (src1, src2, t):
        p.add_node(n)

    first = p.create_link(src1.output_port('out'), t.input_port('in'))
    assert t.input_port('in').link is first

    second = p.create_link(src2.output_port('out'), t.input_port('in'))
    assert t.input_port('in').link is second
    assert p.links == [second]
    # The replaced link is fully unhooked from its producer.
    assert src1.output_port('out').outgoing_links == []
    assert src2.output_port('out').outgoing_links == [second]


def test_remove_link_marks_downstream_stale():
    p, (src, a, b) = _chain()
    p.execute()
    link = a.input_port('in').link
    p.remove_link(link)
    assert src.state == NodeState.Current
    assert a.state == NodeState.Stale
    assert b.state == NodeState.Stale
    assert link not in p.links
    assert src.output_port('out').outgoing_links == []
    assert a.input_port('in').link is None


def test_remove_node_marks_downstream_stale():
    p, (src, a, b) = _chain()
    p.execute()
    p.remove_node(a)
    assert a not in p.nodes
    # Both links touching `a` are gone.
    assert p.links == []
    assert src.output_port('out').outgoing_links == []
    assert b.input_port('in').link is None
    assert b.state == NodeState.Stale
    assert src.state == NodeState.Current


def test_clear_empties_graph():
    p, _ = _chain()
    p.clear()
    assert p.nodes == []
    assert p.links == []


# ---- failure semantics ------------------------------------------------------


def test_failing_node_cascades_and_sibling_branch_still_runs():
    p = Pipeline()
    src = _Source()
    fail = _Fails()
    down = _Double()
    sib = _Double()
    for n in (src, fail, down, sib):
        p.add_node(n)
    p.create_link(src.output_port('out'), fail.input_port('in'))
    p.create_link(fail.output_port('out'), down.input_port('in'))
    p.create_link(src.output_port('out'), sib.input_port('in'))

    finished = []
    p.execution_finished.connect(lambda fut: finished.append(fut))

    future = p.execute()
    assert future.is_finished()
    assert future.succeeded() is False
    assert future.was_canceled() is False
    # The signal carried the same future execute() returned.
    assert finished == [future]

    assert fail.state == NodeState.Stale
    assert fail.exec_state == NodeExecState.Failed
    # Downstream of the failure: skipped, marked Stale, never executed.
    assert down.state == NodeState.Stale
    assert down.run_count == 0
    # The independent sibling branch still ran.
    assert sib.state == NodeState.Current
    assert sib.output_port('out').data().payload == 2.0


# ---- state round-trip -------------------------------------------------------


def test_state_roundtrip_preserves_ids_labels_and_links():
    p, (src, t1, t2) = _chain()
    src.label = 'The Source'
    t1.label = 'First Double'
    t2.label = 'Second Double'

    state = pipeline_to_state_dict(p)
    assert state['schemaVersion'] == 2
    entries = state['pipeline']['nodes']
    assert [e['id'] for e in entries] == [src.id, t1.id, t2.id]
    assert [e['type'] for e in entries] == [
        _Source.type_name, _Double.type_name, _Double.type_name]
    assert state['pipeline']['links'] == [
        {'from': {'node': src.id, 'port': 'out'},
         'to': {'node': t1.id, 'port': 'in'}},
        {'from': {'node': t1.id, 'port': 'out'},
         'to': {'node': t2.id, 'port': 'in'}},
    ]

    restored = pipeline_from_state_dict(state)
    assert len(restored.nodes) == 3
    assert len(restored.links) == 2
    rsrc = restored.node_by_id(src.id)
    rt1 = restored.node_by_id(t1.id)
    rt2 = restored.node_by_id(t2.id)
    assert isinstance(rsrc, _Source)
    assert isinstance(rt1, _Double)
    assert rsrc.label == 'The Source'
    assert rt1.label == 'First Double'
    assert rt2.label == 'Second Double'
    assert rt1.input_port('in').link.from_port is rsrc.output_port('out')
    assert rt2.input_port('in').link.from_port is rt1.output_port('out')
    # A fresh (never-run) graph loads back all-New.
    assert all(n.state == NodeState.New for n in restored.nodes)


def test_state_roundtrip_reset_to_new_resets_states():
    p, nodes = _chain()
    p.execute()
    state = pipeline_to_state_dict(p)
    for entry in state['pipeline']['nodes']:
        assert entry['state'] == 'Current'

    restored = pipeline_from_state_dict(state, reset_to_new=True)
    assert all(n.state == NodeState.New for n in restored.nodes)


def test_state_roundtrip_downgrades_dataless_current_producers():
    p, (src, t) = _chain(n_transforms=1)
    p.execute()
    state = pipeline_to_state_dict(p)

    restored = pipeline_from_state_dict(state)
    rsrc = restored.node_by_id(src.id)
    rt = restored.node_by_id(t.id)
    # src was saved Current but its needed output has no data after a
    # dict-only round trip: it must be downgraded so it re-runs.
    assert rsrc.state == NodeState.Stale
    # The downgrade cascades (C++ PipelineStateIO pass 3 uses markStale):
    # the leaf's result was computed from data that must be regenerated.
    assert rt.state == NodeState.Stale


def test_state_roundtrip_preserves_stale_state():
    p, (src, t) = _chain(n_transforms=1)
    p.execute()
    t.mark_stale()
    state = pipeline_to_state_dict(p)
    entry = [e for e in state['pipeline']['nodes'] if e['id'] == t.id][0]
    assert entry['state'] == 'Stale'

    restored = pipeline_from_state_dict(state)
    assert restored.node_by_id(t.id).state == NodeState.Stale


# ---- node executor block ----------------------------------------------------


def test_node_executor_block_serialized():
    node = _Double()
    assert 'executor' not in node.serialize()
    node.node_executor = _RecordingExecutor()
    entry = node.serialize()
    assert entry['executor'] == {'type': _RecordingExecutor.type_name}


def test_node_executor_block_restored_via_factory():
    p, (src, t) = _chain(n_transforms=1)
    t.node_executor = _RecordingExecutor()
    state = pipeline_to_state_dict(p)

    restored = pipeline_from_state_dict(state)
    rsrc = restored.node_by_id(src.id)
    rt = restored.node_by_id(t.id)
    assert rsrc.node_executor is None
    assert isinstance(rt.node_executor, _RecordingExecutor)

    # The restored executor is actually used to run the node.
    future = restored.execute()
    assert future.succeeded()
    assert rt.output_port('out').data().payload == 2.0
