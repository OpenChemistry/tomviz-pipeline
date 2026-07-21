###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Output-port persistence: transient release, OnDisk spill/reload,
runtime mode switching, serialization, defaults, and the executors'
in-flight residency management. Ports the observable behavior pinned by
the C++ PipelineLibTest persistence battery."""

import gc
import os

import pytest

from tomviz_pipeline import (
    DataLocation,
    PersistenceMode,
    Pipeline,
    PipelineSettings,
    PortData,
    SourceNode,
    TransformNode,
    TransformPersistenceDefault,
    pipeline_from_state_dict,
    pipeline_to_state_dict,
)


@pytest.fixture(autouse=True)
def _restore_pipeline_settings():
    settings = PipelineSettings.instance()
    saved = settings.transform_persistence_default
    yield
    settings.transform_persistence_default = saved


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    """Route port cache files to a per-test dir so tests can observe
    them and nothing leaks into the system temp dir."""
    cache = tmp_path / 'port-cache'
    cache.mkdir()
    monkeypatch.setenv('TOMVIZ_PORT_CACHE_DIR', str(cache))
    return cache


def _cache_files(cache_dir):
    return sorted(p for p in os.listdir(cache_dir)
                  if p.startswith('tomviz_port_'))


class _Const(SourceNode):
    type_name = 'ptest.const'

    def __init__(self, value=1):
        super().__init__()
        self.value = value
        self.run_count = 0
        self.add_output('out', 'ImageData')

    def execute(self):
        self.run_count += 1
        self.output_port('out').set_data(PortData(self.value, 'ImageData'))
        return True


class _Double(TransformNode):
    type_name = 'ptest.double'

    def __init__(self):
        super().__init__()
        self.run_count = 0
        self.add_input('in', 'ImageData')
        self.add_output('out', 'ImageData')

    def transform(self, inputs):
        self.run_count += 1
        return {'out': PortData(inputs['in'].payload * 2, 'ImageData')}


def _chain(n_transforms=2):
    p = Pipeline()
    src = _Const(3)
    p.add_node(src)
    nodes = [src]
    prev = src
    for _ in range(n_transforms):
        t = _Double()
        p.add_node(t)
        p.create_link(prev.output_port('out'), t.input_port('in'))
        nodes.append(t)
        prev = t
    return p, nodes


# ---- transient ---------------------------------------------------------


def test_transient_intermediate_released_after_plan():
    p, (src, t1, t2) = _chain()
    t1.output_port('out').persistent = False

    future = p.execute()
    assert future.succeeded()
    assert t2.output_port('out').data().payload == 12

    # The in-flight table dropped at end of plan: the transient
    # intermediate is gone, while the persistent source and the leaf
    # (never taken) keep their data.
    assert not t1.output_port('out').has_data()
    assert src.output_port('out').has_data()
    assert t2.output_port('out').has_data()


def test_transient_leaf_keeps_data_until_released():
    """Leaf outputs are never taken by the executor, so even a transient
    leaf keeps its payload after the plan (C++ parity: 'no eager
    take')."""
    p, (src, t1) = _chain(n_transforms=1)
    t1.output_port('out').persistent = False

    p.execute()
    assert t1.output_port('out').has_data()
    assert t1.output_port('out').data().payload == 6


def test_transient_data_alive_while_handle_held():
    port = _Const(5).output_port('out')
    port.persistent = False
    port.set_data(PortData(5, 'ImageData'))

    handle = port.take()
    assert handle is not None
    assert port.has_data()          # alive through the held handle
    del handle
    gc.collect()
    assert not port.has_data()      # last holder dropped -> released


def test_evicted_transient_producer_reincluded_and_rerun():
    p, (src, t1, t2) = _chain()
    t1.output_port('out').persistent = False

    p.execute()
    assert t1.run_count == 1
    assert not t1.output_port('out').has_data()

    # A new consumer of t1's evicted output forces t1 (Current, no
    # data) back into the plan; its inputs are satisfied by the
    # persistent source, so nothing else re-runs.
    t3 = _Double()
    p.add_node(t3)
    p.create_link(t1.output_port('out'), t3.input_port('in'))
    plan = p.execution_plan()
    assert t1 in plan and t3 in plan
    assert src not in plan and t2 not in plan

    future = p.execute()
    assert future.succeeded()
    assert t1.run_count == 2
    assert src.run_count == 1
    assert t3.output_port('out').data().payload == 12


def test_shared_handle_across_two_consumers_in_one_plan():
    p = Pipeline()
    src = _Const(3)
    t = _Double()
    a = _Double()
    b = _Double()
    p.add_node(src)
    p.add_node(t)
    p.add_node(a)
    p.add_node(b)
    p.create_link(src.output_port('out'), t.input_port('in'))
    t.output_port('out').persistent = False
    p.create_link(t.output_port('out'), a.input_port('in'))
    p.create_link(t.output_port('out'), b.input_port('in'))

    future = p.execute()
    assert future.succeeded()
    # Both consumers of the transient output saw the data (the take
    # happens once; the handle is shared for the rest of the plan).
    assert a.output_port('out').data().payload == 12
    assert b.output_port('out').data().payload == 12
    assert not t.output_port('out').has_data()


# ---- OnDisk ------------------------------------------------------------


def _on_disk_port(cache_dir, value=7):
    src = _Const(value)
    port = src.output_port('out')
    port.persistence_mode = PersistenceMode.OnDisk  # persistent already True
    src.execute()
    return src, port


def test_on_disk_evicts_on_last_handle_release(_cache_dir):
    src, port = _on_disk_port(_cache_dir)
    assert port.data_location() == DataLocation.InMemory

    handle = port.take()
    assert handle is not None
    del handle
    gc.collect()

    # Payload spilled: still "has data", but only on disk.
    assert port.has_data()
    assert port.data_location() == DataLocation.OnDisk
    assert port.data() is None            # data() is a non-loading peek
    assert len(_cache_files(_cache_dir)) == 1

    reloaded = port.materialize()         # loading read
    assert reloaded is not None
    assert reloaded.data.payload == 7
    assert port.data_location() == DataLocation.InMemory

    del reloaded
    gc.collect()
    assert port.data_location() == DataLocation.OnDisk  # re-evicted


def test_on_disk_re_set_data_overwrites(_cache_dir):
    src, port = _on_disk_port(_cache_dir, value=7)
    handle = port.take()
    del handle
    gc.collect()
    assert port.data_location() == DataLocation.OnDisk

    src.value = 42
    src.execute()                          # new payload
    handle = port.take()
    del handle
    gc.collect()

    reloaded = port.materialize()
    assert reloaded is not None
    assert reloaded.data.payload == 42     # reload sees the NEW payload


def test_on_disk_port_gc_removes_cache_file(_cache_dir):
    src, port = _on_disk_port(_cache_dir)
    handle = port.take()
    del handle
    gc.collect()
    assert len(_cache_files(_cache_dir)) == 1

    del src, port
    gc.collect()
    assert _cache_files(_cache_dir) == []


def test_on_disk_spill_failure_drops_data(_cache_dir):
    """Payloads the serializer cannot handle are dropped on spill; the
    planner then re-runs the producer (C++ parity for non-serializable
    payloads)."""
    src, port = _on_disk_port(_cache_dir)
    port.set_data(PortData(lambda: None, 'ImageData'))  # unpicklable
    handle = port.take()
    del handle
    gc.collect()
    assert not port.has_data()
    assert port.data_location() == DataLocation.Nowhere


def test_on_disk_intermediate_spills_and_feeds_next_plan(_cache_dir):
    """End to end: an OnDisk intermediate spills at end of plan, is NOT
    re-included by the planner (its data is on disk), and a new consumer
    gets the payload via a disk reload instead of a producer re-run."""
    p, (src, t1, t2) = _chain()
    t1.output_port('out').persistence_mode = PersistenceMode.OnDisk

    future = p.execute()
    assert future.succeeded()
    assert t1.output_port('out').data_location() == DataLocation.OnDisk

    t3 = _Double()
    p.add_node(t3)
    p.create_link(t1.output_port('out'), t3.input_port('in'))
    plan = p.execution_plan()
    assert t3 in plan
    assert t1 not in plan                  # data on disk counts as data

    future = p.execute()
    assert future.succeeded()
    assert t1.run_count == 1               # never re-ran
    assert t3.output_port('out').data().payload == 12


# ---- runtime mode switching ---------------------------------------------


def test_switch_in_memory_to_transient_releases_now():
    src = _Const(7)
    port = src.output_port('out')          # persistent InMemory default
    src.execute()
    assert port.has_data()

    port.persistent = False
    assert not port.has_data()


def test_switch_in_memory_to_on_disk_spills_now(_cache_dir):
    src, port = _on_disk_port(_cache_dir)  # switch happens before data
    src.execute()
    assert port.data_location() == DataLocation.InMemory  # port pins

    # Port is the sole holder: releasing the pin spills immediately.
    port2 = _Const(9).output_port('out')
    port2.set_data(PortData(9, 'ImageData'))
    port2.persistence_mode = PersistenceMode.OnDisk
    assert port2.has_data()
    assert port2.data_location() == DataLocation.OnDisk
    reloaded = port2.materialize()
    assert reloaded is not None and reloaded.data.payload == 9


def test_switch_on_disk_to_in_memory_loads_and_pins(_cache_dir):
    src, port = _on_disk_port(_cache_dir)
    handle = port.take()
    del handle
    gc.collect()
    assert port.data_location() == DataLocation.OnDisk
    assert len(_cache_files(_cache_dir)) == 1

    port.persistence_mode = PersistenceMode.InMemory
    assert port.has_data()
    assert port.data() is not None         # peek works: pinned in memory
    assert port.data().payload == 7
    assert _cache_files(_cache_dir) == []  # cache dropped


def test_switch_on_disk_to_transient_drops_data_and_cache(_cache_dir):
    src, port = _on_disk_port(_cache_dir)
    handle = port.take()
    del handle
    gc.collect()
    assert port.data_location() == DataLocation.OnDisk

    port.persistent = False
    assert not port.has_data()
    assert _cache_files(_cache_dir) == []


def test_mode_read_at_release_time_not_capture_time(_cache_dir):
    """A handle created while the port was InMemory must still spill if
    the port switched to OnDisk before the handle dropped."""
    src = _Const(7)
    port = src.output_port('out')
    src.execute()
    handle = port.take()                   # InMemory: port keeps its pin

    port.persistence_mode = PersistenceMode.OnDisk  # drops the pin
    assert port.data_location() == DataLocation.InMemory  # handle alive
    del handle
    gc.collect()
    assert port.data_location() == DataLocation.OnDisk


# ---- serialization -------------------------------------------------------


def test_serialize_persistence_mode_key():
    node = _Const()
    port = node.output_port('out')

    entry = node.serialize()['outputPorts']['out']
    assert entry['persistent'] is True
    assert 'persistenceMode' not in entry  # InMemory omits the key

    port.persistence_mode = PersistenceMode.OnDisk
    entry = node.serialize()['outputPorts']['out']
    assert entry == {'type': 'ImageData', 'persistent': True,
                     'persistenceMode': 'disk'}

    port.persistent = False
    entry = node.serialize()['outputPorts']['out']
    assert entry['persistent'] is False
    assert 'persistenceMode' not in entry  # transient omits the key


def test_persistence_round_trips_through_state_dict():
    from tomviz_pipeline.core import NodeFactory
    NodeFactory.register(_Const.type_name, _Const)
    NodeFactory.register(_Double.type_name, _Double)

    p, (src, t) = _chain(n_transforms=1)
    t.output_port('out').persistence_mode = PersistenceMode.OnDisk
    state = pipeline_to_state_dict(p)

    restored = pipeline_from_state_dict(state)
    port = restored.node_by_id(t.id).output_port('out')
    assert port.persistent is True
    assert port.persistence_mode == PersistenceMode.OnDisk


def test_deserialize_absent_mode_means_in_memory():
    node = _Const()
    node.output_port('out').persistence_mode = PersistenceMode.OnDisk
    node.deserialize({'outputPorts': {'out': {'type': 'ImageData',
                                              'persistent': True}}})
    # No persistenceMode key: an old-style file; mode stays as-is
    # (absence encodes "InMemory or transient" only for ports that were
    # never OnDisk — parity with C++ deserialize which leaves the mode
    # untouched).
    assert node.output_port('out').persistence_mode == PersistenceMode.OnDisk


# ---- transform default policy --------------------------------------------


def test_transform_default_policy_applied_at_add_output():
    settings = PipelineSettings.instance()

    settings.transform_persistence_default = \
        TransformPersistenceDefault.OnDisk
    t = _Double()
    port = t.output_port('out')
    assert port.persistent is True
    assert port.persistence_mode == PersistenceMode.OnDisk

    settings.transform_persistence_default = \
        TransformPersistenceDefault.Transient
    t2 = _Double()
    assert t2.output_port('out').persistent is False

    # Not retroactive: t's port is unaffected by the later change.
    assert port.persistent is True
    assert port.persistence_mode == PersistenceMode.OnDisk


def test_transform_default_does_not_apply_to_sources():
    settings = PipelineSettings.instance()
    settings.transform_persistence_default = \
        TransformPersistenceDefault.Transient
    src = _Const()
    port = src.output_port('out')
    assert port.persistent is True         # sources: persistent InMemory
    assert port.persistence_mode == PersistenceMode.InMemory


def test_explicit_persistence_args_beat_policy():
    settings = PipelineSettings.instance()
    settings.transform_persistence_default = \
        TransformPersistenceDefault.OnDisk

    t = TransformNode()
    port = t.add_output('x', 'ImageData', persistent=False)
    assert port.persistent is False

    t2 = TransformNode()
    port2 = t2.add_output('x', 'ImageData',
                          persistence_mode=PersistenceMode.InMemory)
    assert port2.persistent is True
    assert port2.persistence_mode == PersistenceMode.InMemory


def test_settings_change_signal():
    settings = PipelineSettings.instance()
    seen = []
    conn = settings.transform_persistence_default_changed.connect(
        seen.append)
    try:
        settings.transform_persistence_default = \
            TransformPersistenceDefault.Transient
        settings.transform_persistence_default = \
            TransformPersistenceDefault.Transient  # no-op: no signal
        assert seen == [TransformPersistenceDefault.Transient]
    finally:
        conn.disconnect()


# ---- legacy path ----------------------------------------------------------


def test_legacy_execute_leaves_everything_pinned():
    """The blocking legacy path (used by the CLI runner) does no
    residency management: even transient intermediates stay materialized
    so leaf writers and snapshots see everything."""
    from tomviz_pipeline import DefaultExecutor
    p, (src, t1, t2) = _chain()
    t1.output_port('out').persistent = False

    assert DefaultExecutor(p).execute() is True
    assert t1.output_port('out').has_data()
    assert t1.output_port('out').data().payload == 6
