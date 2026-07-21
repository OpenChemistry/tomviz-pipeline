# tomviz-pipeline

A pure-Python node-graph pipeline engine for processing volumetric /
tomographic data. It originated as the external-execution runtime of
[tomviz](https://tomviz.org) and is being developed as a standalone
library so that other applications can build, run, and persist data
processing pipelines without depending on the tomviz application, Qt,
or VTK.

## Layout

The package is layered:

- `tomviz_pipeline.core` — a generic, dependency-free (stdlib only)
  pipeline engine: `Pipeline`, `Node` (`SourceNode` / `TransformNode` /
  `SinkNode`), typed `InputPort` / `OutputPort`, `Link`, `PortData`,
  dirty tracking (`New` / `Stale` / `Current`), execution planning,
  blocking (`DefaultExecutor`) and threaded (`ThreadedExecutor`)
  pipeline executors, `ExecutionFuture`, thread-safe `Signal` /
  `EventQueue` observers, and a per-node `NodeExecutor` abstraction
  (in-process vs. out-of-process node execution).
- The rest of `tomviz_pipeline` — the tomviz-flavored layer: the numpy
  `Dataset` model (plus pure-Python `Table` and `Molecule` payloads for
  the corresponding port types), EMD/HDF5 I/O, schema-v2 state files
  (`.tvsm` JSON and `.tvh5` HDF5 containers), the built-in
  source/transform/sink nodes,
  Python operator authoring APIs, progress reporting channels, the
  `ExternalNodeExecutor` (run a node under a different Python
  environment), the batch runner, and the `tomviz-pipeline` CLI.

Applications that just need a pipeline engine can depend on
`tomviz_pipeline.core` alone; tomviz uses the whole package.

## Installation

```bash
pip install tomviz-pipeline
```

Requires Python 3.9+. The runtime dependencies are numpy, h5py, click
and tqdm. The optional `tomviz-pipeline[vtk]` extra lets state files
carry tables built directly with VTK by operator scripts; operator
scripts bring their own scientific dependencies (scipy, tomopy, ...).

## Quick start

A python node is built from exactly two artifacts: a **definition**
(the interface — ports, parameters, label; the same JSON vocabulary as
tomviz's operator `.json` sidecar files) and a **kernel** (the compute —
the same class a tomviz operator script contains, so anyone who has
written a tomviz Python operator already knows how):

```python
import numpy as np

from tomviz_pipeline import Pipeline, PythonNode
from tomviz_pipeline.dataset import Dataset
from tomviz_pipeline.kernels import SourceKernel, TransformKernel

class ConstantVolume(SourceKernel):
    def produce(self, value=1.0, side=32):
        arr = np.full((side, side, side), value, dtype=np.float32)
        return {'volume': Dataset({'Scalars': arr}, active='Scalars')}

class Multiply(TransformKernel):
    def transform(self, inputs, factor=2.0):
        ds = inputs['volume']
        return {'volume': ds.apply_to_each_scalar_array(
            lambda a: a * factor)}

CONSTANT_VOLUME = {
    'name': 'ConstantVolume',
    'outputs': [{'name': 'volume', 'type': 'ImageData'}],
    'parameters': [{'name': 'value', 'type': 'double', 'default': 1.0},
                   {'name': 'side', 'type': 'int', 'default': 32}],
}

MULTIPLY = {
    'name': 'Multiply',
    'inputs':  [{'name': 'volume', 'type': 'ImageData'}],
    'outputs': [{'name': 'volume', 'type': 'ImageData'}],
    'parameters': [{'name': 'factor', 'type': 'double', 'default': 2.0}],
}

pipeline = Pipeline()
source = pipeline.add_node(PythonNode(CONSTANT_VOLUME,
                                      kernel=ConstantVolume))
scale = pipeline.add_node(PythonNode(MULTIPLY, kernel=Multiply))
pipeline.create_link(source.output_port('volume'),
                     scale.input_port('volume'))
source.set_parameters(value=21.0)

pipeline.execute()
print(scale.output_port('volume').data().payload.active_scalars.max())
# 42.0
```

Both constructor arguments are polymorphic, with one rule: **strings
are content, paths are files.**

- `definition`: a dict, a JSON string, or a `pathlib.Path` to a
  `.json` file. The definition decides the node shape — no `inputs`
  means a `source.python` node, otherwise `transform.python`.
- `kernel`: a kernel class you hold, the source text of an operator
  script, or a `pathlib.Path` to a `.py` file.

That makes the classic sidecar pair — and kernels whose imports only
resolve in *another* Python environment — work without ever importing
the kernel here:

```python
from pathlib import Path
from tomviz_pipeline import ExternalNodeExecutor

recon = pipeline.add_node(PythonNode(Path('operators/Reconstruct.json'),
                                     kernel=Path('operators/Reconstruct.py')))
recon.set_parameters(iterations=100)
recon.node_executor = ExternalNodeExecutor('/envs/tomopy')
```

Notes on the kernel side:

- Kernels get `self.progress`, `self.canceled`, and `self.completed`
  for long-running work; a kernel sees deep-copied datasets and keyword
  parameters — no graph machinery — which is what lets the same class
  run unchanged inside the tomviz application, in this standalone
  runtime, and in an external environment.
- A class-bound kernel executes in-process directly. Whenever the node
  must be serialized (state files, external execution) the class is
  re-expressed as a script by capturing its source; for that the class
  body must be self-contained apart from numpy / `tomviz_pipeline`
  imports. Kernels given as script text or a `.py` path serialize
  as-is.
- Kernels also get `self.state`: a dict the engine preserves across
  executions of the node (each run gets a fresh instance, so plain
  attributes don't survive). It round-trips through external execution
  but is never written to state files. Keep its values JSON-friendly.
- A kernel may implement `should_auto_execute(self, **parameters) ->
  bool` for periodic execution: when `node.auto_execute_enabled` is set
  an application polls `executor.should_auto_execute(node)` every
  `node.auto_execute_interval_seconds` — through the node's executor,
  so an external node is asked inside its own environment — and
  re-executes the pipeline on `True`. The hook receives the node's
  parameters and shares `self.state` with `produce` / `transform`.
- Older operator scripts that import `tomviz.nodes` (the historical
  spelling, `tomviz.nodes.TransformNode`) keep working: those names
  resolve to the kernel classes through a compatibility alias.

## Parameters

Node configuration lives in a private parameters store (so names can
never collide with node internals like `label` or `state`) and is
changed through `set_parameters()`, which marks the node and everything
downstream stale and emits `parameters_applied`:

```python
scale.set_parameters(factor=3)      # scale (and downstream) now Stale
pipeline.execute()                  # re-runs just what's needed

pipeline.auto_execute = True        # optional: C++-style behavior where
scale.set_parameters(factor=4)      # applying parameters re-executes
```

## Using it from an application

A pipeline embedded in an application must not block the UI thread.
Install a `ThreadedExecutor` once and every `execute()` variant returns
immediately with an `ExecutionFuture`; the plan runs on a worker
thread.

Notifications are `Signal`s. A plain `connect(handler)` is a *direct*
connection — the handler runs on whichever thread emits, which during
execution is the worker, so it must be thread-safe and must not touch
UI. Connecting through an `EventQueue` defers delivery: emission only
enqueues, and the handler runs when your UI thread drains the queue —
the moral equivalent of Qt's queued connections, without Qt. (The
optional second argument to `connect()` is a *dispatcher* — `EventQueue`
here, `AsyncioDispatcher` for event-loop apps below.)

```python
from tomviz_pipeline import EventQueue, ThreadedExecutor

# Fresh pipeline with the ConstantVolume / Multiply kernels from the
# quick start.
pipeline = Pipeline()
source = pipeline.add_node(PythonNode(CONSTANT_VOLUME,
                                      kernel=ConstantVolume))
scale = pipeline.add_node(PythonNode(MULTIPLY, kernel=Multiply))
pipeline.create_link(source.output_port('volume'),
                     scale.input_port('volume'))

pipeline.set_executor(ThreadedExecutor())

# Everything below runs on the thread that drains `events` — never on
# the worker.
events = EventQueue()
pipeline.executor.node_execution_started.connect(
    lambda node: print(f'started:  {node.label}'), events)
pipeline.executor.node_execution_finished.connect(
    lambda node, ok: print(f'finished: {node.label} ok={ok}'), events)
scale.progress_step_changed.connect(
    lambda node, step: print(f'progress: {step}'), events)
pipeline.execution_finished.connect(
    lambda future: print(f'done, succeeded={future.succeeded()}'), events)

future = pipeline.execute()          # returns immediately

# A real application drains the queue from its main loop (see the Qt
# sketch below). A minimal stand-in loop:
while not future.is_finished():
    events.process(block=True, timeout=0.05)
events.process()                     # deliver anything queued after finish
```

Interactive control, all cooperative and non-blocking:

```python
future = pipeline.execute()          # user starts a run...
pipeline.cancel_execution()          # ...and changes their mind: stops at
future.wait(10)                      # the next node boundary; running
                                     # kernels observe self.canceled

pipeline.set_paused(True)            # batch edits without re-running
scale.set_parameters(factor=3.0)
pipeline.set_paused(False)           # un-pausing re-executes what's stale

pipeline.auto_execute = True         # or: every set_parameters() re-runs,
scale.set_parameters(factor=4.0)     # as in the tomviz application

pipeline.executor.cancel_and_wait()  # teardown: join the worker.
                                     # pipeline.clear() does this for you.
```

Calling `execute()` while a run is in flight never queues behind it:
the new plan becomes *pending*, the in-flight run is canceled at its
next node boundary, and the pending plan starts when the worker exits —
the right semantics for "the user dragged the slider again".

In a Qt application the drain is a timer on the GUI thread
(illustrative sketch):

```python
class PipelinePanel(QWidget):
    def __init__(self, pipeline):
        ...
        self.events = EventQueue()
        pipeline.set_executor(ThreadedExecutor(sync_queue=self.events))
        pipeline.execution_finished.connect(self._on_finished,
                                            self.events)
        self._timer = QTimer(self, interval=16)   # ~60 Hz
        self._timer.timeout.connect(self.events.process)
        self._timer.start()
```

The optional `sync_queue` makes the worker wait, after each node, until
the application thread has drained the queue — use it when per-node UI
updates (e.g. re-rendering) must complete before the next node starts,
mirroring the C++ tomviz executor's barrier.

In an asyncio application there is no drain timer to write: an
`AsyncioDispatcher` plays the `EventQueue`'s role, delivering handlers
onto the event loop. Create it once, on the loop, and pass it to any
number of `connect()` calls; plain callbacks run as loop callbacks, and
coroutine-function handlers are scheduled as tasks. Await a run's
completion by parking `future.wait()` on a thread-pool thread:

```python
import asyncio

from tomviz_pipeline import AsyncioDispatcher

async def run_pipeline(pipeline):
    dispatcher = AsyncioDispatcher()        # bound to this event loop

    connection = pipeline.executor.node_execution_finished.connect(
        lambda node, ok: print(f'finished: {node.label} ok={ok}'),
        dispatcher)

    future = pipeline.execute()             # returns immediately
    await asyncio.to_thread(future.wait)    # yields until the run ends
    connection.disconnect()
    return future.succeeded()

scale.set_parameters(factor=5.0)
assert asyncio.run(run_pipeline(pipeline)) is True
```

Cancelling an asyncio task that is awaiting a run does not cancel the
pipeline — awaiting is observation, not ownership. Couple them
explicitly when you want to:

```python
async def run_and_own(pipeline):
    future = pipeline.execute()
    try:
        await asyncio.to_thread(future.wait)
    except asyncio.CancelledError:
        pipeline.cancel_execution()   # propagate deliberately
        raise
```

## Output persistence

Each output port is either *transient* (`port.persistent = False` — its
payload lives only while some consumer holds a handle; the planner
re-runs the producer when the data is needed again) or *persistent*, in
which case `port.persistence_mode` picks the medium: `InMemory` pins the
payload on the port, `OnDisk` spills it to a temp cache file
(`$TOMVIZ_PORT_CACHE_DIR` or the system temp dir) once the last handle
drops and reloads it lazily via `port.materialize()`. `port.data()` is a
non-loading peek. Sources default to persistent-InMemory; transform
outputs follow the pipeline-wide default:

```python
from tomviz_pipeline import PipelineSettings, TransformPersistenceDefault

PipelineSettings.instance().transform_persistence_default = \
    TransformPersistenceDefault.OnDisk   # keep intermediates out of RAM
```

Modes can be switched at runtime and round-trip through schema-v2 state
files (`persistent` / `persistenceMode` keys, identical to the C++
implementation).

## Graph nodes (application integration)

Kernels cover custom *computation*. Applications embedding the engine
that need custom *graph machinery* — their own port layouts, payload
types, serialization, or state handling — subclass the core graph
classes directly and register them under their own type strings:

```python
from tomviz_pipeline import (
    DefaultExecutor, NodeFactory, Pipeline, PortData, SourceNode,
    TransformNode,
)

class Constant(SourceNode):
    type_name = 'example.constant'

    def __init__(self, value=0):
        super().__init__()
        self._parameters['value'] = value
        self.add_output('output', 'ImageData')

    def execute(self):
        self.output_port('output').set_data(
            PortData(self.parameter('value'), 'ImageData'))
        return True

class Scale(TransformNode):
    type_name = 'example.scale'

    def __init__(self, factor=1):
        super().__init__()
        self._parameters['factor'] = factor
        self.add_input('input', 'ImageData')
        self.add_output('output', 'ImageData')

    def transform(self, inputs):
        value = inputs['input'].payload * self.parameter('factor')
        return {'output': PortData(value, 'ImageData')}

NodeFactory.register('example.constant', Constant)
NodeFactory.register('example.scale', Scale)

pipeline = Pipeline()
source = pipeline.add_node(Constant(21))
scale = pipeline.add_node(Scale(2))
pipeline.create_link(source.output_port('output'),
                     scale.input_port('input'))
DefaultExecutor(pipeline).execute()
print(scale.output_port('output').data().payload)  # 42
```

This is the layer the built-in tomviz node types (readers, crops,
reconstructions, ...) are made of. Note the trade-off versus kernels:
graph nodes speak `PortData` and manage their own ports, but only
deserialize in processes where the application's classes are importable
— a kernel-hosted node round-trips anywhere because it travels with its
source.

## Running a state file

```bash
tomviz-pipeline -s pipeline.tvsm -o output/
tomviz-pipeline -s pipeline.tvsm -o output/ --input 'data/*.emd'
```

## Development

```bash
git clone https://github.com/OpenChemistry/tomviz-pipeline
cd tomviz-pipeline
pip install -e .[dev]
flake8 --config setup.cfg src tests
pytest
```

Releases are built and uploaded to PyPI by the `Publish to PyPI`
workflow when a GitHub release is published; bump `__version__` in
`src/tomviz_pipeline/__init__.py` (the single source of the package
version) and tag `v<version>`.

## License

BSD 3-Clause. See `LICENSE`.
