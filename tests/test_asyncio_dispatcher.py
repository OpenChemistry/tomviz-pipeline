###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""AsyncioDispatcher: delivering signal handlers onto an asyncio event
loop from worker threads."""

import asyncio
import logging
import threading

import pytest

from tomviz_pipeline.core import (
    AsyncioDispatcher,
    Pipeline,
    PortData,
    Signal,
    SourceNode,
    ThreadedExecutor,
    TransformNode,
)


def test_requires_running_loop():
    with pytest.raises(RuntimeError):
        AsyncioDispatcher()


def test_handler_runs_on_loop_thread():
    async def main():
        dispatcher = AsyncioDispatcher()
        signal = Signal('s')
        delivered = asyncio.Event()
        seen = []

        def handler(value):
            seen.append((value, threading.get_ident()))
            delivered.set()

        signal.connect(handler, dispatcher)
        worker = threading.Thread(target=signal.emit, args=(42,))
        worker.start()
        worker.join()
        await asyncio.wait_for(delivered.wait(), 10)
        assert seen == [(42, threading.get_ident())]

    asyncio.run(main())


def test_coroutine_handler_runs_as_task():
    async def main():
        dispatcher = AsyncioDispatcher()
        signal = Signal('s')
        delivered = asyncio.Event()
        seen = []

        async def handler(value):
            await asyncio.sleep(0)
            seen.append((value, threading.get_ident()))
            delivered.set()

        signal.connect(handler, dispatcher)
        worker = threading.Thread(target=signal.emit, args=('x',))
        worker.start()
        worker.join()
        await asyncio.wait_for(delivered.wait(), 10)
        assert seen == [('x', threading.get_ident())]

    asyncio.run(main())


def test_handler_exception_logged_other_handlers_survive(caplog):
    async def main():
        dispatcher = AsyncioDispatcher()
        signal = Signal('s')
        delivered = asyncio.Event()

        signal.connect(lambda: 1 / 0, dispatcher)
        signal.connect(lambda: delivered.set(), dispatcher)
        signal.emit()          # same-thread emission also goes via loop
        await asyncio.wait_for(delivered.wait(), 10)

    with caplog.at_level(logging.ERROR, logger='tomviz_pipeline'):
        asyncio.run(main())
    assert any('asyncio signal handler' in r.message for r in caplog.records)


def test_async_handler_exception_logged(caplog):
    async def main():
        dispatcher = AsyncioDispatcher()
        signal = Signal('s')

        async def boom():
            raise RuntimeError('intentional')

        signal.connect(boom, dispatcher)
        signal.emit()
        # Let the delivery callback and the task both run.
        await asyncio.sleep(0.05)

    with caplog.at_level(logging.ERROR, logger='tomviz_pipeline'):
        asyncio.run(main())
    assert any('async signal handler' in r.message for r in caplog.records)


def test_disconnect_before_delivery_suppresses():
    async def main():
        dispatcher = AsyncioDispatcher()
        signal = Signal('s')
        seen = []
        connection = signal.connect(seen.append, dispatcher)
        signal.emit(1)             # queued onto the loop, not yet run
        connection.disconnect()
        await asyncio.sleep(0.05)  # let the delivery callback fire
        assert seen == []

    asyncio.run(main())


def test_explicit_loop_allows_off_loop_construction():
    async def main():
        loop = asyncio.get_running_loop()
        delivered = asyncio.Event()
        seen = []
        signal = Signal('s')

        def build_and_emit():
            dispatcher = AsyncioDispatcher(loop=loop)
            signal.connect(lambda v: (seen.append(v), delivered.set()),
                           dispatcher)
            signal.emit('from-worker')

        worker = threading.Thread(target=build_and_emit)
        worker.start()
        worker.join()
        await asyncio.wait_for(delivered.wait(), 10)
        assert seen == ['from-worker']

    asyncio.run(main())


def test_closed_loop_drops_event():
    signal = Signal('s')
    holder = {}

    async def main():
        holder['dispatcher'] = AsyncioDispatcher()

    asyncio.run(main())            # loop is closed afterwards
    signal.connect(lambda: None, holder['dispatcher'])
    signal.emit()                  # must not raise


class _Const(SourceNode):
    type_name = 'test.const'

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.add_output('output', 'ImageData')

    def execute(self):
        self.output_port('output').set_data(PortData(self.value))
        return True


class _Double(TransformNode):
    type_name = 'test.double'

    def __init__(self):
        super().__init__()
        self.add_input('input', 'ImageData')
        self.add_output('output', 'ImageData')

    def transform(self, inputs):
        return {'output': PortData(inputs['input'].payload * 2)}


def test_pipeline_integration():
    """End to end: ThreadedExecutor emits from its worker; handlers
    connected through the dispatcher observe every node completion on
    the loop thread; the run is awaited without blocking the loop."""

    async def main():
        pipeline = Pipeline()
        source = pipeline.add_node(_Const(21))
        double = pipeline.add_node(_Double())
        pipeline.create_link(source.output_port('output'),
                             double.input_port('input'))
        pipeline.set_executor(ThreadedExecutor())

        dispatcher = AsyncioDispatcher()
        loop_thread = threading.get_ident()
        finished = []
        pipeline.executor.node_execution_finished.connect(
            lambda node, ok: finished.append(
                (node.type_name, ok, threading.get_ident())),
            dispatcher)

        future = pipeline.execute()
        await asyncio.wait_for(asyncio.to_thread(future.wait), 30)
        await asyncio.sleep(0.05)   # drain deliveries queued before finish

        assert future.succeeded()
        assert double.output_port('output').data().payload == 42
        assert finished == [('test.const', True, loop_thread),
                            ('test.double', True, loop_thread)]

    asyncio.run(main())
