###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""tomviz-pipeline: a node-graph pipeline engine for volumetric data.

The generic engine lives in tomviz_pipeline.core (re-exported here). The
tomviz-flavored layer — Dataset, EMD/tvh5 I/O, built-in nodes, the batch
runner and the CLI — is imported lazily on first attribute access so that
core-only consumers don't pay for numpy/h5py."""

from .core import (  # noqa: F401
    AsyncioDispatcher,
    Connection,
    DataLocation,
    DefaultExecutor,
    EventQueue,
    ExecutionFuture,
    IMAGE_PORT_TYPES,
    InputPort,
    InternalNodeExecutor,
    Link,
    Node,
    NodeExecState,
    NodeExecutor,
    NodeExecutorFactory,
    NodeFactory,
    NodeState,
    OutputPort,
    PassthroughOutputPort,
    PersistenceMode,
    Pipeline,
    PipelineExecutor,
    PipelineSettings,
    Port,
    PortData,
    PortDataHandle,
    SCHEMA_VERSION,
    Signal,
    SinkGroupNode,
    SinkNode,
    SourceNode,
    ThreadedExecutor,
    TransformNode,
    TransformPersistenceDefault,
    is_port_type_compatible,
    pipeline_from_state_dict,
    pipeline_to_state_dict,
    set_port_data_serializer,
)

__version__ = '3.2.1'

# Lazily resolved tomviz-layer exports: name -> (module, attribute).
_LAZY = {
    'Dataset': ('tomviz_pipeline.dataset', 'Dataset'),
    'Table': ('tomviz_pipeline.table', 'Table'),
    'Molecule': ('tomviz_pipeline.molecule', 'Molecule'),
    'make_spreadsheet': ('tomviz_pipeline.utils', 'make_spreadsheet'),
    'make_molecule': ('tomviz_pipeline.utils', 'make_molecule'),
    'ExternalNodeExecutor': ('tomviz_pipeline.external',
                             'ExternalNodeExecutor'),
    'Kernel': ('tomviz_pipeline.kernels', 'Kernel'),
    'SourceKernel': ('tomviz_pipeline.kernels', 'SourceKernel'),
    'TransformKernel': ('tomviz_pipeline.kernels', 'TransformKernel'),
    'load_state': ('tomviz_pipeline.state', 'load_state'),
    'ScriptableNode': ('tomviz_pipeline.nodes.scriptable', 'ScriptableNode'),
    'ScriptableSourceNode': ('tomviz_pipeline.nodes.sources.scriptable',
                             'ScriptableSourceNode'),
    'ScriptableTransformNode': ('tomviz_pipeline.nodes.transforms.scriptable',
                                'ScriptableTransformNode'),
    'LegacyScriptableTransformNode': (
        'tomviz_pipeline.nodes.transforms.legacy_scriptable',
        'LegacyScriptableTransformNode'),
    'definition_schema': ('tomviz_pipeline.nodes.kernel_backends',
                          'definition_schema'),
    # Former name of ScriptableNode.
    'PythonNode': ('tomviz_pipeline.nodes.scriptable', 'ScriptableNode'),
    'read_state_json': ('tomviz_pipeline.state', 'read_state_json'),
    'write_state_tvh5': ('tomviz_pipeline.state', 'write_state_tvh5'),
    'save_state': ('tomviz_pipeline.state', 'save_state'),
    'build_state': ('tomviz_pipeline.state', 'build_state'),
    'write_state': ('tomviz_pipeline.state', 'write_state'),
    'register_builtins': ('tomviz_pipeline.nodes', 'register_builtins'),
    'run': ('tomviz_pipeline.runner', 'run'),
}


def __getattr__(name):
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(
            f'module {__name__!r} has no attribute {name!r}')
    import importlib
    module = importlib.import_module(entry[0])
    value = getattr(module, entry[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
