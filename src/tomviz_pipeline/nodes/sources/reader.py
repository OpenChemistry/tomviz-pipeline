###############################################################################
# This source file is part of the tomviz-pipeline project.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""ReaderSourceNode — reads file(s) into a Dataset and exposes it on the
'volume' output port. Mirrors the C++ ReaderSourceNode."""

from __future__ import annotations

from pathlib import Path

from tomviz_pipeline.core import PortData, SourceNode
from tomviz_pipeline.io.emd import load_dataset


class ReaderSourceNode(SourceNode):
    type_name = 'source.reader'

    def __init__(self):
        super().__init__()
        self.add_output('volume', 'ImageData', persistent=True)
        self.label = 'Reader'
        self._parameters['fileNames'] = []
        self._parameters['readerOptions'] = {}

    # Attribute-style access kept for the runner's input-override
    # contract (runner assigns node.file_names directly).
    @property
    def file_names(self) -> list[str]:
        return self._parameters['fileNames']

    @file_names.setter
    def file_names(self, value):
        self._parameters['fileNames'] = list(value)

    @property
    def reader_options(self) -> dict:
        return self._parameters['readerOptions']

    @reader_options.setter
    def reader_options(self, value):
        self._parameters['readerOptions'] = dict(value)

    def serialize(self) -> dict:
        data = super().serialize()
        data['fileNames'] = list(self.file_names)
        if self.reader_options:
            data['readerOptions'] = dict(self.reader_options)
        return data

    def deserialize(self, data: dict) -> bool:
        if not super().deserialize(data):
            return False
        self._parameters['fileNames'] = list(data.get('fileNames', []))
        self._parameters['readerOptions'] = dict(
            data.get('readerOptions', {}))
        return True

    def execute(self) -> bool:
        if not self.file_names:
            return False

        # Translate ParaView reader-descriptor options into kwargs that
        # tomviz_pipeline.io.emd.load_dataset understands. Only the
        # options that actually affect the Python reader are honored.
        read_options = {}
        if 'subsampleSettings' in self.reader_options:
            read_options['subsampleSettings'] = (
                self.reader_options['subsampleSettings'])
        if self.reader_options.get('keepCOrdering'):
            read_options['keep_c_ordering'] = True

        # First file only — mirrors the C++ side: stack support belongs
        # to ParaView readers we don't replicate here.
        path = Path(self.file_names[0])
        if not path.is_absolute():
            # Resolve against the state file directory if possible.
            state_dir = getattr(self, '_state_dir', None)
            if state_dir is not None:
                path = (state_dir / path).resolve()

        dataset = load_dataset(path, read_options or None)

        port = self.output_port('volume')
        port_type = port.port_type if port is not None else 'ImageData'
        if dataset.tilt_angles is not None and port_type == 'ImageData':
            port_type = 'TiltSeries'
        if port is not None:
            port.port_type = port_type
            port.set_data(PortData(dataset, port_type))
        return True
