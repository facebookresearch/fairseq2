# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "DataPipeline",
    "DataPipelineBuilder",
    "DataPipelineError",
    "DataProcessor",
    "RecordError",
    "StreamError",
    "list_files",
    "read_sequence",
    "zip_data_pipelines",
]

from fairseq2._C.data.data_pipeline import (
    DataPipeline,
    DataPipelineBuilder,
    DataPipelineError,
    DataProcessor,
    RecordError,
    StreamError,
    list_files,
    read_sequence,
    zip_data_pipelines,
)


def _set_module() -> None:
    ctypes = [
        DataPipeline,
        DataPipelineBuilder,
        DataPipelineError,
        DataProcessor,
        RecordError,
        StreamError,
        list_files,
        read_sequence,
        zip_data_pipelines,
    ]

    for t in ctypes:
        t.__module__ = __name__


_set_module()
