# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence, final

from fairseq2 import DOC_MODE
from fairseq2.data.string import StringLike
from fairseq2.data.tape import Tape


@final
class DataPipeline:
    def __init__(self) -> None:
        pass

    def __iter__(self) -> Iterator[Any]:
        pass

    def skip(self, num_examples: int) -> int:
        pass

    def reset(self) -> None:
        pass

    def record_position(self, t: Tape) -> None:
        pass

    def reload_position(self, t: Tape) -> None:
        pass

    @property
    def is_broken(self) -> bool:
        pass


class DataProcessor:
    def __call_(self, data: Any) -> Any:
        pass


@final
class DataPipelineBuilder:
    def batch(
        self, batch_size: int, drop_remainder: bool = False
    ) -> "DataPipelineBuilder":
        pass

    def map(self, fn: Callable[[Any], Any]) -> "DataPipelineBuilder":
        pass

    def map_with_processor(self, dp: DataProcessor) -> "DataPipelineBuilder":
        pass

    def shard(self, shard_idx: int, num_shards: int) -> "DataPipelineBuilder":
        pass

    def yield_from(self, fn: Callable[[Any], DataPipeline]) -> "DataPipelineBuilder":
        pass

    def and_return(self) -> DataPipeline:
        pass


class DataPipelineError(RuntimeError):
    pass


def list_files(pathname: StringLike, pattern: StringLike = "") -> "DataPipelineBuilder":
    pass


def read_sequence(s: Sequence[Any]) -> "DataPipelineBuilder":
    pass


def zip_data_pipelines(
    data_pipelines: Sequence[DataPipeline],
) -> "DataPipelineBuilder":
    pass


class RecordError(RuntimeError):
    pass


class StreamError(RuntimeError):
    pass


if not TYPE_CHECKING and not DOC_MODE:
    from fairseq2._C.data.data_pipeline import (  # noqa: F811
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
