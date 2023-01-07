# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from typing import Any, Iterator, final

from fairseq2.data.tape import Tape
from fairseq2.data.typing import StringLike

# fmt: off

@final
class DataPipeline:
    def __init__(self) -> None:
        ...

    def __iter__(self) -> Iterator[Any]:
        ...

    def skip(self, num_examples: int) -> int:
        ...

    def reset(self) -> None:
        ...

    def record_position(self, t: Tape) -> None:
        ...

    def reload_position(self, t: Tape) -> None:
        ...

    @property
    def is_broken(self) -> bool:
        ...


@final
class DataPipelineBuilder:
    def batch(
        self, batch_size: int, drop_remainder: bool = False
    ) -> DataPipelineBuilder:
        ...

    def map(self, fn: Callable[[Any], Any]) -> DataPipelineBuilder:
        ...

    def map_with_processor(self, dp: DataProcessor) -> DataPipelineBuilder:
        ...

    def shard(self, shard_idx: int, num_shards: int) -> DataPipelineBuilder:
        ...

    def yield_from(self, fn: Callable[[Any], DataPipeline]) -> DataPipelineBuilder:
        ...

    def and_return(self) -> DataPipeline:
        ...


class DataProcessor:
    def __call_(self, data: Any) -> Any:
        ...


class DataPipelineError(RuntimeError):
    ...


def list_files(pathname: StringLike, pattern: StringLike = "") -> DataPipelineBuilder:
    ...


def read_sequence(s: Sequence[Any]) -> DataPipelineBuilder:
    ...


def zip_data_pipelines(data_pipelines: Sequence[DataPipeline]) -> DataPipelineBuilder:
    ...


class RecordError(RuntimeError):
    ...


class StreamError(RuntimeError):
    ...
