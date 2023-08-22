# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

from torch import Tensor
from typing_extensions import Self

from fairseq2 import _DOC_MODE
from fairseq2.data.typing import PathLike, StringLike
from fairseq2.memory import MemoryBlock

if TYPE_CHECKING or _DOC_MODE:

    class DataPipeline:
        def __iter__(self) -> Iterator[Any]:
            """Return an iterator over the examples in the data pipeline."""

        def reset(self) -> None:
            """Move back to the first example in the data pipeline."""

        @property
        def is_broken(self) -> bool:
            """Return ``True`` if the data pipeline is broken.

            If ``True``, any future operation on this data pipeline will raise a
            :class:`DataPipelineError`.
            """

        def state_dict(self) -> Dict[str, Any]:
            """Return a dictionary containing the state of the data pipeline.

            The current position of the data pipeline can be restored by passing
            the returned state dictionary to :meth:`load_state_dict`.
            """

        def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True
        ) -> None:
            """Restore the state of the data pipeline from ``state_dict``.

            :param state_dict:
                A state dictionary previously returned by :meth:`state_dict`.
            :param strict:
                If ``True``, enforces that the keys in ``state_dict`` match the
                keys returned by :meth:`state_dict`.
            """

        @staticmethod
        def zip(
            pipelines: Sequence["DataPipeline"],
            names: Optional[Sequence[str]] = None,
            zip_to_shortest: bool = False,
            flatten: bool = False,
            disable_parallelism: bool = False,
        ) -> "DataPipelineBuilder":
            """Zip together examples read from ``pipelines``.

            :param pipelines:
                The data pipelines to zip.
            :param names:
                The names to assign to the data pipelines.
            :param flatten:
            :param disable_parallelism:
                If ``True``, calls each data pipeline sequentially.
            """

        @staticmethod
        def round_robin(pipelines: Sequence["DataPipeline"]) -> "DataPipelineBuilder":
            """Extract examples from ``pipelines`` in round robin.

            :param pipelines:
                The data pipelines to round robin.
            """

        @staticmethod
        def constant(example: Any, key: Optional[str] = None) -> "DataPipelineBuilder":
            ...

        @staticmethod
        def count(start: int = 0, key: Optional[str] = None) -> "DataPipelineBuilder":
            ...

    class DataPipelineBuilder:
        def bucket(self, bucket_size: int, drop_remainder: bool = False) -> Self:
            """Combine a number of consecutive examples into a single example.

            :param bucket_size:
                The number of examples to combine.
            :param drop_remainder:
                If ``True``, drops the last bucket in case it has fewer than
                ``bucket_size`` examples.
            """

        def bucket_by_length(
            self,
            bucket_sizes: Sequence[Tuple[int, int]],
            selector: Optional[str] = None,
            drop_remainder: bool = False,
        ) -> Self:
            """Combine examples of similar shape into batches."""

        def filter(self, predicate: Callable[[Any], Any]) -> Self:
            """Filter examples from data pipeline and keep only those who match
            ``predicate``.

            :param predicate:
                The predicate used to select examples to keep.
            """

        def map(
            self,
            fn: Union[Callable[[Any], Any], Sequence[Callable[[Any], Any]]],
            selector: Optional[str] = None,
            num_parallel_calls: int = 1,
        ) -> Self:
            """Apply ``fn`` to every example.

            :param fn:
                The function to apply.
            :param selector:
            :param num_parallel_calls:
                The number of examples to process in parallel.
            """

        def prefetch(self, num_examples: int) -> Self:
            """Prefetch examples in the background while the current example is
            being processed.

            :param num_examples:
                The number of examples to prefetch.
            """

        def shard(self, shard_idx: int, num_shards: int) -> Self:
            """Read only 1/``num_shards`` of the examples in the data pipeline.

            :param shard_idx:
                The shard index.
            :param num_shards:
                The number of shards.
            """

        def shuffle(
            self, shuffle_window: int, strict: bool = True, enabled: bool = True
        ) -> Self:
            """Shuffle examples using a fixed sized buffer.

            :param shuffle_window:
                The size of the intermediate buffer used for shuffling. Examples
                will be randomly sampled from this buffer, and selected examples
                will be replaced with new examples. If ``0``, all examples will
                be loaded into memory for full shuffling.
            :param strict:
                If ``True``, the intermediate shuffle buffer will be saved as
                part of ``state_dict``. This ensures that on preemption no
                example will be lost, but for large buffers this can
                significantly increase the state size and the time to restore
                the data pipeline.
            :param enabled:
                If ``False``, disables shuffling.
            """

        def skip(self, num_examples: int) -> Self:
            """Skip ``num_examples`` examples."""

        def take(self, num_examples: int) -> Self:
            """Return at most ``num_examples`` examples."""

        def yield_from(self, fn: Callable[[Any], DataPipeline]) -> Self:
            """
            Map every example to a data pipeline and yield the examples returned
            from the mapped data pipelines.

            :param fn:
                The function to map examples to data pipelines.
            """

        def and_return(self, max_num_warnings: int = 0) -> DataPipeline:
            """Return a new :class:`DataPipeline` instance."""

    class DataPipelineError(RuntimeError):
        """Raised when an error occurs while reading from a data pipeline."""

    def get_last_failed_example() -> Any:
        ...

    def list_files(
        pathname: PathLike, pattern: Optional[StringLike] = None
    ) -> "DataPipelineBuilder":
        """List recursively all files under ``pathname`` that matches ``pattern``.

        :param pathname:
            The path to traverse.
        :param pattern:
            If non-empty, a pattern that follows the syntax of :mod:`fnmatch`.
        """

    def read_sequence(seq: Sequence[Any]) -> "DataPipelineBuilder":
        """Read every element in ``seq``.

        :param seq:
            The sequence to read.
        """

    def read_zipped_records(pathname: PathLike) -> DataPipelineBuilder:
        ...

    class CollateOptionsOverride:
        def __init__(
            self,
            selector: str,
            pad_idx: Optional[int] = None,
            pad_to_multiple: int = 1,
        ) -> None:
            ...

        @property
        def selector(self) -> str:
            ...

        @property
        def pad_idx(self) -> Optional[int]:
            ...

        @property
        def pad_to_multiple(self) -> int:
            ...

    class Collater:
        def __init__(
            self,
            pad_idx: Optional[int] = None,
            pad_to_multiple: int = 1,
            overrides: Optional[Sequence[CollateOptionsOverride]] = None,
        ) -> None:
            ...

        def __call__(self, data: Any) -> Any:
            ...

    class FileMapper:
        def __init__(
            self,
            root_dir: Optional[PathLike] = None,
            cached_fd_count: Optional[int] = None,
        ) -> None:
            ...

        def __call__(self, pathname: PathLike) -> "FileMapperOutput":
            ...

    class ByteStreamError(RuntimeError):
        """Raised when a dataset cannot be read."""

    class RecordError(RuntimeError):
        """Raised when a corrupt record is encountered while reading a dataset."""

else:
    from fairseq2n.bindings.data.data_pipeline import ByteStreamError as ByteStreamError
    from fairseq2n.bindings.data.data_pipeline import (
        CollateOptionsOverride as CollateOptionsOverride,
    )
    from fairseq2n.bindings.data.data_pipeline import Collater as Collater
    from fairseq2n.bindings.data.data_pipeline import DataPipeline as DataPipeline
    from fairseq2n.bindings.data.data_pipeline import (
        DataPipelineBuilder as DataPipelineBuilder,
    )
    from fairseq2n.bindings.data.data_pipeline import (
        DataPipelineError as DataPipelineError,
    )
    from fairseq2n.bindings.data.data_pipeline import FileMapper as FileMapper
    from fairseq2n.bindings.data.data_pipeline import RecordError as RecordError
    from fairseq2n.bindings.data.data_pipeline import (
        get_last_failed_example as get_last_failed_example,
    )
    from fairseq2n.bindings.data.data_pipeline import list_files as list_files
    from fairseq2n.bindings.data.data_pipeline import read_sequence as read_sequence
    from fairseq2n.bindings.data.data_pipeline import (
        read_zipped_records as read_zipped_records,
    )

    def _set_module_name() -> None:
        ctypes = [
            ByteStreamError,
            CollateOptionsOverride,
            Collater,
            DataPipeline,
            DataPipelineBuilder,
            DataPipelineError,
            FileMapper,
            RecordError,
            get_last_failed_example,
            list_files,
            read_sequence,
            read_zipped_records,
        ]

        for t in ctypes:
            t.__module__ = __name__

    _set_module_name()


class SequenceData(TypedDict):
    seqs: Tensor
    seq_lens: Tensor
    is_ragged: bool


class FileMapperOutput(TypedDict):
    path: PathLike
    data: MemoryBlock
