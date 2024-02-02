# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

from fairseq2n import DOC_MODE
from torch import Tensor
from typing_extensions import Self

from fairseq2.data.typing import PathLike, StringLike
from fairseq2.memory import MemoryBlock

if TYPE_CHECKING or DOC_MODE:

    class DataPipeline(Iterable[Any]):
        """fairseq2 native data pipeline.

        The pipeline state can be persisted to the disk, allowing it to be resumed later.
        It is a Python Iterable, but it also contains the iterator states.

        Calling `iter` twice will create two iterators reading from the same dataloader,
        and sharing the same state, so it will behave inconcistently.
        """

        def __iter__(self) -> Iterator[Any]:
            """Return an iterator over the examples in the data pipeline.

            The iterator will modify the internal state of the this DataPipeline,
            so it's not safe to have several iterators over the same DataPipeline.
            """

        def reset(self) -> None:
            """Move back to the first example in the data pipeline."""

        def is_infinite(self) -> bool:
            ...

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

        def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
            """Restore the state of the data pipeline from ``state_dict``.

            :param state_dict:
                A state dictionary previously returned by :meth:`state_dict`.
            """

        @staticmethod
        def concat(pipelines: Sequence[DataPipeline]) -> DataPipelineBuilder:
            """Concatenate examples from ``pipelines``.

            :param pipelines:
                The data pipelines to concatenate.
            """

        @staticmethod
        def constant(example: Any, key: Optional[str] = None) -> DataPipelineBuilder:
            ...

        @staticmethod
        def count(
            start: int = 0, step: int = 1, key: Optional[str] = None
        ) -> DataPipelineBuilder:
            ...

        @staticmethod
        def round_robin(
            pipelines: Sequence[DataPipeline], stop_at_shortest: bool = False
        ) -> DataPipelineBuilder:
            """Extract examples from ``pipelines`` in round robin.

            :param pipelines:
                The data pipelines to round robin.
            :param stop_at_shortest:
                If ``True``, stop round_robin when first pipeline reaches its end.
                If ``False``, circle around finished pipelines until all pipelines
                reach their end.
            """

        @staticmethod
        def sample(
            pipelines: Sequence[DataPipeline], weights: Optional[Sequence[float]] = None
        ) -> DataPipelineBuilder:
            """Extract examples from ``pipelines`` by sampling based on ``weights``.

            :param data_pipelines:
                The data pipelines to sample from.
            :param weights:
                Desired distribution of pipelines. If None, use uniform distribution.
            """

        @staticmethod
        def zip(
            pipelines: Sequence[DataPipeline],
            names: Optional[Sequence[str]] = None,
            zip_to_shortest: bool = False,
            flatten: bool = False,
            disable_parallelism: bool = False,
        ) -> DataPipelineBuilder:
            """Zip together examples read from ``pipelines``.

            :param pipelines:
                The data pipelines to zip.
            :param names:
                The names to assign to the data pipelines.
            :param flatten:
            :param disable_parallelism:
                If ``True``, calls each data pipeline sequentially.
            """

    class DataPipelineBuilder:
        """API to create DataPipeline"""

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
            skip_long_examples: bool = False,
            drop_remainder: bool = False,
        ) -> Self:
            """Combine examples of similar shape into batches."""

        def collate(
            self,
            pad_value: Optional[int] = None,
            pad_to_multiple: int = 1,
            overrides: Optional[Sequence[CollateOptionsOverride]] = None,
        ) -> Self:
            """Concatenate a list of inputs into a single inputs.

            This is equivalent to calling `.map(Collater())`.
            See :py:class:`fairseq2.data.Collater` for details.
            """

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
            """Apply ``fn`` to each example.

            Example usage::

                data = [2, 5]
                data.map(lambda x: x + 10)
                # yields: 12, 15
                data.map(lambda x: x + 10, num_parallel_calls=8)
                # same results but will use more cores
                data = [{"a": 2, "b": 1}, {"a": 5, "b": 3}]
                data.map(lambda x: x + 10, selector="a")
                # yields: {"a": 12, "b": 1}, {"a": 15, "b": 3}
                data.map(lambda x: x + 10, selector="a,b")
                # yields: {"a": 12, "b": 11}, {"a": 15, "b": 13}

            :param fn:
                The function to apply.
                If it's a list of function, they will be automatically chained.
                ``.map([f1, f2])`` is the more efficient version of ``.map(f1).map(f2)``

            :param selector:
                The column to apply the function to. Several colums can be specified by separating them with a ",".
                See :ref:`reference/data:column syntax` for more details.
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
    ) -> DataPipelineBuilder:
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
        """Read each file in a zip archive"""
        ...

    class CollateOptionsOverride:
        """Overrides how the collater should create batch for a particular column.

        Useful if not all columns should use the same padding idx, or padding multiple.
        See :py:class:`Collater` for details.

        :param selector:
            The columns this overrides applies to.
            See :ref:`reference/data:column syntax` for details on how to specify columns.
        """

        def __init__(
            self,
            selector: str,
            pad_value: Optional[int] = None,
            pad_to_multiple: int = 1,
        ) -> None:
            ...

        @property
        def selector(self) -> str:
            ...

        @property
        def pad_value(self) -> Optional[int]:
            ...

        @property
        def pad_to_multiple(self) -> int:
            ...

    class Collater:
        """Concatenate a list of inputs into a single inputs.

        Used to create batches.
        If all tensors in the input example have the same last dimension,
        ``Collater`` returns the concatenated tensors.

        Otherwise ``pad_value`` is required, and the last dimension of the batch will
        be made long enough to fit the longest tensor, rounded up to ``pad_to_multiple``.
        The returned batch is then a dictionary with the following keys::

            {
                "is_ragged": True/False # True if padding was needed
                "seqs": [[1, 4, 5, 0], [1, 2, 3, 4]]  # "(Tensor) concatenated and padded tensors from the input
                "seq_lens": [3, 4]  # A tensor describing the original length of each input tensor
            }

        Collater preserves the shape of the original data.
        For a tuple of lists, it returns a tuple of batches.
        For a dict of lists, it returns a dict of lists.

        :param pad_value:
            When concatenating tensors of different lengths,
            the value used to pad the shortest tensor

        :param pad_to_multiple:
            Always pad to a length of that multiple.

        :param overrides:
            List of overrides :py:class:`CollateOptionsOverride`.
            Allows to override ``pad_value`` and ``pad_to_multiple`` for specific columns.
        """

        def __init__(
            self,
            pad_value: Optional[int] = None,
            pad_to_multiple: int = 1,
            overrides: Optional[Sequence[CollateOptionsOverride]] = None,
        ) -> None:
            ...

        def __call__(self, data: Any) -> Any:
            """Concatenate the input tensors"""
            ...

    class FileMapper:
        """For a given file name, returns the file content as bytes.

        The file name can also specify a slice of the file in bytes:
        ``FileMapper("big_file.txt:1024:48")`` will read 48 bytes at offset 1024.

        :param root_dir:
            Root directory for looking up relative file names.
            Warning, this is not enforced, FileMapper will happily read any file on the system.

        :param cached_fd_count:
            Enables an LRU cache on the last ``cached_fd_count`` files read.
            ``FileMapper`` will memory map all the cached file,
            so this is especially useful for reading several slices of the same file.
        """

        def __init__(
            self,
            root_dir: Optional[PathLike] = None,
            cached_fd_count: Optional[int] = None,
        ) -> None:
            ...

        def __call__(self, filename: PathLike) -> FileMapperOutput:
            """Parses the file name and returns the file bytes.

            :returns:
                A dict with the following keys::

                    {
                        "path": "the/path.txt" # the relative path of the file
                        "data": MemoryBlock  # a memory block with the content of the file. You can use `bytes` to get a regular python object.
                    }
            """
            ...

    class ByteStreamError(RuntimeError):
        """Raised when a dataset file can't be read."""

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


def create_bucket_sizes(
    max_num_elements: int, max_seq_len: int, min_seq_len: int = 1
) -> List[Tuple[int, int]]:
    """Create optimal bucket sizes for ``max_num_elements`` with ``max_seq_len``.

    This is a convenience function that can be used with the
    :meth:`DataPipeline.bucket_by_length` operator.
    """
    if max_num_elements < max_seq_len:
        raise ValueError(
            f"`max_seq_len` must be less than or equal to `max_num_elements` ({max_num_elements}), but is {max_seq_len} instead."
        )

    if min_seq_len > max_seq_len:
        raise ValueError(
            f"`min_seq_len` must be less than or equal to `max_seq_len` ({max_seq_len}), but is {min_seq_len} instead."
        )

    bucket_sizes = []

    seq_len = min_seq_len

    batch_size = max_num_elements // seq_len

    while seq_len <= max_seq_len:
        bucket_sizes.append((batch_size, seq_len))

        batch_size = max_num_elements // (seq_len + 1)

        seq_len = max_num_elements // batch_size

    return bucket_sizes
