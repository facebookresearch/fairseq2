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
    Union,
)

from fairseq2 import _DOC_MODE
from fairseq2.data.typing import PathLike, StringLike

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
            warn_only: bool = False,
            disable_parallelism: bool = False,
        ) -> "DataPipelineBuilder":
            """Zip together examples read from ``pipelines``.

            :param pipelines:
                The data pipelines to zip.
            :param warn_only:
                If ``True``, prints a warning, instead of raising an error, when the
                data pipelines do not have equal length.
            :param disable_parallelism:
                If ``True``, calls each data pipeline sequentially.
            """

        @staticmethod
        def round_robin(pipelines: Sequence["DataPipeline"]) -> "DataPipelineBuilder":
            """Extract examples from ``pipelines`` in round robin.

            :param pipelines:
                The data pipelines to round robin.
            """

    class DataPipelineBuilder:
        def batch(
            self, batch_size: int, drop_remainder: bool = False
        ) -> "DataPipelineBuilder":
            """Combine a number of consecutive examples into a single example.

            :param batch_size:
                The number of examples to combine.
            :param drop_remainder:
                If ``True``, drops the last batch in case it has fewer than
                ``batch_size`` examples.
            """

        def batch_by_length(
            self,
            bucket_sizes: Sequence[Tuple[int, int]],
            max_seq_len: int,
            selector: Optional[str] = None,
            drop_remainder: bool = False,
            warn_only: bool = False,
        ) -> "DataPipelineBuilder":
            """Combine examples of similar shape into batches."""

        def collate(self, pad_idx: Optional[int] = None) -> "DataPipelineBuilder":
            ...

        def filter(self, predicate: Callable[[Any], Any]) -> "DataPipelineBuilder":
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
        ) -> "DataPipelineBuilder":
            """Apply ``fn`` to every example.

            :param fn:
                The function to apply.
            :param num_parallel_calls:
                The number of examples to process in parallel.
            """

        def prefetch(self, num_examples: int) -> "DataPipelineBuilder":
            """Prefetch examples in the background while the current example is
            being processed.

            :param num_examples:
                The number of examples to prefetch.
            """

        def shard(self, shard_idx: int, num_shards: int) -> "DataPipelineBuilder":
            """Read only 1/``num_shards`` of the examples in the data pipeline.

            :param shard_idx:
                The shard index.
            :param num_shards:
                The number of shards.
            """

        def shuffle(
            self, shuffle_window: int, strict: bool = True, enabled: bool = True
        ) -> "DataPipelineBuilder":
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

        def skip(self, num_examples: int) -> "DataPipelineBuilder":
            """Skip ``num_examples`` examples."""

        def take(self, num_examples: int) -> "DataPipelineBuilder":
            """Return at most ``num_examples`` examples."""

        def yield_from(
            self, fn: Callable[[Any], DataPipeline]
        ) -> "DataPipelineBuilder":
            """
            Map every example to a data pipeline and yield the examples returned
            from the mapped data pipelines.

            :param fn:
                The function to map examples to data pipelines.
            """

        def and_return(self) -> DataPipeline:
            """Return a new :class:`DataPipeline` instance."""

    class DataPipelineError(RuntimeError):
        """Raised when an error occurs while reading from a data pipeline."""

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

    class StreamError(RuntimeError):
        """Raised when a dataset cannot be read."""

    class RecordError(RuntimeError):
        """Raised when a corrupt record is encountered while reading a dataset."""

else:
    from fairseq2.C.data.data_pipeline import DataPipeline as DataPipeline
    from fairseq2.C.data.data_pipeline import DataPipelineBuilder as DataPipelineBuilder
    from fairseq2.C.data.data_pipeline import DataPipelineError as DataPipelineError
    from fairseq2.C.data.data_pipeline import RecordError as RecordError
    from fairseq2.C.data.data_pipeline import StreamError as StreamError
    from fairseq2.C.data.data_pipeline import list_files as list_files
    from fairseq2.C.data.data_pipeline import read_sequence as read_sequence
    from fairseq2.C.data.data_pipeline import read_zipped_records as read_zipped_records

    def _set_module_name() -> None:
        ctypes = [
            DataPipeline,
            DataPipelineBuilder,
            DataPipelineError,
            RecordError,
            StreamError,
            list_files,
            read_sequence,
            read_zipped_records,
        ]

        for t in ctypes:
            t.__module__ = __name__

    _set_module_name()
