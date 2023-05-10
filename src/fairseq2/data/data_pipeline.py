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
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from fairseq2 import DOC_MODE
from fairseq2.data.string import StringLike
from fairseq2.data.typing import PathLike

if TYPE_CHECKING or DOC_MODE:

    class DataPipeline:
        def __iter__(self) -> Iterator[Any]:
            """Return an iterator over the examples in the data pipeline."""

        def skip(self, num_examples: int) -> int:
            """Skip reading a specified number of examples.

            :param num_examples:
                The number of examples to skip.

            :returns:
                The number of examples skipped. It can be less than
                ``num_examples`` if the end of the data pipeline is reached.
            """

        def reset(self) -> None:
            """Move back to the first example in the data pipeline."""

        @property
        def is_broken(self) -> bool:
            """Return whether the data pipeline is broken.

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

    class DataPipelineBuilder:
        def batch(
            self,
            batch_size: int,
            drop_remainder: bool = False,
            pad_idx: Optional[int] = None,
        ) -> "DataPipelineBuilder":
            """Combine a number of consecutive examples into a single example.

            :param batch_size:
                The number of examples to combine.
            :param drop_remainder:
                If ``True``, drops the last batch in case it has fewer than
                ``batch_size`` examples.
            :param pad_idx:
                Required to be able to batch tensors with different lengths.
            """

        def batch_by_length(
            self, batch_shapes: Sequence[Tuple[int, int]], pad_idx: int
        ) -> "DataPipelineBuilder":
            """Combine examples of similar shape into batches.

            :param batch_shapes:
                The allowed batch shapes.
            """

        def map(
            self, fn: Callable[[Any], Any], chunk_size: int = 1
        ) -> "DataPipelineBuilder":
            """Apply ``fn`` to every example in the data pipeline.

            :param fn:
                The function to apply.
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
            self, buffer_size: int, seed: int = 0, deterministic: bool = False
        ) -> "DataPipelineBuilder":
            """Shuffle elements using a fixed sized buffer.

            :param buffer_size:
                Intermediate buffer used for shuffling.
            :param seed:
                Seed of the RNG used by the shuffle operation
            :param deterministic:
                In deterministic mode, we will fully save all the state of the buffer when checkpointing.
                This ensures that on preemption restart no sample will be lost.
                This can be expensive for large shuffle size.
            """

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

    def read_sequence(s: Sequence[Any]) -> "DataPipelineBuilder":
        """Read every element in ``s``.

        :param s:
            The sequence to read.
        """

    def read_zipped_records(pathname: PathLike) -> DataPipelineBuilder:
        ...

    def round_robin_data_pipelines(
        data_pipelines: Sequence[DataPipeline], probs: List[float] = []
    ) -> "DataPipelineBuilder":
        """Do a round robin on all pipelines.

        :param data_pipelines:
            The data pipelines to round robin.
        """

    def zip_data_pipelines(
        data_pipelines: Sequence[DataPipeline],
    ) -> "DataPipelineBuilder":
        """Zip together examples read from ``data_pipelines``.

        :param data_pipelines:
            The data pipelines to zip.
        """

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
    from fairseq2.C.data.data_pipeline import (
        round_robin_data_pipelines as round_robin_data_pipelines,
    )
    from fairseq2.C.data.data_pipeline import zip_data_pipelines as zip_data_pipelines

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
            zip_data_pipelines,
            round_robin_data_pipelines,
        ]

        for t in ctypes:
            t.__module__ = __name__

    _set_module_name()
