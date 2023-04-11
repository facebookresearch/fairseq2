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
    final,
)

from fairseq2 import DOC_MODE
from fairseq2.data.string import StringLike
from fairseq2.data.typing import PathLike


@final
class DataPipeline:
    def __iter__(self) -> Iterator[Any]:
        """Return an iterator over the examples in the data pipeline."""
        return iter([])

    def skip(self, num_examples: int) -> int:
        """Skip reading a specified number of examples.

        :param num_examples:
            The number of examples to skip.

        :returns:
            The number of examples skipped. It can be less than ``num_examples``
            if the end of the data pipeline is reached.
        """
        return 0

    def reset(self) -> None:
        """Move back to the first example in the data pipeline."""

    @property
    def is_broken(self) -> bool:
        """Return whether the data pipeline is broken.

        If ``True``, any future operation on this data pipeline will raise a
        :class:`DataPipelineError`.
        """
        return False

    def state_dict(self) -> Dict[str, Any]:
        """Return a dictionary containing the state of the data pipeline.

        The current position of the data pipeline can be restored by passing the
        returned state dictionary to :meth:`load_state_dict`.
        """
        return {}

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ) -> None:
        """Restore the state of the data pipeline from ``state_dict``.

        :param state_dict:
            A state dictionary previously returned by :meth:`state_dict`.
        :param strict:
            If ``True``, enforces that the keys in ``state_dict`` match the keys
            returned by :meth:`state_dict`.
        """


@final
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
        return self

    def batch_by_length(
        self, batch_shapes: Sequence[Tuple[int, int]], pad_idx: int
    ) -> "DataPipelineBuilder":
        """Combine examples of similar shape into batches.

        :param batch_shapes:
            The allowed batch shapes.
        """
        return self

    def map(
        self, fn: Callable[[Any], Any], chunk_size: int = 1
    ) -> "DataPipelineBuilder":
        """Apply ``fn`` to every example in the data pipeline.

        :param fn:
            The function to apply.
        """
        return self

    def prefetch(self, num_examples: int) -> "DataPipelineBuilder":
        """Prefetch examples in the background while the current example is
        being processed.

        :param num_examples:
            The number of examples to prefetch.
        """
        return self

    def shard(self, shard_idx: int, num_shards: int) -> "DataPipelineBuilder":
        """Read only 1/``num_shards`` of the examples in the data pipeline.

        :param shard_idx:
            The shard index.
        :param num_shards:
            The number of shards.
        """
        return self

    def yield_from(self, fn: Callable[[Any], DataPipeline]) -> "DataPipelineBuilder":
        """Map every example to a data pipeline and yield the examples returned
        from the mapped data pipelines.

        :param fn:
            The function to map examples to data pipelines.
        """
        return self

    def and_return(self) -> DataPipeline:
        """Return a new :class:`DataPipeline` instance."""
        raise NotImplementedError()


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
    raise NotImplementedError()


def read_sequence(s: Sequence[Any]) -> "DataPipelineBuilder":
    """Read every element in ``s``.

    :param s:
        The sequence to read.
    """
    raise NotImplementedError()


def zip_data_pipelines(
    data_pipelines: Sequence[DataPipeline],
) -> "DataPipelineBuilder":
    """Zip together examples read from ``data_pipelines``.

    :param data_pipelines:
        The data pipelines to zip.
    """
    raise NotImplementedError()


class StreamError(RuntimeError):
    """Raised when a dataset cannot be read."""


class RecordError(RuntimeError):
    """Raised when a corrupt record is encountered while reading a dataset."""


if not TYPE_CHECKING and not DOC_MODE:
    from fairseq2.C.data.data_pipeline import (  # noqa: F811
        DataPipeline,
        DataPipelineBuilder,
        DataPipelineError,
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
            RecordError,
            StreamError,
            list_files,
            read_sequence,
            zip_data_pipelines,
        ]

        for t in ctypes:
            t.__module__ = __name__

    _set_module()
