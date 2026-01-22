# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from torch import Tensor
from typing_extensions import override

from fairseq2.device import Device, SupportsDeviceTransfer
from fairseq2.error import InvalidOperationError
from fairseq2.nn import BatchLayout


@final
class SequenceBatch(SupportsDeviceTransfer):
    _seqs: Tensor
    _seq_lens: list[int]
    _packed: bool
    _target_mask: Tensor | None
    _batch_size: int
    _num_examples: int
    _num_elements: int
    _num_target_elements: int
    _padding: int
    _example: object

    def __init__(
        self,
        seqs: Tensor,
        seq_lens: list[int] | None,
        *,
        packed: bool = False,
        target_mask: Tensor | None = None,
        example: object | None = None,
    ) -> None:
        self._seqs = seqs

        self._packed = packed

        if seq_lens is not None and not seq_lens:
            raise ValueError("`seq_lens` must be non-empty.")

        if packed:
            if seqs.ndim == 0:
                raise ValueError(
                    "`seqs` must be at least one-dimensional, but has 0 dimension(s) instead."
                )

            batch_width = seqs.size(0)

            if batch_width == 0:
                raise ValueError("`seqs.shape[0]` must be greater than or equal to 1.")

            if seq_lens is None:
                seq_lens = [batch_width]

            self._num_elements = 0

            for idx, seq_len in enumerate(seq_lens):
                if seq_len < 1:
                    raise ValueError(
                        f"All lengths in `seq_lens` must be greater than or equal to 1, but the length at index {idx} is {seq_len} instead."
                    )

                self._num_elements += seq_len

            if self._num_elements > batch_width:
                raise ValueError(
                    f"`sum(seq_lens)` must be less than or equal to `seqs.shape[0]` ({batch_width}), but is {self._num_elements} instead."
                )

            self._seq_lens = seq_lens

            self._batch_size = 1

            self._num_examples = len(seq_lens)

            self._padding = batch_width - self._num_elements
        else:
            if seqs.ndim < 2:
                raise ValueError(
                    f"`seqs` must be at least two-dimensional, but has {seqs.ndim} dimension(s) instead."
                )

            batch_size, batch_width = seqs.shape[:2]

            if batch_width == 0:
                raise ValueError("`seqs.shape[1]` must be greater than or equal to 1.")

            if seq_lens is None:
                seq_lens = [batch_width] * batch_size

            if len(seq_lens) != batch_size:
                raise ValueError(
                    f"`len(seq_lens)` must be equal to `seqs.shape[0]` ({batch_size}), but is {len(seq_lens)} instead."
                )

            self._num_elements = 0

            self._padding = 0

            for idx, seq_len in enumerate(seq_lens):
                if seq_len < 1:
                    raise ValueError(
                        f"All lengths in `seq_lens` must be greater than or equal to 1, but the length at index {idx} is {seq_len} instead."
                    )

                if seq_len > batch_width:
                    raise ValueError(
                        f"All lengths in `seq_lens` must be less than or equal to `seqs.shape[1]` ({batch_width}), but the length at index {idx} is {seq_len} instead."
                    )

                self._num_elements += seq_len

                self._padding += batch_width - seq_len

            self._seq_lens = seq_lens

            self._batch_size = batch_size

            self._num_examples = batch_size

        if target_mask is not None:
            if packed:
                if target_mask.ndim != 1:
                    raise ValueError(
                        f"`target_mask` must be one-dimensional, but has {target_mask.ndim} dimension(s) instead."
                    )

                if target_mask.shape != seqs.shape[:1]:
                    raise ValueError(
                        f"`target_mask` must have shape of {seqs.shape[:1]}, but has a shape of {target_mask.shape} instead."
                    )
            else:
                if target_mask.ndim != 2:
                    raise ValueError(
                        f"`target_mask` must be two-dimensional, but has {target_mask.ndim} dimension(s) instead."
                    )

                if target_mask.shape != seqs.shape[:2]:
                    raise ValueError(
                        f"`target_mask` must have shape of {seqs.shape[:2]}, but has a shape of {target_mask.shape} instead."
                    )

        self._target_mask = target_mask

        if target_mask is None:
            self._num_target_elements = self._num_elements
        else:
            self._num_target_elements = int(target_mask.sum())

        self._example = example

    def as_auto_regressive(self) -> tuple[SequenceBatch, SequenceBatch]:
        """Trims the batch to train an auto-regressive model."""
        if self._packed:
            seqs = self._seqs[:-1]

            seq_lens = self._seq_lens.copy()

            if seq_lens[-1] == 1:
                if len(seq_lens) == 1:
                    raise InvalidOperationError(
                        "Length of the sequence at index 0 is already 1 and cannot be trimmed to 0."
                    )

                del seq_lens[-1]
            else:
                seq_lens[-1] -= 1
        else:
            seqs = self._seqs[:, :-1]

            seq_lens = []

            for idx, seq_len in enumerate(self._seq_lens):
                if seq_len == 1:
                    raise InvalidOperationError(
                        f"Length of the sequence at index {idx} is already 1 and cannot be trimmed to 0."
                    )

                seq_lens.append(seq_len - 1)

        batch = SequenceBatch(
            seqs, seq_lens, packed=self._packed, example=self._example
        )

        if self._packed:
            targets = self._seqs[1:]

            if self._target_mask is None:
                target_mask = None
            else:
                target_mask = self._target_mask[1:]
        else:
            targets = self._seqs[:, 1:]

            if self._target_mask is None:
                target_mask = None
            else:
                target_mask = self._target_mask[:, 1:]

        target_batch = SequenceBatch(
            targets, seq_lens, packed=self._packed, target_mask=target_mask
        )

        return batch, target_batch

    def as_input(self) -> tuple[Tensor, BatchLayout]:
        seqs_layout = BatchLayout.of(self._seqs, self._seq_lens, packed=self._packed)

        return self._seqs, seqs_layout

    @override
    def to(self, device: Device, *, non_blocking: bool = False) -> None:
        self._seqs = self._seqs.to(device, non_blocking=non_blocking)

        if self._target_mask is not None:
            self._target_mask = self._target_mask.to(device, non_blocking=non_blocking)

    @property
    def seqs(self) -> Tensor:
        return self._seqs

    @property
    def seq_lens(self) -> Sequence[int]:
        return self._seq_lens

    @property
    def packed(self) -> bool:
        return self._packed

    @property
    def target_mask(self) -> Tensor | None:
        return self._target_mask

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_examples(self) -> int:
        return self._num_examples

    @property
    def num_elements(self) -> int:
        return self._num_elements

    @property
    def num_target_elements(self) -> int:
        return self._num_target_elements

    @property
    def padding(self) -> int:
        return self._padding

    @property
    def example(self) -> object:
        return self._example

    def __repr__(self) -> str:
        s = (
            f"seqs={self._seqs}, "
            f"seq_lens={self._seq_lens}, "
            f"packed={self._packed}, "
            f"batch_size={self._batch_size}, "
            f"num_examples={self._num_examples}, "
            f"num_elements={self._num_elements}, "
            f"num_target_elements={self._num_target_elements}, "
            f"padding={self._padding}, "
            f"example={self._example}"
        )

        return f"SequenceBatch({s})"


@final
class Seq2SeqBatch(SupportsDeviceTransfer):
    _source_seqs: Tensor
    _source_seq_lens: list[int]
    _target_seqs: Tensor
    _target_seq_lens: list[int]
    _packed: bool
    _target_mask: Tensor | None
    _batch_size: int
    _num_examples: int
    _num_elements: int
    _num_source_elements: int
    _num_target_elements: int
    _padding: int
    _example: object

    def __init__(
        self,
        source_seqs: Tensor,
        source_seq_lens: list[int] | None,
        target_seqs: Tensor,
        target_seq_lens: list[int] | None,
        *,
        packed: bool = False,
        target_mask: Tensor | None = None,
        example: object | None = None,
    ) -> None:
        self._source_seqs = source_seqs
        self._target_seqs = target_seqs

        self._packed = packed

        if source_seq_lens is not None and not source_seq_lens:
            raise ValueError("`source_seq_lens` must be non-empty.")

        if target_seq_lens is not None and not target_seq_lens:
            raise ValueError("`target_seq_lens` must be non-empty.")

        if packed:
            if source_seqs.ndim == 0:
                raise ValueError(
                    "`source_seqs` must be at least one-dimensional, but has 0 dimensions instead."
                )

            if target_seqs.ndim == 0:
                raise ValueError(
                    "`target_seqs` must be at least one-dimensional, but has 0 dimensions instead."
                )

            source_batch_width = source_seqs.size(0)
            target_batch_width = target_seqs.size(0)

            if source_batch_width == 0:
                raise ValueError(
                    "`source_seqs.shape[0]` must be greater than or equal to 1."
                )

            if target_batch_width == 0:
                raise ValueError(
                    "`target_seqs.shape[0]` must be greater than or equal to 1."
                )

            if source_seq_lens is None:
                source_seq_lens = [source_batch_width]

            if target_seq_lens is None:
                target_seq_lens = [target_batch_width]

            if len(source_seq_lens) != len(target_seq_lens):
                raise ValueError(
                    f"`len(source_seq_lens)` and `len(target_seq_lens)` must be equal, but they are {len(source_seq_lens)} and {len(target_seq_lens)} instead."
                )

            self._num_source_elements = 0
            self._num_target_elements = 0

            for idx, seq_len in enumerate(source_seq_lens):
                if seq_len < 1:
                    raise ValueError(
                        f"All lengths in `source_seq_lens` must be greater than or equal to 1, but the length at index {idx} is {seq_len} instead."
                    )

                self._num_source_elements += seq_len

            for idx, seq_len in enumerate(target_seq_lens):
                if seq_len < 1:
                    raise ValueError(
                        f"All lengths in `target_seq_lens` must be greater than or equal to 1, but the length at index {idx} is {seq_len} instead."
                    )

                self._num_target_elements += seq_len

            if self._num_source_elements > source_batch_width:
                raise ValueError(
                    f"`sum(source_seq_lens)` must be less than or equal to `source_seqs.shape[0]` ({source_batch_width}), but is {self._num_source_elements} instead."
                )

            if self._num_target_elements > target_batch_width:
                raise ValueError(
                    f"`sum(target_seq_lens)` must be less than or equal to `target_seqs.shape[0]` ({target_batch_width}), but is {self._num_target_elements} instead."
                )

            self._source_seq_lens = source_seq_lens
            self._target_seq_lens = target_seq_lens

            self._batch_size = 1

            self._num_examples = len(source_seq_lens)

            source_padding = source_batch_width - self._num_source_elements
            target_padding = target_batch_width - self._num_target_elements

            self._padding = source_padding + target_padding
        else:
            if source_seqs.ndim < 2:
                raise ValueError(
                    f"`source_seqs` must be at least two-dimensional, but has {source_seqs.ndim} dimension(s) instead."
                )

            if target_seqs.ndim < 2:
                raise ValueError(
                    f"`target_seqs` must be at least two-dimensional, but has {target_seqs.ndim} dimension(s) instead."
                )

            source_batch_size, source_batch_width = source_seqs.shape[:2]
            target_batch_size, target_batch_width = target_seqs.shape[:2]

            if source_batch_size != target_batch_size:
                raise ValueError(
                    f"`source_seqs.shape[0]` and `target_seqs.shape[0]` must be equal, but they are {source_batch_size} and {target_batch_size} instead."
                )

            if source_batch_width == 0:
                raise ValueError(
                    "`source_seqs.shape[1]` must be greater than or equal to 1."
                )

            if target_batch_width == 0:
                raise ValueError(
                    "`target_seqs.shape[1]` must be greater than or equal to 1."
                )

            if source_seq_lens is None:
                source_seq_lens = [source_batch_width] * source_batch_size

            if target_seq_lens is None:
                target_seq_lens = [target_batch_width] * target_batch_size

            if len(source_seq_lens) != source_batch_size:
                raise ValueError(
                    f"`len(source_seq_lens)` must be equal to `source_seqs.shape[0]` ({source_batch_size}), but is {len(source_seq_lens)} instead."
                )

            if len(target_seq_lens) != target_batch_size:
                raise ValueError(
                    f"`len(target_seq_lens)` must be equal to `target_seqs.shape[0]` ({target_batch_size}), but is {len(target_seq_lens)} instead."
                )

            self._num_source_elements = 0
            self._num_target_elements = 0

            self._padding = 0

            for idx, seq_len in enumerate(source_seq_lens):
                if seq_len < 1:
                    raise ValueError(
                        f"All lengths in `source_seq_lens` must be greater than or equal to 1, but the length at index {idx} is {seq_len} instead."
                    )

                if seq_len > source_batch_width:
                    raise ValueError(
                        f"All lengths in `source_seq_lens` must be less than or equal to `source_seqs.shape[1]` ({source_batch_width}), but the length at index {idx} is {seq_len} instead."
                    )

                self._num_source_elements += seq_len

                self._padding += source_batch_width - seq_len

            for idx, seq_len in enumerate(target_seq_lens):
                if seq_len < 1:
                    raise ValueError(
                        f"All lengths in `target_seq_lens` must be greater than or equal to 1, but the length at index {idx} is {seq_len} instead."
                    )

                if seq_len > target_batch_width:
                    raise ValueError(
                        f"All lengths in `target_seq_lens` must be less than or equal to `target_seqs.shape[1]` ({target_batch_width}), but the length at index {idx} is {seq_len} instead."
                    )

                self._num_target_elements += seq_len

                self._padding += target_batch_width - seq_len

            self._source_seq_lens = source_seq_lens
            self._target_seq_lens = target_seq_lens

            self._batch_size = source_batch_size

            self._num_examples = source_batch_size

        self._num_elements = self._num_source_elements + self._num_target_elements

        if target_mask is not None:
            if packed:
                if target_mask.ndim != 1:
                    raise ValueError(
                        f"`target_mask` must be one-dimensional, but has {target_mask.ndim} dimension(s) instead."
                    )

                if target_mask.shape != target_seqs.shape[:1]:
                    raise ValueError(
                        f"`target_mask` must have shape of {target_seqs.shape[:1]}, but has a shape of {target_mask.shape} instead."
                    )
            else:
                if target_mask.ndim != 2:
                    raise ValueError(
                        f"`target_mask` must be two-dimensional, but has {target_mask.ndim} dimension(s) instead."
                    )

                if target_mask.shape != target_seqs.shape[:2]:
                    raise ValueError(
                        f"`target_mask` must have shape of {target_seqs.shape[:2]}, but has a shape of {target_mask.shape} instead."
                    )

        self._target_mask = target_mask

        if target_mask is None:
            self._num_target_elements = self._num_elements
        else:
            self._num_target_elements = int(target_mask.sum())

        self._example = example

    def as_auto_regressive(self) -> tuple[Seq2SeqBatch, SequenceBatch]:
        """Trims the batch to train an auto-regressive model."""
        if self._packed:
            seqs = self._target_seqs[:-1]

            seq_lens = self._target_seq_lens.copy()

            if seq_lens[-1] == 1:
                if len(seq_lens) == 1:
                    raise InvalidOperationError(
                        "Length of the target sequence at index 0 is already 1 and cannot be trimmed to 0."
                    )

                del seq_lens[-1]
            else:
                seq_lens[-1] -= 1
        else:
            seqs = self._target_seqs[:, :-1]

            seq_lens = []

            for idx, seq_len in enumerate(self._target_seq_lens):
                if seq_len == 1:
                    raise InvalidOperationError(
                        f"Length of the target sequence at index {idx} is already 1 and cannot be trimmed to 0."
                    )

                seq_lens.append(seq_len - 1)

        batch = Seq2SeqBatch(
            self._source_seqs,
            self._source_seq_lens,
            seqs,
            seq_lens,
            packed=self._packed,
            example=self._example,
        )

        if self._packed:
            targets = self._target_seqs[1:]

            if self._target_mask is None:
                target_mask = None
            else:
                target_mask = self._target_mask[1:]
        else:
            targets = self._target_seqs[:, 1:]

            if self._target_mask is None:
                target_mask = None
            else:
                target_mask = self._target_mask[:, 1:]

        target_batch = SequenceBatch(
            targets, seq_lens, packed=self._packed, target_mask=target_mask
        )

        return batch, target_batch

    def as_source_input(self) -> tuple[Tensor, BatchLayout]:
        source_seqs_layout = BatchLayout.of(
            self._source_seqs, self._source_seq_lens, packed=self._packed
        )

        return self._source_seqs, source_seqs_layout

    def as_target_input(self) -> tuple[Tensor, BatchLayout]:
        target_seqs_layout = BatchLayout.of(
            self._target_seqs, self._target_seq_lens, packed=self._packed
        )

        return self._target_seqs, target_seqs_layout

    @override
    def to(self, device: Device, *, non_blocking: bool = False) -> None:
        self._source_seqs = self._source_seqs.to(device, non_blocking=non_blocking)
        self._target_seqs = self._target_seqs.to(device, non_blocking=non_blocking)

        if self._target_mask is not None:
            self._target_mask = self._target_mask.to(device, non_blocking=non_blocking)

    @property
    def source_seqs(self) -> Tensor:
        return self._source_seqs

    @property
    def source_seq_lens(self) -> Sequence[int]:
        return self._source_seq_lens

    @property
    def target_seqs(self) -> Tensor:
        return self._target_seqs

    @property
    def target_seq_lens(self) -> Sequence[int]:
        return self._target_seq_lens

    @property
    def packed(self) -> bool:
        return self._packed

    @property
    def target_mask(self) -> Tensor | None:
        return self._target_mask

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_examples(self) -> int:
        return self._num_examples

    @property
    def num_elements(self) -> int:
        return self._num_elements

    @property
    def num_source_elements(self) -> int:
        return self._num_source_elements

    @property
    def num_target_elements(self) -> int:
        return self._num_target_elements

    @property
    def padding(self) -> int:
        return self._padding

    @property
    def example(self) -> object:
        return self._example

    def __repr__(self) -> str:
        s = (
            f"source_seqs={self._source_seqs}, "
            f"source_seq_lens={self._source_seq_lens}, "
            f"target_seqs={self._target_seqs}, "
            f"target_seq_lens={self._target_seq_lens}, "
            f"packed={self._packed}, "
            f"batch_size={self._batch_size}, "
            f"num_examples={self._num_examples}, "
            f"num_elements={self._num_elements}, "
            f"num_source_elements={self._num_source_elements}, "
            f"num_target_elements={self._num_target_elements}, "
            f"padding={self._padding}, "
            f"example={self._example}"
        )

        return f"Seq2SeqBatch({s})"
