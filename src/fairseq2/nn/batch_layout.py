# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, final

import torch
from torch import Tensor

from fairseq2.device import Device
from fairseq2.utils.tensor import to_tensor


@final
class BatchLayout:
    _width: int
    _seq_begin_indices: list[int]
    _seq_begin_indices_pt: Tensor
    _seq_lens: list[int]
    _seq_lens_pt: Tensor
    _position_indices: Tensor
    _min_seq_len: int
    _max_seq_len: int
    _packed: bool
    _padded: bool

    def __init__(
        self,
        shape: tuple[int, ...],
        seq_lens: Sequence[int] | None,
        *,
        packed: bool = False,
        device: Device | None = None,
    ) -> None:
        self._packed = packed

        if packed:
            if len(shape) != 1:
                raise ValueError(
                    f"`shape` must be 1 dimensional, but is {len(shape)} dimensional instead."
                )

            batch_width = shape[0]

            if batch_width < 1:
                raise ValueError("`shape[0]` must be greater than or equal to 1.")

            if seq_lens is None:
                seq_lens = [batch_width]

            self._seq_begin_indices = [0]

            self._seq_lens = []

            self._position_indices = torch.arange(batch_width, device=device)

            self._num_elements = 0

            self._min_seq_len = batch_width
            self._max_seq_len = 0

            seq_beg = 0
            seq_end = 0

            for idx, seq_len in enumerate(seq_lens):
                if seq_len < 1:
                    raise ValueError(
                        f"All lengths in `seq_lens` must be greater than or equal to 1, but the length at index {idx} is {seq_len} instead."
                    )

                seq_end = seq_beg + seq_len

                if seq_end > batch_width:
                    raise ValueError(
                        f"`sum(seq_lens)` must be less than or equal to `shape[0]` ({batch_width}), but is {sum(seq_lens)} instead."
                    )

                self._seq_begin_indices.append(seq_end)

                self._seq_lens.append(seq_len)

                self._position_indices[seq_beg:seq_end] -= seq_beg

                self._min_seq_len = min(self._min_seq_len, seq_len)
                self._max_seq_len = max(self._max_seq_len, seq_len)

                seq_beg = seq_end

            self._position_indices[seq_end:] = -1  # pad

            self._padded = seq_end < batch_width
        else:
            if len(shape) != 2:
                raise ValueError(
                    f"`shape` must be 2 dimensional, but is {len(shape)} dimensional instead."
                )

            batch_size, batch_width = shape

            if batch_width < 1:
                raise ValueError("`shape[1]` must be greater than or equal to 1.")

            if seq_lens is None:
                seq_lens = [batch_width] * batch_size

            if len(seq_lens) != batch_size:
                raise ValueError(
                    f"`len(seq_lens)` must be equal to `shape[0]` ({batch_size}), but is {len(seq_lens)} instead."
                )

            self._seq_begin_indices = list(
                range(0, (batch_size * batch_width) + 1, batch_width)
            )

            self._seq_lens = []

            indices = torch.arange(batch_width, device=device)

            # (S) -> (N, S)
            self._position_indices = indices.expand(batch_size, -1).contiguous()

            self._min_seq_len = batch_width
            self._max_seq_len = 0

            self._padded = False

            for idx, seq_len in enumerate(seq_lens):
                if seq_len < 1:
                    raise ValueError(
                        f"All lengths in `seq_lens` must be greater than or equal to 1, but the length at index {idx} is {seq_len} instead."
                    )

                if seq_len > batch_width:
                    raise ValueError(
                        f"All lengths in `seq_lens` must be less than or equal to `shape[1]` ({batch_width}), but the length at index {idx} is {seq_len} instead."
                    )

                self._seq_lens.append(seq_len)

                if seq_len < batch_width:
                    self._padded = True

                self._position_indices[idx, seq_len:] = -1  # pad

                self._min_seq_len = min(self._min_seq_len, seq_len)
                self._max_seq_len = max(self._max_seq_len, seq_len)

        self._width = batch_width

        self._seq_begin_indices_pt = to_tensor(
            self._seq_begin_indices, dtype=torch.int32, device=device
        )

        self._seq_lens_pt = to_tensor(self._seq_lens, dtype=torch.int32, device=device)

        # Both `seq_begin_indices` and `seq_lens` are inherently dynamic and
        # require to be marked so to avoid redundant recompilations.
        torch._dynamo.maybe_mark_dynamic(self._seq_begin_indices_pt, 0)
        torch._dynamo.maybe_mark_dynamic(self._seq_lens_pt, 0)

    @staticmethod
    def of(
        batch: Tensor, seq_lens: list[int] | None = None, *, packed: bool = False
    ) -> BatchLayout:
        shape = batch.shape[:1] if packed else batch.shape[:2]

        return BatchLayout(shape, seq_lens, packed=packed, device=batch.device)

    @property
    def width(self) -> int:
        return self._width

    @property
    def seq_begin_indices(self) -> Sequence[int]:
        return self._seq_begin_indices

    @property
    def seq_begin_indices_pt(self) -> Tensor:
        return self._seq_begin_indices_pt

    @property
    def seq_lens(self) -> Sequence[int]:
        return self._seq_lens

    @property
    def seq_lens_pt(self) -> Tensor:
        return self._seq_lens_pt

    @property
    def min_seq_len(self) -> int:
        return self._min_seq_len

    compiled_max_seq_len: ClassVar[int | None] = None

    @property
    def max_seq_len(self) -> int:
        # TODO: As of PyTorch 2.7, integers cannot be marked as dynamic during
        # compilation. This is a workaround till that gets fixed.
        if torch.compiler.is_compiling():
            if self.compiled_max_seq_len is not None:
                return self.compiled_max_seq_len

        return self._max_seq_len

    @property
    def position_indices(self) -> Tensor:
        return self._position_indices

    @property
    def padded(self) -> bool:
        return self._padded

    @property
    def packed(self) -> bool:
        return self._packed

    def __repr__(self) -> str:
        s = (
            f"width={self._width}, "
            f"seq_begin_indices={self._seq_begin_indices}, "
            f"seq_lens={self._seq_lens}, "
            f"min_seq_len={self._min_seq_len}, "
            f"max_seq_len={self._max_seq_len}, "
            f"padded={self._padded}, "
            f"packed={self._packed}"
        )

        return f"BatchLayout({s})"
