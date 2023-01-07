# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import final

import torch
from torch import Tensor

from fairseq2.data.data_pipeline import DataProcessor
from fairseq2.data.string import String
from fairseq2.data.typing import StringLike
from fairseq2.typing import DataType, Device

# fmt: off

@final
class SentencePieceModel:
    def __init__(
        self,
        pathname: StringLike,
        control_tokens: Sequence[StringLike] | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
        reverse: bool = False,
    ) -> None:
        ...

    def token_to_index(self, token: StringLike) -> int:
        ...

    def index_to_token(self, idx: int) -> String:
        ...

    @property
    def unk_idx(self) -> int:
        ...

    @property
    def bos_idx(self) -> int:
        ...

    @property
    def eos_idx(self) -> int:
        ...

    @property
    def pad_idx(self) -> int:
        ...

    @property
    def vocabulary_size(self) -> int:
        ...


@final
class SentencePieceEncoder(DataProcessor):
    def __init__(
        self,
        model: SentencePieceModel,
        enable_sampling: bool = False,
        nbest_size: int = -1,
        alpha: float = 0.1,
        batch_size: int | None = None,
        pad_to_length: int | None = None,
        pad_to_multiple: int = 1,
        lef_pad: bool = False,
        dtype: DataType = torch.int32,
        device: Device | None = None,
        pin_memory: bool = False,
        disable_parallelism: bool = False,
    ) -> None:
        ...

    def __call__(self, sentences: StringLike | Sequence[StringLike]) -> Tensor:
        ...


@final
class SentencePieceDecoder(DataProcessor):
    def __init__(self, model: SentencePieceModel) -> None:
        ...

    def __call__(self, token_indices: Tensor) -> list[String]:
        ...
