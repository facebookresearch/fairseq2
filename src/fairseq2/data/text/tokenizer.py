# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List, Sequence, Union

from torch import Tensor

from fairseq2.data.string import StringLike


class TokenEncoder(ABC):
    """Encodes sentences into token indices."""

    @abstractmethod
    def __call__(self, sentences: Union[StringLike, Sequence[StringLike]]) -> Tensor:
        """
        :param sentences:
            The sentences to encode.
        """


class TokenDecoder(ABC):
    """Decodes sentences from token indices."""

    @abstractmethod
    def __call__(self, token_indices: Tensor) -> List[StringLike]:
        """
        :param token_indices:
            The token indices to decode from.
        """
