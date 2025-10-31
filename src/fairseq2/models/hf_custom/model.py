# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Literal, final, overload

import torch
from torch import Tensor
from typing_extensions import override

from torch.nn import Module
import torch.nn

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.clm import CausalLM
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.models.transformer_lm.decoder import TransformerLMDecoder
from fairseq2.nn import BatchLayout, IncrementalStateBag, Projection
from fairseq2.nn.functional import cross_entropy

from dataclasses import dataclass

from transformers import Qwen2_5OmniForConditionalGeneration

class HFCustomModel(Module):
    """Represents a custom model checkpoint loaded from HuggingFace."""

    def __init__(
        self,
        checkpoint_name: str,
        dtype: DataType = torch.float32,
        **kwargs
    ) -> None:
        """
        :param decoder_frontend: The decoder frontend.
        """
        super().__init__()
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(checkpoint_name, dtype=dtype, **kwargs)
                
    def forward(
        self,
        **inputs,
    ) -> Tensor:
        return model(
