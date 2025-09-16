# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from typing import List

import torch

from fairseq2.models.llama4.config import Llama4VisionEncoderConfig


class VisionEmbeddings(torch.nn.Module):
    """
    The Llama 4 vision embedding class has two main parts:
    1. A transformer encoder, which could be frozen during finetuning
    2. An adapter, which should be always be trained.

    TODO: Placeholder for now, this is WIP.
    Reference implementation available at:
    https://github.com/llamastack/llama-stack/blob/main/llama_stack/models/llama/llama4/vision/embedding.py
    """

    def __init__(self, config: Llama4VisionEncoderConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        image_batch: List[List[torch.Tensor]],
        image_mask: torch.Tensor,
        h_ref: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("VisionEmbeddings is not implemented yet")
