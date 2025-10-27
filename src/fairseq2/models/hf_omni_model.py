# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from typing import final
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Module

from fairseq2.device import Device

from fairseq2.nn import BatchLayout, Linear
from fairseq2.nn.functional import cross_entropy

def create_processor(model_name:str, **kwargs):
    return Qwen2_5OmniProcessor.from_pretrained(model_name, **kwargs, is_fast=True)

def create_model(model_name:str, from_config=False, output_hidden_states=False, device="cpu", dtype="auto", **kwargs):
    return HFModel(from_config, model_name, device, dtype, **kwargs)

@final
class HFModel(Module):
    """Represents a PyTorch model as can be found
    on huggingface.co"""

    def __init__(
        self,
        from_config: bool,
        model_name: str,
        output_hidden_states: bool,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param config:
            The imported huggingface model config.
        :param model:
            The imported huggingface model.
        :param device:
            The device to load the model onto.
        :param dtype:
            The tensor precision to use (e.g. float32, float16, bfloat16, etc.)
        """
        super().__init__()

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, dtype=dtype, device_map=device)
        if output_hidden_states:
            self.model.config.output_hidden_states = True

    def forward(self, **inputs):
        logits = self.model(**inputs)
        return logits
