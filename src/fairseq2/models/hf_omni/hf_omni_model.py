# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from typing import final
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Module

from fairseq2.device import Device

from fairseq2.nn import BatchLayout, Linear
from fairseq2.nn.functional import cross_entropy

from fairseq2.models import FairseqModel

class HFQwen2_5OmniModel(FairseqModel):
    """Represents a Qwen Omni model as found on
    huggingface.co"""

    def __init__(
        self,
        **kwargs: dict,
    ) -> None:
        super(HFQwen2_5OmniModel, self).__init__(**kwargs)
        self.model_name = "Qwen/Qwen2.5-Omni-7B"
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(self.model_name, attn_implementation, **kwargs)

    def forward(self, **inputs):
        logits = self.model(**inputs)
        return logits
    
class HFModel:
    """Represents a PyTorch model as can be found
    on huggingface.co"""

    def create_omni_model(model_name:str, **kwargs):
        if model_name == "Qwen/Qwen2.5-Omni-7B":
            return HFQwen2_5OmniModel(**kwargs)
        else:
            print(f"Model not currently supported")
    
class HFProcessor:
    """Represents a processor as can be found
    on huggingface.co"""

    def create_omni_processor(model_name, **kwargs):
        if model_name == "Qwen/Qwen2.5-Omni-7B":
            return Qwen2_5OmniProcessor.from_pretrained(model_name, **kwargs)
        else:
            return AutoProcessor.from_pretrained(model_name, **kwargs)
