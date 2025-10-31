# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from torch import Tensor

from transformers import AutoProcessor

from fairseq2.device import Device


class HFProcessor():
    """Represents a HuggingFace multimodal processor
    to encode and decode multiple modalities."""

    def __init__(self, model_name:str) -> None:
        self.processor = AutoProcessor.from_pretrained(model_name)

    def apply_chat_template(self, sequence) -> Tensor:
        return self.processor.apply_chat_template(sequence)

    def decode_text(self, **encoded_sequence) -> str:
        return self.processor.decode(**encoded_sequence)

def create_processor(model_name:str):
    return HFProcessor(model_name)
