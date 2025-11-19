# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# From huggingface/accelerate: src/accelerate/utils/dataclasses.py

from __future__ import annotations

from typing import Optional
from torch.nn import Module

from transformers import PreTrainedModel
from fairseq2.nn.fsdp import FSDPWrapper


def get_auto_wrap_policy(model, wrapper, transformer_cls_names_to_wrap: Optional[list[str]] = None):
    no_split_modules = getattr(model, "_no_split_modules", None)
    default_transformer_cls_names_to_wrap = list(no_split_modules) if no_split_modules is not None else []
    if transformer_cls_names_to_wrap is None:
        transformer_cls_names_to_wrap = default_transformer_cls_names_to_wrap
    return transformer_cls_names_to_wrap


def replace_layers(model, transformer_cls_to_wrap, wrapper):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, transformer_cls_to_wrap, wrapper)
        if module.__class__.__name__ in transformer_cls_to_wrap:
            setattr(model, name, wrapper(module))


def apply_fsdp_to_hg_transformer_lm(
    model: PreTrainedModel, granularity: str, wrapper: FSDPWrapper
) -> Module:
    transformer_cls_names_to_wrap = get_auto_wrap_policy(
        model,
        wrapper,
        # TODO: make this configurable!
        transformer_cls_names_to_wrap=["Qwen2_5OmniDecoderLayer", "Qwen2_5OmniVisionBlock", "Qwen2_5OmniAudioEncoderLayer"]
    )
    replace_layers(model, set(transformer_cls_names_to_wrap), wrapper)
    return model
