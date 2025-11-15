# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# From huggingface/accelerate: src/accelerate/utils/dataclasses.py

from __future__ import annotations

from typing import Optional
from functools import partial
from torch.nn import Module
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers import PreTrainedModel
from fairseq2.nn.fsdp import FSDPWrapper


def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class


def get_auto_wrap_policy(model, transformer_cls_names_to_wrap: Optional[list[str]] = None):
    no_split_modules = getattr(model, "_no_split_modules", None)
    default_transformer_cls_names_to_wrap = list(no_split_modules) if no_split_modules is not None else []
    auto_wrap_policy = transformer_auto_wrap_policy
    if transformer_cls_names_to_wrap is None:
        transformer_cls_names_to_wrap = default_transformer_cls_names_to_wrap
    transformer_cls_to_wrap = set()
    for layer_class in transformer_cls_names_to_wrap:
        transformer_cls = get_module_class_from_name(model, layer_class)
        if transformer_cls is None:
            raise ValueError(f"Could not find the transformer layer class {layer_class} in the model.")
        transformer_cls_to_wrap.add(transformer_cls)
    # Finally we set the auto_wrap_policy to a callable
    auto_wrap_policy = partial(
        auto_wrap_policy, transformer_layer_cls=transformer_cls_to_wrap
    )
    return auto_wrap_policy


def apply_fsdp_to_hg_transformer_lm(
    model: PreTrainedModel, granularity: str, wrapper: FSDPWrapper
) -> Module:
    auto_wrap_policy = get_auto_wrap_policy(
        model,
        # TODO: make this configurable!
        transformer_cls_names_to_wrap=["Qwen2_5OmniDecoderLayer", "Qwen2_5OmniVisionBlock", "Qwen2_5OmniAudioEncoderLayer"]
    )
    model = wrapper(model, auto_wrap_policy=auto_wrap_policy)
    if not isinstance(model, wrapper):
        model = wrapper(model)
    return model
