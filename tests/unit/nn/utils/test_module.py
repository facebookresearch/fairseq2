# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Parameter

from fairseq2 import get_runtime_context
from fairseq2.device import META_DEVICE
from fairseq2.models.transformer import TransformerConfig, TransformerFactory
from fairseq2.nn.utils.module import select_parameters


def test_select_parameters() -> None:
    context = get_runtime_context()

    model_config_registry = context.get_config_registry(TransformerConfig)

    model_config = model_config_registry.get("nllb_dense_1b")

    model_factory = TransformerFactory(model_config)

    with META_DEVICE:
        model = model_factory.create_model()

    output = select_parameters(model, [r".*\.encoder_decoder_attn_layer_norm\.bias$"])

    for idx, (name, param) in enumerate(output):
        assert name == f"decoder.layers.{idx}.encoder_decoder_attn_layer_norm.bias"

        assert isinstance(param, Parameter)

    assert idx == model_config.num_decoder_layers - 1


def test_select_parameters_when_exclude_is_true() -> None:
    context = get_runtime_context()

    model_config_registry = context.get_config_registry(TransformerConfig)

    model_config = model_config_registry.get("nllb_dense_1b")

    model_factory = TransformerFactory(model_config)

    with META_DEVICE:
        model = model_factory.create_model()

    names = [r".*\.encoder_decoder_attn_layer_norm\.bias$", "decoder.layer_norm.weight"]

    output = select_parameters(model, names, exclude=True)

    for idx, (name, param) in enumerate(output):
        assert name != "decoder.layer_norm.weight"

        assert not name.endswith("encoder_decoder_attn_layer_norm.bias")

        assert isinstance(param, Parameter)

    num_params = len(list(model.parameters())) - model_config.num_decoder_layers - 1

    assert idx == num_params - 1
