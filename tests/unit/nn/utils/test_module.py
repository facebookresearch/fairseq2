# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import Parameter

from fairseq2.models.transformer import create_transformer_model, transformer_archs
from fairseq2.nn.utils.module import select_parameters
from fairseq2.typing import META


def test_select_parameters() -> None:
    model_config = transformer_archs.get("nllb_dense_1b")

    model = create_transformer_model(model_config, device=META)

    output = select_parameters(model, [r".*\.encoder_decoder_attn_layer_norm\.bias$"])

    for idx, (name, param) in enumerate(output):
        assert name == f"decoder.layers.{idx}.encoder_decoder_attn_layer_norm.bias"

        assert isinstance(param, Parameter)

    assert idx == model_config.num_decoder_layers - 1


def test_select_parameters_when_exclude_is_true() -> None:
    model_config = transformer_archs.get("nllb_dense_1b")

    model = create_transformer_model(model_config, device=META)

    names = [r".*\.encoder_decoder_attn_layer_norm\.bias$", "decoder.layer_norm.weight"]

    output = select_parameters(model, names, exclude=True)

    for idx, (name, param) in enumerate(output):
        assert name != "decoder.layer_norm.weight"

        assert not name.endswith("encoder_decoder_attn_layer_norm.bias")

        assert isinstance(param, Parameter)

    num_params = len(list(model.parameters())) - model_config.num_decoder_layers - 1

    assert idx == num_params - 1
