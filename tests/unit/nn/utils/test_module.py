# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import Parameter

from fairseq2.models.nllb import create_nllb_model, nllb_archs
from fairseq2.nn.utils.module import select_parameters
from fairseq2.typing import Device


def test_select_parameters() -> None:
    config = nllb_archs.get_config("dense_1b")

    model = create_nllb_model(config, device=Device("meta"))

    output = select_parameters(model, [r".*\.encoder_decoder_attn_layer_norm\.bias$"])

    for idx, (name, param) in enumerate(output):
        assert name == f"decoder.layers.{idx}.encoder_decoder_attn_layer_norm.bias"

        assert isinstance(param, Parameter)

    assert idx == config.num_decoder_layers - 1


def test_select_parameters_when_exclude_is_true() -> None:
    config = nllb_archs.get_config("dense_1b")

    model = create_nllb_model(config, device=Device("meta"))

    names = [r".*\.encoder_decoder_attn_layer_norm\.bias$", "decoder.layer_norm.weight"]

    output = select_parameters(model, names, exclude=True)

    for idx, (name, param) in enumerate(output):
        assert name != "decoder.layer_norm.weight"

        assert not name.endswith("encoder_decoder_attn_layer_norm.bias")

        assert isinstance(param, Parameter)

    num_params = len(list(model.parameters())) - config.num_decoder_layers - 1

    assert idx == num_params - 1
