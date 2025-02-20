# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast

import torch
from torch import Tensor

from fairseq2.context import RuntimeContext
from fairseq2.models import register_model_family
from fairseq2.models.utils.checkpoint import convert_fairseq_checkpoint
from fairseq2.models.wav2vec2._config import (
    WAV2VEC2_MODEL_FAMILY,
    Wav2Vec2Config,
    register_wav2vec2_configs,
)
from fairseq2.models.wav2vec2._factory import Wav2Vec2Factory
from fairseq2.models.wav2vec2._model import Wav2Vec2Model
from fairseq2.nn.transformer import TransformerNormOrder
from fairseq2.typing import CPU


def register_wav2vec2_family(context: RuntimeContext) -> None:
    default_arch = "base"

    register_model_family(
        context,
        WAV2VEC2_MODEL_FAMILY,
        Wav2Vec2Model,
        Wav2Vec2Config,
        default_arch,
        create_wav2vec2_model,
        checkpoint_converter=convert_wav2vec2_checkpoint,
    )

    register_wav2vec2_configs(context)


def create_wav2vec2_model(config: Wav2Vec2Config) -> Wav2Vec2Model:
    return Wav2Vec2Factory(config).create_model()


def convert_wav2vec2_checkpoint(
    checkpoint: dict[str, object], config: Wav2Vec2Config
) -> dict[str, object]:
    state_dict = cast(MutableMapping[str, Tensor], checkpoint["model"])

    # Check if we have a fairseq2 checkpoint.
    if "project_q.weight" not in state_dict:
        return checkpoint

    if config.encoder_config.norm_order == TransformerNormOrder.POST:
        # fmt: off
        state_dict["encoder_frontend.layer_norm.weight"] = state_dict["encoder.layer_norm.weight"]
        state_dict["encoder_frontend.layer_norm.bias"]   = state_dict["encoder.layer_norm.bias"]
        # fmt: on

        del state_dict["encoder.layer_norm.weight"]
        del state_dict["encoder.layer_norm.bias"]

    state_dict["quantizer.num_updates"] = torch.zeros((), device=CPU)

    key_map = {
        # fmt: off
        r"^encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.": r"encoder.layers.\1.self_attn.output_proj.",
        r"^encoder\.layers\.([0-9]+)\.fc1\.":                 r"encoder.layers.\1.ffn.inner_proj.",
        r"^encoder\.layers\.([0-9]+)\.fc2\.":                 r"encoder.layers.\1.ffn.output_proj.",
        r"^encoder\.layers\.([0-9]+)\.final_layer_norm\.":    r"encoder.layers.\1.ffn_layer_norm.",
        r"^encoder\.embed_tokens\.":                          r"encoder_frontend.embed.",
        r"^encoder\.pos_conv\.([0-4])\.0.":                   r"encoder_frontend.pos_encoder.layers.\1.conv.",
        r"^feature_extractor\.conv_layers\.([0-9]+)\.0\.":    r"encoder_frontend.feature_extractor.layers.\1.conv.",
        r"^feature_extractor\.conv_layers\.([0-9]+)\.2\.1\.": r"encoder_frontend.feature_extractor.layers.\1.layer_norm.",
        r"^feature_extractor\.conv_layers\.0\.2\.":           r"encoder_frontend.feature_extractor.layers.0.group_norm.",
        r"^layer_norm\.":                                     r"encoder_frontend.post_extract_layer_norm.",
        r"^post_extract_proj\.":                              r"encoder_frontend.model_dim_proj.",
        r"^mask_emb":                                         r"masker.temporal_mask_embed",
        r"^quantizer\.vars":                                  r"quantizer.entries",
        r"^quantizer\.weight_proj\.":                         r"quantizer.entry_proj.",
        r"^project_q\.":                                      r"final_target_proj.",
        # fmt: on
    }

    return convert_fairseq_checkpoint(checkpoint, key_map)
