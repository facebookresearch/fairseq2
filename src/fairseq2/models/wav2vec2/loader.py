# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch

from fairseq2.models.config_loader import StandardModelConfigLoader
from fairseq2.models.loader import StandardModelLoader, load_model
from fairseq2.models.utils.checkpoint import convert_fairseq_checkpoint
from fairseq2.models.wav2vec2.factory import (
    WAV2VEC2_FAMILY,
    Wav2Vec2Config,
    create_wav2vec2_model,
    wav2vec2_archs,
)
from fairseq2.nn.transformer import TransformerNormOrder

load_wav2vec2_config = StandardModelConfigLoader(
    family=WAV2VEC2_FAMILY, config_kls=Wav2Vec2Config, arch_configs=wav2vec2_archs
)


def convert_wav2vec2_checkpoint(
    checkpoint: Dict[str, Any], config: Wav2Vec2Config
) -> Dict[str, Any]:
    """Convert a fairseq wav2vec 2.0 checkpoint to fairseq2 format."""
    try:
        state_dict = checkpoint["model"]
    except KeyError:
        return checkpoint

    if "project_q.weight" not in state_dict:
        return checkpoint

    if config.encoder_config.norm_order == TransformerNormOrder.POST:
        # fmt: off
        state_dict["encoder_frontend.layer_norm.weight"] = state_dict["encoder.layer_norm.weight"]
        state_dict["encoder_frontend.layer_norm.bias"]   = state_dict["encoder.layer_norm.bias"]
        # fmt: on

        del state_dict["encoder.layer_norm.weight"]
        del state_dict["encoder.layer_norm.bias"]

    state_dict["quantizer.num_updates"] = torch.zeros((), device="cpu")

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


load_wav2vec2_model = StandardModelLoader(
    config_loader=load_wav2vec2_config,
    factory=create_wav2vec2_model,
    checkpoint_converter=convert_wav2vec2_checkpoint,
    restrict_checkpoints=False,
)

load_model.register(WAV2VEC2_FAMILY, load_wav2vec2_model)
