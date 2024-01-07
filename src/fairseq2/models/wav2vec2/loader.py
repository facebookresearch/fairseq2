# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch

from fairseq2.assets import asset_store, download_manager
from fairseq2.models.utils import ConfigLoader, ModelLoader
from fairseq2.models.utils.checkpoint import convert_fairseq_checkpoint
from fairseq2.models.wav2vec2.builder import (
    Wav2Vec2Config,
    create_wav2vec2_model,
    wav2vec2_archs,
)
from fairseq2.models.wav2vec2.model import Wav2Vec2Model
from fairseq2.nn.transformer import TransformerNormOrder


def convert_wav2vec2_checkpoint(
    checkpoint: Dict[str, Any], config: Wav2Vec2Config
) -> Dict[str, Any]:
    """Convert a fairseq wav2vec 2.0 checkpoint to fairseq2."""
    state_dict = checkpoint["model"]

    # Check if we have a fairseq2 checkpoint.
    if "final_target_proj.weight" in state_dict:
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
        r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":    r"decoder.layers.\1.ffn_layer_norm.",
        r"^encoder\.embed_tokens\.":                          r"encoder_frontend.embed.",
        r"^encoder\.pos_conv\.0\.":                           r"encoder_frontend.pos_encoder.conv.",
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


load_wav2vec2_config = ConfigLoader[Wav2Vec2Config](asset_store, wav2vec2_archs)

load_wav2vec2_model = ModelLoader[Wav2Vec2Model, Wav2Vec2Config](
    asset_store,
    download_manager,
    load_wav2vec2_config,
    create_wav2vec2_model,
    convert_wav2vec2_checkpoint,
)
