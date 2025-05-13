# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from fairseq2.device import CPU
from fairseq2.models.transformer import TransformerNormOrder
from fairseq2.models.utils.checkpoint import convert_fairseq_checkpoint

# isort: split

from fairseq2.models.wav2vec2._config import Wav2Vec2Config


def convert_wav2vec2_checkpoint(
    checkpoint: dict[str, object], config: Wav2Vec2Config
) -> dict[str, object]:
    try:
        checkpoint = cast(dict[str, object], checkpoint["model"])
    except KeyError:
        pass

    if "mask_emb.weight" in checkpoint:  # fairseq
        if config.encoder_config.norm_order == TransformerNormOrder.POST:
            # fmt: off
            checkpoint["encoder_frontend.layer_norm.weight"] = checkpoint["encoder.layer_norm.weight"]
            checkpoint["encoder_frontend.layer_norm.bias"]   = checkpoint["encoder.layer_norm.bias"]
            # fmt: on

            del checkpoint["encoder.layer_norm.weight"]
            del checkpoint["encoder.layer_norm.bias"]

        checkpoint["quantizer.num_updates"] = torch.zeros((), device=CPU)

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

    consume_prefix_in_state_dict_if_present(checkpoint, prefix="module.")  # legacy

    return checkpoint
