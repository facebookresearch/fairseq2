# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from fairseq2.models.nllb.config import NllbConfig
from fairseq2.models.utils.checkpoint import convert_fairseq_state_dict


def convert_nllb_state_dict(
    state_dict: dict[str, object], config: NllbConfig
) -> dict[str, object]:
    try:
        state_dict = cast(dict[str, object], state_dict["model"])
    except KeyError:
        pass

    if "decoder.embed_tokens.weight" in state_dict:  # fairseq
        key_map = {
            # fmt: off
            r"^encoder\.embed_tokens\.":                              r"encoder_frontend.embed.",
            r"^decoder\.embed_tokens\.":                              r"decoder_frontend.embed.",
            r"^encoder\.layernorm_embedding\.":                       r"encoder_frontend.layer_norm.",
            r"^decoder\.layernorm_embedding\.":                       r"decoder_frontend.layer_norm.",
            r"^decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"decoder.layers.\1.self_attn.output_proj.",
            r"^encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"encoder.layers.\1.self_attn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":  r"decoder.layers.\1.encoder_decoder_attn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn\.":            r"decoder.layers.\1.encoder_decoder_attn.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.": r"decoder.layers.\1.encoder_decoder_attn_layer_norm.",
            r"^encoder\.layers\.([0-9]+)\.fc1\.":                     r"encoder.layers.\1.ffn.inner_proj.",
            r"^decoder\.layers\.([0-9]+)\.fc1\.":                     r"decoder.layers.\1.ffn.inner_proj.",
            r"^encoder\.layers\.([0-9]+)\.fc2\.":                     r"encoder.layers.\1.ffn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.fc2\.":                     r"decoder.layers.\1.ffn.output_proj.",
            r"^encoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"encoder.layers.\1.ffn_layer_norm.",
            r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"decoder.layers.\1.ffn_layer_norm.",
            r"^decoder\.output_projection\.":                         r"final_proj.",
            # fmt: on
        }

        state_dict = convert_fairseq_state_dict(state_dict, key_map)

        embeds = cast(Tensor, state_dict["final_proj.weight"])

        # fairseq had a bug that accidentally introduced a dummy token in the
        # embedding table of NLLB-100. We just discard it.
        if embeds.size(0) == 256103:  # means NLLB-100
            embeds = embeds[:-1]

            state_dict["final_proj.weight"] = embeds

        # fairseq checkpoints have duplicate embedding weights. Ensure that we
        # use a single embedding table in fairseq2.
        state_dict["encoder_frontend.embed.weight"] = embeds
        state_dict["decoder_frontend.embed.weight"] = embeds

        # The embedding positions of the control symbols in fairseq's dict do
        # not match the SentencePiece model of the tokenizer.
        with torch.inference_mode():
            # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
            embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]

        return state_dict

    consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")

    return state_dict
