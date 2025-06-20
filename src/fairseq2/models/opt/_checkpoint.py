# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final, cast

from fairseq2.models.utils.checkpoint import convert_checkpoint

# isort: split

from fairseq2.models.opt._config import OPTConfig

_OPT_HG_KEY_MAP = {
    # fmt: off
    r"^model\.decoder\.embed_tokens\.": r"decoder_frontend.embed.",
    r"^model\.decoder\.embed_positions\.": r"decoder_frontend.pos_encoder.",
    r"^model\.decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.": r"decoder.layers.\1.self_attn_layer_norm.",
    r"^model\.decoder\.layers\.([0-9]+)\.self_attn\.q_proj\.": r"decoder.layers.\1.self_attn.q_proj.",
    r"^model\.decoder\.layers\.([0-9]+)\.self_attn\.k_proj\.": r"decoder.layers.\1.self_attn.k_proj.",
    r"^model\.decoder\.layers\.([0-9]+)\.self_attn\.v_proj\.": r"decoder.layers.\1.self_attn.v_proj.",
    r"^model\.decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.": r"decoder.layers.\1.self_attn.output_proj.",
    r"^model\.decoder\.layers\.([0-9]+)\.mlp\.gate_proj\.": r"decoder.layers.\1.ffn.gate_proj.",
    r"^model\.decoder\.layers\.([0-9]+)\.fc1\.": r"decoder.layers.\1.ffn.inner_proj.",
    r"^model\.decoder\.layers\.([0-9]+)\.fc2\.": r"decoder.layers.\1.ffn.output_proj.",
    r"^model\.decoder\.layers\.([0-9]+)\.final_layer_norm\.": r"decoder.layers.\1.ffn_layer_norm.",
    r"^model\.decoder\.final_layer_norm\.": r"decoder.layer_norm.",
    r"^lm_head\.": r"final_proj.",
    # fmt: on
}


def convert_opt_checkpoint(checkpoint: dict[str, object], config: OPTConfig) -> dict[str, object]:
    try:
        checkpoint = cast(dict[str, object], checkpoint["model"])  # legacy
    except KeyError:
        pass

    if "model.decoder.embed_tokens.weight" in checkpoint:  # Hugging Face
        checkpoint = convert_checkpoint(checkpoint, _OPT_HG_KEY_MAP)

    return checkpoint
