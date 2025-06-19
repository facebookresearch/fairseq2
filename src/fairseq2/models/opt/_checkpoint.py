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

_OPT_HG_KEY_MAP: Final = {
    # fmt: off
    r"^model\.embed_tokens\.": r"decoder_frontend.embed.",  # decoder_frontend all
    r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\.": r"decoder.layers.\1.self_attn.q_proj.",  #
    r"^model\.layers\.([0-9]+)\.self_attn\.k_proj\.": r"decoder.layers.\1.self_attn.k_proj.",  #
    r"^model\.layers\.([0-9]+)\.self_attn\.v_proj\.": r"decoder.layers.\1.self_attn.v_proj.",  #
    r"^model\.layers\.([0-9]+)\.self_attn\.out_proj\.": r"decoder.layers.\1.self_attn.output_proj.",  #
    r"^model\.layers\.([0-9]+)\.post_attention_layernorm\.": r"decoder.layers.\1.ffn_layer_norm.",
    r"^model\.layers\.([0-9]+)\.mlp\.gate_proj\.": r"decoder.layers.\1.ffn.gate_proj.",
    r"^model\.layers\.([0-9]+)\.fc1\.": r"decoder.layers.\1.ffn.inner_proj.",  #
    r"^model\.layers\.([0-9]+)\.fc2\.": r"decoder.layers.\1.ffn.output_proj.",  #
    r"^model\.layers\.([0-9]+)\.input_layernorm\.": r"decoder.layers.\1.self_attn_layer_norm.",
    r"^model\.norm\.": r"decoder.layer_norm.",
    r"^lm_head\.": r"final_proj.",  # ok
    # fmt: on
}


def convert_opt_checkpoint(checkpoint: dict[str, object], config: OPTConfig) -> dict[str, object]:
    if "tok_embeddigs.weight" in checkpoint:  # reference
        # TODO
        key_map = {
            # fmt: off
            r"^layers\.([0-9]+)\.attention\.wq\.": r"decoder.layers.\1.self_attn.q_proj.",
            r"^layers\.([0-9]+)\.attention\.wk\.": r"decoder.layers.\1.self_attn.k_proj.",
            r"^layers\.([0-9]+)\.attention\.wv\.": r"decoder.layers.\1.self_attn.v_proj.",
            r"^layers\.([0-9]+)\.attention\.wo\.": r"decoder.layers.\1.self_attn.output_proj.",
            r"^layers\.([0-9]+)\.attention_norm\.": r"decoder.layers.\1.self_attn_layer_norm.",
            r"^layers\.([0-9]+)\.feed_forward\.w1\.": r"decoder.layers.\1.ffn.gate_proj.",
            r"^layers\.([0-9]+)\.feed_forward\.w2\.": r"decoder.layers.\1.ffn.output_proj.",
            r"^layers\.([0-9]+)\.feed_forward\.w3\.": r"decoder.layers.\1.ffn.inner_proj.",
            r"^layers\.([0-9]+)\.ffn_norm\.": r"decoder.layers.\1.ffn_layer_norm.",
            r"^norm\.": r"decoder.layer_norm.",
            r"^tok_embeddings\.": r"decoder_frontend.embed.",
            r"^output\.": r"final_proj.",
            # fmt: on
        }

        return convert_checkpoint(checkpoint, key_map)

    return checkpoint
