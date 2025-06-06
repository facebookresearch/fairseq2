# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.utils.checkpoint import convert_checkpoint


def convert_to_reference_llama_checkpoint(
    checkpoint: dict[str, object],
) -> dict[str, object]:
    """Convert a fairseq2 LLaMA checkpoint to the reference format."""
    key_map = {
        # fmt: off
        r"^decoder\.layers\.([0-9]+)\.self_attn\.q_proj\.":      r"layers.\1.attention.wq.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.k_proj\.":      r"layers.\1.attention.wk.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.v_proj\.":      r"layers.\1.attention.wv.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.output_proj\.": r"layers.\1.attention.wo.",
        r"^decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":   r"layers.\1.attention_norm.",
        r"^decoder\.layers\.([0-9]+)\.ffn\.gate_proj\.":         r"layers.\1.feed_forward.w1.",
        r"^decoder\.layers\.([0-9]+)\.ffn\.output_proj\.":       r"layers.\1.feed_forward.w2.",
        r"^decoder\.layers\.([0-9]+)\.ffn\.inner_proj\.":        r"layers.\1.feed_forward.w3.",
        r"^decoder\.layers\.([0-9]+)\.ffn_layer_norm\.":         r"layers.\1.ffn_norm.",
        r"^decoder\.layer_norm\.":                               r"norm.",
        r"^decoder_frontend\.embed\.":                           r"tok_embeddings.",
        r"^final_proj\.":                                        r"output.",
        # fmt: on
    }

    return convert_checkpoint(checkpoint, key_map)
