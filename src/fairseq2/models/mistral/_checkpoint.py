# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.mistral._config import MistralConfig
from fairseq2.models.utils.checkpoint import convert_model_state_dict


def convert_mistral_checkpoint(
    checkpoint: dict[str, object], config: MistralConfig
) -> dict[str, object]:
    # Check if we have a fairseq2 checkpoint.
    if "model" in checkpoint:
        return checkpoint

    key_map = {
        # fmt: off
        r"^layers\.([0-9]+)\.attention\.wq\.":    r"decoder.layers.\1.self_attn.q_proj.",
        r"^layers\.([0-9]+)\.attention\.wk\.":    r"decoder.layers.\1.self_attn.k_proj.",
        r"^layers\.([0-9]+)\.attention\.wv\.":    r"decoder.layers.\1.self_attn.v_proj.",
        r"^layers\.([0-9]+)\.attention\.wo\.":    r"decoder.layers.\1.self_attn.output_proj.",
        r"^layers\.([0-9]+)\.attention_norm\.":   r"decoder.layers.\1.self_attn_layer_norm.",
        r"^layers\.([0-9]+)\.feed_forward\.w1\.": r"decoder.layers.\1.ffn.gate_proj.",
        r"^layers\.([0-9]+)\.feed_forward\.w2\.": r"decoder.layers.\1.ffn.output_proj.",
        r"^layers\.([0-9]+)\.feed_forward\.w3\.": r"decoder.layers.\1.ffn.inner_proj.",
        r"^layers\.([0-9]+)\.ffn_norm\.":         r"decoder.layers.\1.ffn_layer_norm.",
        r"^norm\.":                               r"decoder.layer_norm.",
        r"^tok_embeddings\.":                     r"decoder_frontend.embed.",
        r"^output\.":                             r"final_proj.",
        # fmt: on
    }

    checkpoint = convert_model_state_dict(checkpoint, key_map)

    return {"model": checkpoint}
