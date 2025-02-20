# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.models import register_model_family
from fairseq2.models.mistral._config import (
    MISTRAL_MODEL_FAMILY,
    MistralConfig,
    register_mistral_configs,
)
from fairseq2.models.mistral._factory import MistralFactory
from fairseq2.models.transformer_decoder import TransformerDecoderModel
from fairseq2.models.utils.checkpoint import convert_model_state_dict


def register_mistral_family(context: RuntimeContext) -> None:
    default_arch = "7b"

    register_model_family(
        context,
        MISTRAL_MODEL_FAMILY,
        TransformerDecoderModel,
        MistralConfig,
        default_arch,
        create_mistral_model,
        checkpoint_converter=convert_mistral_checkpoint,
    )

    register_mistral_configs(context)


def create_mistral_model(config: MistralConfig) -> TransformerDecoderModel:
    return MistralFactory(config).create_model()


def convert_mistral_checkpoint(
    checkpoint: dict[str, object], config: MistralConfig
) -> dict[str, object]:
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
