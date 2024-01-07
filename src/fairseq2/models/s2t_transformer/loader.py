# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Dict, final

from fairseq2.assets import AssetCard, asset_store, download_manager
from fairseq2.models.s2t_transformer.builder import (
    S2TTransformerConfig,
    create_s2t_transformer_model,
    s2t_transformer_archs,
)
from fairseq2.models.s2t_transformer.tokenizer import S2TTransformerTokenizer
from fairseq2.models.transformer import TransformerModel
from fairseq2.models.utils import ConfigLoader, ModelLoader, TokenizerLoaderBase
from fairseq2.models.utils.checkpoint import convert_fairseq_checkpoint
from fairseq2.typing import finaloverride


def convert_s2t_transformer_checkpoint(
    checkpoint: Dict[str, Any], config: S2TTransformerConfig
) -> Dict[str, Any]:
    """Convert a fairseq S2T Transformer checkpoint to fairseq2."""
    key_map = {
        # fmt: off
        r"^encoder\.subsample\.conv_layers\.([0-9]+)\.":                   r"encoder_frontend.feature_extractor.layers.\1.conv.",
        r"^encoder\.transformer_layers\.([0-9]+)\.self_attn_layer_norm\.": r"encoder.layers.\1.self_attn_layer_norm.",
        r"^encoder\.transformer_layers\.([0-9]+)\.self_attn\.out_proj\.":  r"encoder.layers.\1.self_attn.output_proj.",
        r"^encoder\.transformer_layers\.([0-9]+)\.self_attn\.":            r"encoder.layers.\1.self_attn.",
        r"^encoder\.transformer_layers\.([0-9]+)\.final_layer_norm\.":     r"encoder.layers.\1.ffn_layer_norm.",
        r"^encoder\.transformer_layers\.([0-9]+)\.fc1\.":                  r"encoder.layers.\1.ffn.inner_proj.",
        r"^encoder\.transformer_layers\.([0-9]+)\.fc2\.":                  r"encoder.layers.\1.ffn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":           r"decoder.layers.\1.encoder_decoder_attn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":           r"decoder.layers.\1.encoder_decoder_attn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":              r"decoder.layers.\1.self_attn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.":                     r"decoder.layers.\1.encoder_decoder_attn.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.":          r"decoder.layers.\1.encoder_decoder_attn_layer_norm.",
        r"^decoder\.layers\.([0-9]+)\.fc1\.":                              r"decoder.layers.\1.ffn.inner_proj.",
        r"^decoder\.layers\.([0-9]+)\.fc2\.":                              r"decoder.layers.\1.ffn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":                 r"decoder.layers.\1.ffn_layer_norm.",
        r"^decoder\.embed_tokens\.":                                       r"decoder_frontend.embed.",
        r"^decoder\.output_projection\.":                                  r"final_proj.",

        # S2T Conformer
        r"^encoder\.linear\.":                                                   r"encoder_frontend.proj.",
        r"^encoder\.conformer_layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":         r"encoder.layers.\1.ffn\2_layer_norm.",
        r"^encoder\.conformer_layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                r"encoder.layers.\1.ffn\2.inner_proj.",
        r"^encoder\.conformer_layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                r"encoder.layers.\1.ffn\2.output_proj.",
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn_layer_norm\.":         r"encoder.layers.\1.self_attn_layer_norm.",
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_q\.":          r"encoder.layers.\1.self_attn.q_proj.",
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_k\.":          r"encoder.layers.\1.self_attn.k_proj.",
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_v\.":          r"encoder.layers.\1.self_attn.v_proj.",
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_out\.":        r"encoder.layers.\1.self_attn.output_proj.",
        r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.layer_norm\.":      r"encoder.layers.\1.conv_layer_norm.",
        r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.pointwise_conv1\.": r"encoder.layers.\1.conv.pointwise_conv1.",
        r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.depthwise_conv\.":  r"encoder.layers.\1.conv.depthwise_conv.",
        r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.batch_norm\.":      r"encoder.layers.\1.conv.batch_norm.",
        r"^encoder\.conformer_layers\.([0-9]+)\.conv_module\.pointwise_conv2\.": r"encoder.layers.\1.conv.pointwise_conv2.",
        r"^encoder\.conformer_layers\.([0-9]+)\.final_layer_norm\.":             r"encoder.layers.\1.layer_norm.",

        # S2T Conformer - RelPos
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.pos_bias_u":   r"encoder.layers.\1.self_attn.sdpa.u_bias",
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.pos_bias_v":   r"encoder.layers.\1.self_attn.sdpa.v_bias",
        r"^encoder\.conformer_layers\.([0-9]+)\.self_attn\.linear_pos\.": r"encoder.layers.\1.self_attn.sdpa.r_proj.",
        # fmt: on
    }

    return convert_fairseq_checkpoint(checkpoint, key_map)


@final
class S2TTransformerTokenizerLoader(TokenizerLoaderBase[S2TTransformerTokenizer]):
    """Loads tokenizers used by S2T Transformer models."""

    @finaloverride
    def _load(self, path: Path, card: AssetCard) -> S2TTransformerTokenizer:
        task = card.field("task").as_one_of({"translation", "transcription"})

        target_langs = card.field("target_langs").as_list(str)

        return S2TTransformerTokenizer(
            path, task, set(target_langs), default_target_lang=target_langs[0]
        )


load_s2t_transformer_config = ConfigLoader[S2TTransformerConfig](
    asset_store, s2t_transformer_archs
)

load_s2t_transformer_model = ModelLoader[TransformerModel, S2TTransformerConfig](
    asset_store,
    download_manager,
    load_s2t_transformer_config,
    create_s2t_transformer_model,
    convert_s2t_transformer_checkpoint,
    restrict_checkpoints=False,
)

load_s2t_transformer_tokenizer = S2TTransformerTokenizerLoader(
    asset_store, download_manager
)
