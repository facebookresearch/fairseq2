# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, final

import torch

from fairseq2.assets import asset_store, download_manager
from fairseq2.models.utils.checkpoint_loader import upgrade_fairseq_checkpoint
from fairseq2.models.utils.model_loader import ModelConfigLoader, ModelLoader
from fairseq2.models.w2vbert.builder import (
    W2VBertConfig,
    create_w2vbert_model,
    w2vbert_archs,
)
from fairseq2.models.w2vbert.model import W2VBertModel
from fairseq2.typing import finaloverride


@final
class W2VBertLoader(ModelLoader[W2VBertModel, W2VBertConfig]):
    """Loads w2v-BERT models."""

    @finaloverride
    def _upgrade_checkpoint(
        self, checkpoint: Mapping[str, Any], config: W2VBertConfig
    ) -> Mapping[str, Any]:
        state_dict = checkpoint["model"]

        # Check if we have a fairseq2 checkpoint.
        if "w2v2.final_target_proj.weight" in state_dict:
            return checkpoint

        state_dict["w2v2.quantizer.num_updates"] = torch.zeros((), device="cpu")

        key_map = self._fairseq_key_map()

        return upgrade_fairseq_checkpoint(checkpoint, key_map)

    @staticmethod
    def _fairseq_key_map() -> Dict[str, str]:
        return {
            # fmt: off
            r"^encoder\.pos_conv\.0\.":                                    r"w2v2.encoder_frontend.pos_encoder.conv.",
            r"^layer_norm\.":                                              r"w2v2.encoder_frontend.post_extract_layer_norm.",
            r"^mask_emb":                                                  r"w2v2.masker.temporal_mask_embed",
            r"^post_extract_proj\.":                                       r"w2v2.encoder_frontend.model_dim_proj.",
            r"^quantizer\.vars":                                           r"w2v2.quantizer.entries",
            r"^quantizer\.weight_proj\.":                                  r"w2v2.quantizer.entry_proj.",
            r"^final_proj\.":                                              r"w2v2.final_proj.",
            r"^project_q\.":                                               r"w2v2.final_target_proj.",
            r"^encoder\.layers\.([0-9]+)\.conv_module\.batch_norm\.":      r"w2v2.encoder.layers.\1.conv.batch_norm.",
            r"^encoder\.layers\.([0-9]+)\.conv_module\.depthwise_conv\.":  r"w2v2.encoder.layers.\1.conv.depthwise_conv.",
            r"^encoder\.layers\.([0-9]+)\.conv_module\.layer_norm\.":      r"w2v2.encoder.layers.\1.conv_layer_norm.",
            r"^encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv1\.": r"w2v2.encoder.layers.\1.conv.pointwise_conv1.",
            r"^encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv2\.": r"w2v2.encoder.layers.\1.conv.pointwise_conv2.",
            r"^encoder\.layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":         r"w2v2.encoder.layers.\1.ffn\2_layer_norm.",
            r"^encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                r"w2v2.encoder.layers.\1.ffn\2.inner_proj.",
            r"^encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                r"w2v2.encoder.layers.\1.ffn\2.output_proj.",
            r"^encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":         r"w2v2.encoder.layers.\1.self_attn_layer_norm.",
            r"^encoder\.layers\.([0-9]+)\.self_attn\.linear_q\.":          r"w2v2.encoder.layers.\1.self_attn.q_proj.",
            r"^encoder\.layers\.([0-9]+)\.self_attn\.linear_k\.":          r"w2v2.encoder.layers.\1.self_attn.k_proj.",
            r"^encoder\.layers\.([0-9]+)\.self_attn\.linear_v\.":          r"w2v2.encoder.layers.\1.self_attn.v_proj.",
            r"^encoder\.layers\.([0-9]+)\.self_attn\.linear_out\.":        r"w2v2.encoder.layers.\1.self_attn.output_proj.",
            r"^encoder\.layers\.([0-9]+)\.self_attn\.linear_pos\.":        r"w2v2.encoder.layers.\1.self_attn.sdpa.r_proj.",
            r"^encoder\.layers\.([0-9]+)\.self_attn\.pos_bias_u":          r"w2v2.encoder.layers.\1.self_attn.sdpa.u_bias",
            r"^encoder\.layers\.([0-9]+)\.self_attn\.pos_bias_v":          r"w2v2.encoder.layers.\1.self_attn.sdpa.v_bias",
            r"^encoder\.layers\.([0-9]+)\.final_layer_norm\.":             r"w2v2.encoder.layers.\1.layer_norm.",
            r"^encoder\.layer_norm\.":                                     r"w2v2.encoder.layer_norm.",
            r"^mlm_proj\.":                                                r"final_bert_proj.",
            # fmt: on
        }


load_w2vbert_model = W2VBertLoader(
    asset_store, download_manager, create_w2vbert_model, w2vbert_archs
)


load_w2vbert_config = ModelConfigLoader[W2VBertConfig](asset_store, w2vbert_archs)
