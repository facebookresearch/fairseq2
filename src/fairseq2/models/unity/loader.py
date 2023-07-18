# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, final

import torch
from overrides import override as finaloverride

from fairseq2.assets import asset_store, download_manager
from fairseq2.models.nllb.loader import NllbTokenizerLoader
from fairseq2.models.transformer import TransformerModel
from fairseq2.models.unity.builder import (
    UnitYConfig,
    UnitYS2TConfig,
    create_unity_model,
    create_unity_s2t_model,
    unity_archs,
    unity_s2t_archs,
)
from fairseq2.models.unity.model import UnitYModel
from fairseq2.models.utils.checkpoint import upgrade_fairseq_checkpoint
from fairseq2.models.utils.model_loader import ModelLoader


@final
class UnitYS2TLoader(ModelLoader[TransformerModel, UnitYS2TConfig]):
    """Loads S2T UnitY models."""

    @finaloverride
    def _upgrade_checkpoint(
        self, checkpoint: Dict[str, Any], config: UnitYS2TConfig
    ) -> Dict[str, Any]:
        key_map = self._fairseq_key_map(config)

        checkpoint = upgrade_fairseq_checkpoint(checkpoint, key_map)

        state_dict = checkpoint["model"]

        del state_dict["encoder.w2v_encoder.w2v_model.mask_emb"]

        # fairseq checkpoints have duplicate embedding weights.
        embeds = state_dict["final_proj.weight"]

        state_dict["decoder_frontend.embed.weight"] = embeds

        # The embedding positions of the control tokens do not match the
        # SentencePiece model of the tokenizer.
        with torch.inference_mode():
            # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
            embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]

        return checkpoint

    @staticmethod
    def _fairseq_key_map(config: UnitYS2TConfig) -> Dict[str, str]:
        key_map = {
            # fmt: off

            # S2T Encoder
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.pos_conv\.0\.":                                    r"encoder_frontend.pos_encoder.conv.",
            r"^encoder\.w2v_encoder\.w2v_model\.layer_norm\.":                                              r"encoder_frontend.post_extract_layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.post_extract_proj\.":                                       r"encoder_frontend.model_dim_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.batch_norm\.":      r"encoder.inner.layers.\1.conv.batch_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.depthwise_conv\.":  r"encoder.inner.layers.\1.conv.depthwise_conv.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.layer_norm\.":      r"encoder.inner.layers.\1.conv_layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv1\.": r"encoder.inner.layers.\1.conv.pointwise_conv1.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv2\.": r"encoder.inner.layers.\1.conv.pointwise_conv2.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":         r"encoder.inner.layers.\1.ffn\2_layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                r"encoder.inner.layers.\1.ffn\2.inner_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                r"encoder.inner.layers.\1.ffn\2.output_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":         r"encoder.inner.layers.\1.self_attn_layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_q\.":          r"encoder.inner.layers.\1.self_attn.q_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_k\.":          r"encoder.inner.layers.\1.self_attn.k_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_v\.":          r"encoder.inner.layers.\1.self_attn.v_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_out\.":        r"encoder.inner.layers.\1.self_attn.output_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_pos\.":        r"encoder.inner.layers.\1.self_attn.sdpa.r_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.pos_bias_u":          r"encoder.inner.layers.\1.self_attn.sdpa.u_bias",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.pos_bias_v":          r"encoder.inner.layers.\1.self_attn.sdpa.v_bias",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.final_layer_norm\.":             r"encoder.inner.layers.\1.layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layer_norm\.":                                     r"encoder.inner.layer_norm.",

            # S2T Encoder Adaptor
            r"^encoder\.adaptor\.proj\.0\.": r"encoder.proj1.",
            r"^encoder\.adaptor\.proj\.2\.": r"encoder.proj2.",
            r"^encoder\.adaptor\.out_ln\.":  r"encoder.layer_norm.",

            # ST2 Decoder
            r"^decoder\.embed_tokens\.":                              r"decoder_frontend.embed.",
            r"^decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"decoder.layers.\1.self_attn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.self_attn\.":               r"decoder.layers.\1.self_attn.",
            r"^decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":    r"decoder.layers.\1.self_attn_layer_norm.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":  r"decoder.layers.\1.encoder_decoder_attn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn\.":            r"decoder.layers.\1.encoder_decoder_attn.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.": r"decoder.layers.\1.encoder_decoder_attn_layer_norm.",
            r"^decoder\.layers\.([0-9]+)\.fc1\.":                     r"decoder.layers.\1.ffn.inner_proj.",
            r"^decoder\.layers\.([0-9]+)\.fc2\.":                     r"decoder.layers.\1.ffn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"decoder.layers.\1.ffn_layer_norm.",
            r"^decoder\.layer_norm\.":                                r"decoder.layer_norm.",
            r"^decoder\.output_projection\.":                         r"final_proj.",
            # fmt: on
        }

        # fmt: off
        if config.use_conformer_adaptor:
            key_map.update(
                {
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn\.out_proj\.":          r"encoder.adaptor_layers.\1.block.self_attn.output_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn\.":                    r"encoder.adaptor_layers.\1.block.self_attn.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn_layer_norm\.":         r"encoder.adaptor_layers.\1.block.self_attn_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":         r"encoder.adaptor_layers.\1.block.ffn\2_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                r"encoder.adaptor_layers.\1.block.ffn\2.inner_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                r"encoder.adaptor_layers.\1.block.ffn\2.output_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.batch_norm\.":      r"encoder.adaptor_layers.\1.block.conv.batch_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.depthwise_conv\.":  r"encoder.adaptor_layers.\1.block.conv.depthwise_conv.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.layer_norm\.":      r"encoder.adaptor_layers.\1.block.conv_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.pointwise_conv1\.": r"encoder.adaptor_layers.\1.block.conv.pointwise_conv1.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.pointwise_conv2\.": r"encoder.adaptor_layers.\1.block.conv.pointwise_conv2.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.final_layer_norm\.":             r"encoder.adaptor_layers.\1.block.layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_ln\.":                      r"encoder.adaptor_layers.\1.layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_pool\.1\.":                 r"encoder.adaptor_layers.\1.conv.",
                }
            )
        else:
            key_map.update(
                {
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.residual_layer_norm\.":  r"encoder.adaptor_layers.\1.residual_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.residual_pool\.1\.":     r"encoder.adaptor_layers.\1.residual_conv.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.attn_pool\.1\.":         r"encoder.adaptor_layers.\1.self_attn_conv.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn\.out_proj\.":  r"encoder.adaptor_layers.\1.self_attn.output_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn\.":            r"encoder.adaptor_layers.\1.self_attn.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn_layer_norm\.": r"encoder.adaptor_layers.\1.self_attn_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.fc1\.":                  r"encoder.adaptor_layers.\1.ffn.inner_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.fc2\.":                  r"encoder.adaptor_layers.\1.ffn.output_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.final_layer_norm\.":     r"encoder.adaptor_layers.\1.ffn_layer_norm.",
                }
            )
        # fmt: on

        return key_map


load_unity_s2t_model = UnitYS2TLoader(
    asset_store, download_manager, create_unity_s2t_model, unity_s2t_archs
)


@final
class UnitYLoader(ModelLoader[UnitYModel, UnitYConfig]):
    """Loads UnitY models."""

    @finaloverride
    def _upgrade_checkpoint(
        self, checkpoint: Dict[str, Any], config: UnitYConfig
    ) -> Dict[str, Any]:
        key_map = self._fairseq_key_map(config)

        checkpoint = upgrade_fairseq_checkpoint(checkpoint, key_map)

        state_dict = checkpoint["model"]

        del state_dict["target_letter_decoder.version"]
        del state_dict["target_letter_decoder.embed_positions._float_tensor"]
        del state_dict["encoder.w2v_encoder.w2v_model.mask_emb"]

        # TODO: Do for Unit embeddings??
        # TODO: Unit pad index?

        # fairseq checkpoints have duplicate embedding weights.
        embeds = state_dict["s2t_model.final_proj.weight"]

        state_dict["s2t_model.decoder_frontend.embed.weight"] = embeds

        # The embedding positions of the control tokens do not match the
        # SentencePiece model of the tokenizer.
        with torch.inference_mode():
            # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
            embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]

        return checkpoint

    @staticmethod
    def _fairseq_key_map(config: UnitYConfig) -> Dict[str, str]:
        key_map = {
            # fmt: off

            # S2T Encoder
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.pos_conv\.0\.":                                    r"s2t_model.encoder_frontend.pos_encoder.conv.",
            r"^encoder\.w2v_encoder\.w2v_model\.layer_norm\.":                                              r"s2t_model.encoder_frontend.post_extract_layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.post_extract_proj\.":                                       r"s2t_model.encoder_frontend.model_dim_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.batch_norm\.":      r"s2t_model.encoder.inner.layers.\1.conv.batch_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.depthwise_conv\.":  r"s2t_model.encoder.inner.layers.\1.conv.depthwise_conv.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.layer_norm\.":      r"s2t_model.encoder.inner.layers.\1.conv_layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv1\.": r"s2t_model.encoder.inner.layers.\1.conv.pointwise_conv1.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv2\.": r"s2t_model.encoder.inner.layers.\1.conv.pointwise_conv2.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":         r"s2t_model.encoder.inner.layers.\1.ffn\2_layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                r"s2t_model.encoder.inner.layers.\1.ffn\2.inner_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                r"s2t_model.encoder.inner.layers.\1.ffn\2.output_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":         r"s2t_model.encoder.inner.layers.\1.self_attn_layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_q\.":          r"s2t_model.encoder.inner.layers.\1.self_attn.q_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_k\.":          r"s2t_model.encoder.inner.layers.\1.self_attn.k_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_v\.":          r"s2t_model.encoder.inner.layers.\1.self_attn.v_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_out\.":        r"s2t_model.encoder.inner.layers.\1.self_attn.output_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_pos\.":        r"s2t_model.encoder.inner.layers.\1.self_attn.sdpa.r_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.pos_bias_u":          r"s2t_model.encoder.inner.layers.\1.self_attn.sdpa.u_bias",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.pos_bias_v":          r"s2t_model.encoder.inner.layers.\1.self_attn.sdpa.v_bias",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.final_layer_norm\.":             r"s2t_model.encoder.inner.layers.\1.layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layer_norm\.":                                     r"s2t_model.encoder.inner.layer_norm.",

            # S2T Encoder Adaptor
            r"^encoder\.adaptor\.proj\.0\.":                                        r"s2t_model.encoder.proj1.",
            r"^encoder\.adaptor\.proj\.2\.":                                        r"s2t_model.encoder.proj2.",
            r"^encoder\.adaptor\.out_ln\.":                                         r"s2t_model.encoder.layer_norm.",

            # ST2 Decoder
            r"^target_letter_decoder\.embed_tokens\.":                              r"s2t_model.decoder_frontend.embed.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"s2t_model.decoder.layers.\1.self_attn.output_proj.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.self_attn\.":               r"s2t_model.decoder.layers.\1.self_attn.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":    r"s2t_model.decoder.layers.\1.self_attn_layer_norm.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":  r"s2t_model.decoder.layers.\1.encoder_decoder_attn.output_proj.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.encoder_attn\.":            r"s2t_model.decoder.layers.\1.encoder_decoder_attn.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.": r"s2t_model.decoder.layers.\1.encoder_decoder_attn_layer_norm.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.fc1\.":                     r"s2t_model.decoder.layers.\1.ffn.inner_proj.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.fc2\.":                     r"s2t_model.decoder.layers.\1.ffn.output_proj.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"s2t_model.decoder.layers.\1.ffn_layer_norm.",
            r"^target_letter_decoder\.layer_norm\.":                                r"s2t_model.decoder.layer_norm.",
            r"^target_letter_decoder\.output_projection\.":                         r"s2t_model.final_proj.",

            # T2U Encoder
            r"^synthesizer_encoder\.linear_layers\.([0-9]+)\.":                   r"t2u_proj.\1.",
            r"^synthesizer_encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"t2u_encoder.layers.\1.self_attn.output_proj.",
            r"^synthesizer_encoder\.layers\.([0-9]+)\.self_attn\.":               r"t2u_encoder.layers.\1.self_attn.",
            r"^synthesizer_encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":    r"t2u_encoder.layers.\1.self_attn_layer_norm.",
            r"^synthesizer_encoder\.layers\.([0-9]+)\.fc1\.":                     r"t2u_encoder.layers.\1.ffn.inner_proj.",
            r"^synthesizer_encoder\.layers\.([0-9]+)\.fc2\.":                     r"t2u_encoder.layers.\1.ffn.output_proj.",
            r"^synthesizer_encoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"t2u_encoder.layers.\1.ffn_layer_norm.",
            r"^synthesizer_encoder\.layer_norm\.":                                r"t2u_encoder.layer_norm.",

            # T2U Decoder
            r"^decoder\.embed_tokens\.":                              r"t2u_decoder_frontend.embed.",
            r"^decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"t2u_decoder.layers.\1.self_attn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.self_attn\.":               r"t2u_decoder.layers.\1.self_attn.",
            r"^decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":    r"t2u_decoder.layers.\1.self_attn_layer_norm.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":  r"t2u_decoder.layers.\1.encoder_decoder_attn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn\.":            r"t2u_decoder.layers.\1.encoder_decoder_attn.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.": r"t2u_decoder.layers.\1.encoder_decoder_attn_layer_norm.",
            r"^decoder\.layers\.([0-9]+)\.fc1\.":                     r"t2u_decoder.layers.\1.ffn.inner_proj.",
            r"^decoder\.layers\.([0-9]+)\.fc2\.":                     r"t2u_decoder.layers.\1.ffn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"t2u_decoder.layers.\1.ffn_layer_norm.",
            r"^decoder\.layer_norm\.":                                r"t2u_decoder.layer_norm.",
            r"^decoder\.output_projection\.":                         r"final_proj.",
            # fmt: on
        }

        # fmt: off
        if config.s2t_model_config.use_conformer_adaptor:
            key_map.update(
                {
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn\.out_proj\.":          r"s2t_model.encoder.adaptor_layers.\1.block.self_attn.output_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn\.":                    r"s2t_model.encoder.adaptor_layers.\1.block.self_attn.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn_layer_norm\.":         r"s2t_model.encoder.adaptor_layers.\1.block.self_attn_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":         r"s2t_model.encoder.adaptor_layers.\1.block.ffn\2_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                r"s2t_model.encoder.adaptor_layers.\1.block.ffn\2.inner_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                r"s2t_model.encoder.adaptor_layers.\1.block.ffn\2.output_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.batch_norm\.":      r"s2t_model.encoder.adaptor_layers.\1.block.conv.batch_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.depthwise_conv\.":  r"s2t_model.encoder.adaptor_layers.\1.block.conv.depthwise_conv.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.layer_norm\.":      r"s2t_model.encoder.adaptor_layers.\1.block.conv_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.pointwise_conv1\.": r"s2t_model.encoder.adaptor_layers.\1.block.conv.pointwise_conv1.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.pointwise_conv2\.": r"s2t_model.encoder.adaptor_layers.\1.block.conv.pointwise_conv2.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.final_layer_norm\.":             r"s2t_model.encoder.adaptor_layers.\1.block.layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_ln\.":                      r"s2t_model.encoder.adaptor_layers.\1.layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_pool\.1\.":                 r"s2t_model.encoder.adaptor_layers.\1.conv.",
                }
            )
        else:
            key_map.update(
                {
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.residual_layer_norm\.":  r"s2t_model.encoder.adaptor_layers.\1.residual_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.residual_pool\.1\.":     r"s2t_model.encoder.adaptor_layers.\1.residual_conv.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.attn_pool\.1\.":         r"s2t_model.encoder.adaptor_layers.\1.self_attn_conv.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn\.out_proj\.":  r"s2t_model.encoder.adaptor_layers.\1.self_attn.output_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn\.":            r"s2t_model.encoder.adaptor_layers.\1.self_attn.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn_layer_norm\.": r"s2t_model.encoder.adaptor_layers.\1.self_attn_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.fc1\.":                  r"s2t_model.encoder.adaptor_layers.\1.ffn.inner_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.fc2\.":                  r"s2t_model.encoder.adaptor_layers.\1.ffn.output_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.final_layer_norm\.":     r"s2t_model.encoder.adaptor_layers.\1.ffn_layer_norm.",
                }
            )
        # fmt: on

        return key_map


load_unity_model = UnitYLoader(
    asset_store, download_manager, create_unity_model, unity_archs
)

load_unity_text_tokenizer = NllbTokenizerLoader(asset_store, download_manager)
