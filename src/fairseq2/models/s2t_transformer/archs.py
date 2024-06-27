# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.config_registry import ConfigRegistry
from fairseq2.data import VocabularyInfo
from fairseq2.models.s2t_transformer.factory import S2TTransformerConfig

s2t_transformer_archs = ConfigRegistry[S2TTransformerConfig]()

s2t_transformer_arch = s2t_transformer_archs.decorator


def _tiny() -> S2TTransformerConfig:
    config = _medium()

    config.model_dim = 256
    config.num_encoder_layers = 6
    config.num_decoder_layers = 3
    config.num_encoder_attn_heads = 4
    config.num_decoder_attn_heads = 4
    config.ffn_inner_dim = 256 * 4
    config.dropout_p = 0.3

    return config


def _small() -> S2TTransformerConfig:
    config = _medium()

    config.model_dim = 256
    config.num_encoder_attn_heads = 4
    config.num_decoder_attn_heads = 4
    config.ffn_inner_dim = 256 * 8
    config.dropout_p = 0.1

    return config


def _medium() -> S2TTransformerConfig:
    return S2TTransformerConfig()


def _large() -> S2TTransformerConfig:
    config = _medium()

    config.model_dim = 1024
    config.num_encoder_attn_heads = 16
    config.num_decoder_attn_heads = 16
    config.ffn_inner_dim = 1024 * 4
    config.dropout_p = 0.2

    return config


def _conformer_medium() -> S2TTransformerConfig:
    return S2TTransformerConfig(
        model_dim=256,
        max_source_seq_len=6000,
        num_fbank_channels=80,
        max_target_seq_len=1024,
        target_vocab_info=VocabularyInfo(
            size=181, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1
        ),
        use_relative_pos=False,
        use_conformer=True,
        num_encoder_layers=12,
        num_decoder_layers=6,
        num_encoder_attn_heads=4,
        num_decoder_attn_heads=8,
        ffn_inner_dim=512 * 4,
        dropout_p=0.1,
        depthwise_conv_kernel_size=31,
    )


def _register_s2t_transformer_archs() -> None:
    # fmt: off
    s2t_transformer_archs.register("tiny",   _tiny)
    s2t_transformer_archs.register("small",  _small)
    s2t_transformer_archs.register("medium", _medium)
    s2t_transformer_archs.register("large",  _large)
    s2t_transformer_archs.register("conformer_medium", _conformer_medium)
    # fmt: on
