# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.wav2vec2.factory import (
    Wav2Vec2Config,
    Wav2Vec2EncoderConfig,
    wav2vec2_arch,
    wav2vec2_encoder_arch,
)
from fairseq2.nn.transformer import TransformerNormOrder


@wav2vec2_arch("base")
def _base() -> Wav2Vec2Config:
    return Wav2Vec2Config()


@wav2vec2_arch("large")
def _large() -> Wav2Vec2Config:
    config = _base()

    config.encoder_config.model_dim = 1024
    config.encoder_config.num_encoder_layers = 24
    config.encoder_config.num_encoder_attn_heads = 16
    config.encoder_config.ffn_inner_dim = 4096
    config.encoder_config.dropout_p = 0.0
    config.encoder_config.layer_drop_p = 0.2
    config.quantized_dim = 768
    config.final_dim = 768

    return config


@wav2vec2_arch("large_lv60k")  # LibriVox 60k
def _large_lv60k() -> Wav2Vec2Config:
    config = _large()

    config.encoder_config.layer_norm_features = False
    config.encoder_config.feature_extractor_bias = True
    config.encoder_config.feature_extractor_layer_norm_convs = True
    config.encoder_config.layer_drop_p = 0.0
    config.encoder_config.norm_order = TransformerNormOrder.PRE
    config.codebook_sampling_temperature = (2.0, 0.1, 0.999995)

    return config


@wav2vec2_arch("xlsr_base")
def _xlsr_base() -> Wav2Vec2Config:
    config = _large_lv60k()
    config.encoder_config.attn_dropout_p = 0.0
    config.encoder_config.feature_gradient_scale = 1.0
    return config


@wav2vec2_arch("base_conformer")
def _base_conformer() -> Wav2Vec2Config:
    config = _xlsr_base()

    config.encoder_config.use_conformer = True
    config.encoder_config.norm_order = TransformerNormOrder.POST
    config.encoder_config.depthwise_conv_kernel_size = 31
    # pos_encoder_type

    return config


@wav2vec2_arch("1b")
def _1b() -> Wav2Vec2Config:
    config = _xlsr_base()

    config.encoder_config.model_dim = 1280
    config.encoder_config.num_encoder_layers = 48
    config.encoder_config.ffn_inner_dim = 5120
    config.encoder_config.dropout_p = 0.0
    config.quantized_dim = 1024
    config.final_dim = 1024
    config.encoder_config.first_pass_dropout_p = 0.1

    return config


@wav2vec2_arch("2b")
def _2b() -> Wav2Vec2Config:
    config = _1b()

    config.encoder_config.model_dim = 1920
    config.encoder_config.ffn_inner_dim = 7680

    return config


@wav2vec2_arch("3b")
def _3b() -> Wav2Vec2Config:
    config = _1b()

    config.encoder_config.num_encoder_layers = 60
    config.encoder_config.model_dim = 2048
    config.encoder_config.ffn_inner_dim = 8192

    return config


@wav2vec2_arch("3b_mel")
def _3b_mel() -> Wav2Vec2Config:
    config = _3b()

    config.encoder_config.use_fbank = True
    config.encoder_config.num_fbank_channels = 80
    config.encoder_config.fbank_stride = 2
    config.encoder_config.sample_fbank_every_k = 1
    config.encoder_config.feature_dim = 160

    return config


@wav2vec2_arch("3.25b")
def _3b_higher() -> Wav2Vec2Config:
    config = _1b()

    config.encoder_config.num_encoder_layers = 64
    config.encoder_config.model_dim = 2048
    config.encoder_config.ffn_inner_dim = 8192
    config.encoder_config.num_encoder_attn_heads = 32
    config.quantized_dim = 1280
    config.final_dim = 1280

    return config


@wav2vec2_arch("4b")
def _4b() -> Wav2Vec2Config:
    config = _2b()

    config.quantized_dim = 1280
    config.final_dim = 1280
    config.encoder_config.num_encoder_layers = 64
    config.encoder_config.model_dim = 2304
    config.encoder_config.ffn_inner_dim = 9216
    config.encoder_config.num_encoder_attn_heads = 32

    return config


@wav2vec2_arch("7b")
def _7b() -> Wav2Vec2Config:
    config = _3b()

    config.encoder_config.num_encoder_layers = 64
    config.encoder_config.model_dim = 3072
    config.encoder_config.ffn_inner_dim = 11520
    config.encoder_config.num_encoder_attn_heads = 48
    config.quantized_dim = 1536     # Not sure if increasing this is actually useful
    config.final_dim = 1536

    return config


@wav2vec2_arch("7b_llama")
def _7b_llama() -> Wav2Vec2Config:
    config = _7b()

    config.encoder_config.num_encoder_layers = 32
    config.encoder_config.model_dim = 4096
    config.encoder_config.ffn_inner_dim = 11008
    config.encoder_config.num_encoder_attn_heads = 32
    config.quantized_dim = 1536
    config.final_dim = 1536

    return config


@wav2vec2_arch("7b_llama_rope")
def _7b_llama_rope() -> Wav2Vec2Config:
    config = _7b_llama()

    config.encoder_config.pos_encoder_type = "rotary"

    return config


@wav2vec2_arch("7b_llama_l40")
def _7b_llama_l40() -> Wav2Vec2Config:
    config = _7b_llama()

    config.encoder_config.num_encoder_layers = 40

    return config


@wav2vec2_arch("8b_llama")
def _8b_llama() -> Wav2Vec2Config:
    config = _7b()

    config.encoder_config.num_encoder_layers = 32
    config.encoder_config.model_dim = 4096
    config.encoder_config.ffn_inner_dim = 14336
    config.encoder_config.num_encoder_attn_heads = 32
    config.quantized_dim = 1536
    config.final_dim = 1536

    return config


@wav2vec2_arch("1b_llama")
def _1b_llama() -> Wav2Vec2Config:
    config = _xlsr_base()

    config.encoder_config.model_dim = 2048
    config.encoder_config.num_encoder_layers = 16
    config.encoder_config.ffn_inner_dim = int(2048 * 4 * 1.5)
    config.encoder_config.num_encoder_attn_heads = 32
    config.encoder_config.dropout_p = 0.0
    config.quantized_dim = 1024
    config.final_dim = 1024
    config.encoder_config.first_pass_dropout_p = 0.1

    return config


@wav2vec2_arch("2b")
def _2b() -> Wav2Vec2Config:
    config = _1b()

    config.encoder_config.model_dim = 1920
    config.encoder_config.ffn_inner_dim = 7680

    return config


@wav2vec2_arch("3b")
def _3b() -> Wav2Vec2Config:
    config = _1b()

    config.encoder_config.num_encoder_layers = 60
    config.encoder_config.model_dim = 2048
    config.encoder_config.ffn_inner_dim = 8192

    return config


@wav2vec2_arch("3b_mel")
def _3b_mel() -> Wav2Vec2Config:
    config = _3b()

    config.encoder_config.use_fbank = True
    config.encoder_config.num_fbank_channels = 80
    config.encoder_config.fbank_stride = 2
    config.encoder_config.sample_fbank_every_k = 1
    config.encoder_config.feature_dim = 160

    return config


@wav2vec2_arch("3.25b")
def _3b_higher() -> Wav2Vec2Config:
    config = _1b()

    config.encoder_config.num_encoder_layers = 64
    config.encoder_config.model_dim = 2048
    config.encoder_config.ffn_inner_dim = 8192
    config.encoder_config.num_encoder_attn_heads = 32
    config.quantized_dim = 1280
    config.final_dim = 1280


@wav2vec2_arch("3b_llama")
def _3b_llama() -> Wav2Vec2Config:
    config = _1b_llama()

    config.encoder_config.model_dim = 3072
    config.encoder_config.num_encoder_layers = 28
    config.encoder_config.ffn_inner_dim = int(3072 * 4 * 1.0)
    config.encoder_config.num_encoder_attn_heads = 24
    return config


@wav2vec2_arch("4b")
def _4b() -> Wav2Vec2Config:
    config = _2b()

    config.quantized_dim = 1280
    config.final_dim = 1280
    config.encoder_config.num_encoder_layers = 64
    config.encoder_config.model_dim = 2304
    config.encoder_config.ffn_inner_dim = 9216
    config.encoder_config.num_encoder_attn_heads = 32

    return config


@wav2vec2_arch("1b_llama")
def _1b_llama() -> Wav2Vec2Config:
    config = _xlsr_base()

    config.encoder_config.model_dim = 2048
    config.encoder_config.num_encoder_layers = 16
    config.encoder_config.ffn_inner_dim = int(2048 * 4 * 1.5)
    config.encoder_config.num_encoder_attn_heads = 32
    config.encoder_config.dropout_p = 0.0
    config.quantized_dim = 1024
    config.final_dim = 1024
    config.encoder_config.first_pass_dropout_p = 0.1

    return config


@wav2vec2_arch("3b_llama")
def _3b_llama() -> Wav2Vec2Config:
    config = _1b_llama()

    config.encoder_config.model_dim = 2560
    config.encoder_config.num_encoder_layers = 32
    config.encoder_config.ffn_inner_dim = int(2560 * 4 * 1.0)
    config.quantized_dim = 2048
    config.final_dim = 2048

    return config


@wav2vec2_arch("5b")
def _5b() -> Wav2Vec2Config:
    config = _3b()

    config.encoder_config.num_encoder_layers = 96
    config.encoder_config.model_dim = 2048
    config.encoder_config.ffn_inner_dim = 8192
    config.encoder_config.num_encoder_attn_heads = 16
    config.quantized_dim = 1024
    config.final_dim = 1024

    return config


@wav2vec2_arch("7b_l120")
def _7b_l120() -> Wav2Vec2Config:
    config = _5b()

    config.encoder_config.num_encoder_layers = 128
    config.encoder_config.model_dim = 2048
    config.encoder_config.ffn_inner_dim = 8192
    config.encoder_config.num_encoder_attn_heads = 16
    config.quantized_dim = 1024
    config.final_dim = 1024

    return config


@wav2vec2_arch("pseudo_dinosr_base")
def _pseudo_dinosr_base() -> Wav2Vec2Config:
    layer_descs = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 3

    encoder_config = Wav2Vec2EncoderConfig(
        model_dim=768,
        max_seq_len=100000,
        feature_dim=512,
        use_fbank=False,
        first_pass_dropout_p=0.0,
        layer_norm_features=True,
        feature_extractor_layer_descs=layer_descs,
        feature_extractor_bias=False,
        feature_extractor_layer_norm_convs=True,
        feature_gradient_scale=0.1,
        num_fbank_channels=0,
        fbank_stride=0,
        sample_fbank_every_k=0,
        pos_encoder_type="conv",
        pos_encoder_depth=5,
        pos_conv_kernel_size=95,
        num_pos_conv_groups=16,
        use_conformer=False,
        num_encoder_layers=12,
        num_encoder_attn_heads=12,
        ffn_inner_dim=3072,
        dropout_p=0.1,
        attn_dropout_p=0.1,
        layer_drop_p=0.0,
        norm_order=TransformerNormOrder.POST,
        depthwise_conv_kernel_size=31,
    )

    return Wav2Vec2Config(
        encoder_config=encoder_config,
        final_dim=256,
        final_proj_bias=True,
        temporal_mask_span_len=10,
        max_temporal_mask_prob=0.65,
        spatial_mask_span_len=10,
        max_spatial_mask_prob=0.0,
        quantized_dim=256,
        num_codebooks=2,
        num_codebook_entries=320,
        codebook_sampling_temperature=(2.0, 0.5, 0.999995),
        num_distractors=100,
        logit_temp=0.1,
    )


@wav2vec2_encoder_arch("base")
def _base_encoder() -> Wav2Vec2EncoderConfig:
    config = _base()

    return config.encoder_config


@wav2vec2_encoder_arch("large")
def _large_encoder() -> Wav2Vec2EncoderConfig:
    config = _large()

    return config.encoder_config


@wav2vec2_encoder_arch("large_lv60k")  # LibriVox 60k
def _large_lv60k_encoder() -> Wav2Vec2EncoderConfig:
    config = _large_lv60k()

    return config.encoder_config


@wav2vec2_encoder_arch("1b")
def _1b_encoder() -> Wav2Vec2EncoderConfig:
    config = _1b()

    return config.encoder_config


@wav2vec2_encoder_arch("1b_llama")
def _1b_llama_encoder() -> Wav2Vec2EncoderConfig:
    config = _1b_llama()

    return config.encoder_config


@wav2vec2_encoder_arch("2b")
def _2b_encoder() -> Wav2Vec2EncoderConfig:
    config = _2b()

    return config.encoder_config


@wav2vec2_encoder_arch("3b")
def _3b_encoder() -> Wav2Vec2EncoderConfig:
    config = _3b()

    return config.encoder_config


@wav2vec2_encoder_arch("5b")
def _5b_encoder() -> Wav2Vec2EncoderConfig:
    config = _5b()

    return config.encoder_config


@wav2vec2_encoder_arch("7b_l120")
def _7b_l120_encoder() -> Wav2Vec2EncoderConfig:
    config = _7b_l120()

    return config.encoder_config


@wav2vec2_encoder_arch("7b_llama")
def _7b_llama_encoder() -> Wav2Vec2EncoderConfig:
    config = _7b_llama()

    return config.encoder_config


@wav2vec2_encoder_arch("3.25b")
def _3b_higher_encoder() -> Wav2Vec2EncoderConfig:
    config = _3b_higher()

    return config.encoder_config
