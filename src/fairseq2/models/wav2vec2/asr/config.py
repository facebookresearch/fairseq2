# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.data import VocabularyInfo
from fairseq2.models.wav2vec2 import Wav2Vec2Config, Wav2Vec2EncoderConfig
from fairseq2.runtime.config_registry import ConfigRegistrar, get_config
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver

WAV2VEC2_ASR_MODEL_FAMILY: Final = "wav2vec2_asr"

@dataclass(kw_only=True)
class Wav2Vec2AsrConfig:
    """Holds the configuration of a wav2vec 2.0 ASR model.

    The default values correspond to the base 10h architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    encoder_config: Wav2Vec2EncoderConfig = field(
        default_factory=lambda: Wav2Vec2EncoderConfig(
            feature_grad_scale=1.0,
            dropout_p=0.0,
            attn_dropout_p=0.0,
            ffn_inner_dropout_p=0.1,
        )
    )
    """The configuration of the encoder."""

    vocab_info: VocabularyInfo = field(
        default_factory=lambda: VocabularyInfo(
            size=32, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1
        )
    )
    """The vocabulary information."""

    final_dropout_p: float = 0.0
    """The dropout probability on the output of the encoder."""

    # Mask
    mask_codebase: str = "fairseq2"

    use_masking: bool = True
    """If ``True``, masks features as regularization."""

    temporal_mask_span_len: int = 10
    """The length of each temporal mask span that is applied over time steps."""

    max_temporal_mask_prob: float = 0.69
    """The maximum probability of masking a time step. Note that, due to mask
    span overlap, the effective probability will be lower."""

    min_num_temporal_mask_spans: int = 2
    """The minimum number of temporal masks sampled per sequence."""

    spatial_mask_span_len: int = 64
    """The length of each spatial mask span that is applied over features."""

    max_spatial_mask_prob: float = 0.55
    """The maximum probability of masking a feature. Note that, due to mask span
    overlap, the effective probability will be lower."""

    min_num_spatial_mask_spans: int = 2
    """The minimum number of spatial masks sampled per sequence."""


def _register_wav2vec2_asr_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, Wav2Vec2AsrConfig)

    @arch("base_10h")
    def base_10h() -> Wav2Vec2AsrConfig:
        return Wav2Vec2AsrConfig()

    @arch("base_100h")
    def base_100h() -> Wav2Vec2AsrConfig:
        config = base_10h()

        config.encoder_config.layer_drop_p = 0.1

        return config

    @arch("large_10h", resolver=True)
    def large_10h(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()

        w2v2_config = get_config(resolver, Wav2Vec2Config, "large")

        config.encoder_config = w2v2_config.encoder_config
        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.max_temporal_mask_prob = 0.80
        config.max_spatial_mask_prob = 0.30

        return config

    @arch("large_100h", resolver=True)
    def large_100h(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = large_10h(resolver)
        config.max_temporal_mask_prob = 0.53
        config.max_spatial_mask_prob = 0.55

        return config

    @arch("large_lv60k_10h", resolver=True)
    def large_lv60k_10h(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()

        w2v2_config = get_config(resolver, Wav2Vec2Config, "large_lv60k")

        config.encoder_config = w2v2_config.encoder_config
        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.max_temporal_mask_prob = 0.80
        config.max_spatial_mask_prob = 0.30

        return config

    @arch("large_lv60k_100h", resolver=True)
    def large_lv60k_100h(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = large_lv60k_10h(resolver)
        config.max_temporal_mask_prob = 0.53
        config.max_spatial_mask_prob = 0.55

        return config

    @arch("300m_bib61", resolver=True)
    def bib61_300m(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()
        w2v2_config = get_config(resolver, Wav2Vec2Config, "large_lv60k")
        config.encoder_config = w2v2_config.encoder_config

        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.vocab_info.size = 2475

        return config

    @arch("300m_bib1143", resolver=True)
    def bib1143_300m(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = bib61_300m(resolver=resolver)
        config.vocab_info.size = 3335
        return config

    @arch("1b_bib61", resolver=True)
    def bib61_1b(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()
        w2v2_config = get_config(resolver, Wav2Vec2Config, "1b")
        config.encoder_config = w2v2_config.encoder_config

        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.vocab_info.size = 2475

        return config

    @arch("1b_llama_bib61", resolver=True)
    def llama_bib61_1b(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()
        w2v2_config = get_config(resolver, Wav2Vec2Config, "1b")
        config.encoder_config = w2v2_config.encoder_config

        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.vocab_info.size = 2475

        return config

    @arch("2b_bib61", resolver=True)
    def bib61_2b(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()
        w2v2_config = get_config(resolver, Wav2Vec2Config, "2b")
        config.encoder_config = w2v2_config.encoder_config

        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.vocab_info.size = 2475

        return config

    @arch("3b_bib61", resolver=True)
    def bib61_3b(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()
        w2v2_config = get_config(resolver, Wav2Vec2Config, "3b")
        config.encoder_config = w2v2_config.encoder_config

        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.vocab_info.size = 2475

        return config

    @arch("5b_bib61", resolver=True)
    def bib61_5b(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()
        w2v2_config = get_config(resolver, Wav2Vec2Config, "5b")
        config.encoder_config = w2v2_config.encoder_config

        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.vocab_info.size = 2475

        return config

    @arch("7b_bib61", resolver=True)
    def bib61_7b(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()
        w2v2_config = get_config(resolver, Wav2Vec2Config, "7b")
        config.encoder_config = w2v2_config.encoder_config

        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.vocab_info.size = 2475

        return config

    @arch("3.25b_bib61", resolver=True)
    def higher_bib61_3b(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()
        w2v2_config = get_config(resolver, Wav2Vec2Config, "3.25b")
        config.encoder_config = w2v2_config.encoder_config

        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.vocab_info.size = 2475

        return config

    @arch("5b_front51", resolver=True)
    def front51_5b(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()
        w2v2_config = get_config(resolver, Wav2Vec2Config, "5b")
        config.encoder_config = w2v2_config.encoder_config

        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.vocab_info.size = 222

        return config

    @arch("7b_front51", resolver=True)
    def front51_7b(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()
        w2v2_config = get_config(resolver, Wav2Vec2Config, "7b")
        config.encoder_config = w2v2_config.encoder_config

        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.vocab_info.size = 222

        return config

    @arch("1b_bib1143", resolver=True)
    def bib1143_1b(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()
        w2v2_config = get_config(resolver, Wav2Vec2Config, "1b")
        config.encoder_config = w2v2_config.encoder_config

        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.vocab_info.size = 3335

        return config

    @arch("3b_bib1143", resolver=True)
    def bib1143_3b(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()
        w2v2_config = get_config(resolver, Wav2Vec2Config, "3b")
        config.encoder_config = w2v2_config.encoder_config

        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.vocab_info.size = 3335

        return config

    @arch("5b_bib1143", resolver=True)
    def bib1143_5b(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()
        w2v2_config = get_config(resolver, Wav2Vec2Config, "5b")
        config.encoder_config = w2v2_config.encoder_config

        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.vocab_info.size = 3335  # following bibfront1194's vocab size

        return config

    @arch("7b_bib1143", resolver=True)
    def bib1143_7b(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = base_10h()
        w2v2_config = get_config(resolver, Wav2Vec2Config, "7b")
        config.encoder_config = w2v2_config.encoder_config

        config.encoder_config.feature_grad_scale = 1.0
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.vocab_info.size = 3335

        return config
