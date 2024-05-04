# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Final, Optional

from fairseq2.config_registry import ConfigRegistry
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrModel
from fairseq2.models.wav2vec2.factory import (
    Wav2Vec2EncoderBuilder,
    Wav2Vec2EncoderConfig,
    wav2vec2_encoder_archs,
)
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker
from fairseq2.typing import DataType, Device

WAV2VEC2_ASR_FAMILY: Final = "wav2vec2_asr"


def _base_10h_encoder() -> Wav2Vec2EncoderConfig:
    config = wav2vec2_encoder_archs.get("base")

    config.feature_gradient_scale = 1.0
    config.dropout_p = 0.0
    config.attn_dropout_p = 0.0
    config.ffn_inner_dropout_p = 0.1

    return config


def _base_100h_encoder() -> Wav2Vec2EncoderConfig:
    config = _base_10h_encoder()

    config.layer_drop_p = 0.1

    return config


def _ls960_encoder() -> Wav2Vec2EncoderConfig:
    config = wav2vec2_encoder_archs.get("large_ls960")

    config.feature_gradient_scale = 1.0
    config.dropout_p = 0.0
    config.attn_dropout_p = 0.0
    config.ffn_inner_dropout_p = 0.1
    config.layer_drop_p = 0.1

    return config


def _lv60k_encoder() -> Wav2Vec2EncoderConfig:
    config = wav2vec2_encoder_archs.get("large_lv60k")

    config.feature_gradient_scale = 1.0
    config.dropout_p = 0.0
    config.attn_dropout_p = 0.0
    config.ffn_inner_dropout_p = 0.1
    config.layer_drop_p = 0.1

    return config


@dataclass
class Wav2Vec2AsrConfig:
    """Holds the configuration of a wav2vec 2.0 ASR model.

    The default values correspond to the base 10h architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    encoder_config: Wav2Vec2EncoderConfig = field(default_factory=_base_10h_encoder)
    """The configuration of the encoder."""

    final_dim: int = 32
    """The dimensionality of the final projection."""

    final_dropout_p: float = 0.0
    """The dropout probability on the output of the encoder."""

    # Mask
    use_masking: bool = True
    """If ``True``, masks features as regularization."""

    temporal_mask_span_len: int = 10
    """The length of each temporal mask span that is applied over time steps."""

    max_temporal_mask_prob: float = 0.70
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


wav2vec2_asr_archs = ConfigRegistry[Wav2Vec2AsrConfig]()

wav2vec2_asr_arch = wav2vec2_asr_archs.decorator


@wav2vec2_asr_arch("base_10h")
def _base_10h() -> Wav2Vec2AsrConfig:
    return Wav2Vec2AsrConfig()


@wav2vec2_asr_arch("base_100h")
def _base_100h() -> Wav2Vec2AsrConfig:
    config = _base_10h()

    config.encoder_config = _base_100h_encoder()
    return config


@wav2vec2_asr_arch("large_ls960_10h")
def _large_ls960_10h() -> Wav2Vec2AsrConfig:
    """wav2vec2 large arch trained on the Librispeech 960h dataset."""
    return Wav2Vec2AsrConfig(
        encoder_config=_ls960_encoder(),
        max_temporal_mask_prob=0.80,
        max_spatial_mask_prob=0.30,
    )


@wav2vec2_asr_arch("large_ls960_100h")
def _large_ls960_100h() -> Wav2Vec2AsrConfig:
    """wav2vec2 large arch trained on the Librispeech 960h dataset."""
    return Wav2Vec2AsrConfig(
        encoder_config=_ls960_encoder(),
        max_temporal_mask_prob=0.53,
        max_spatial_mask_prob=0.55,
    )


@wav2vec2_asr_arch("large_lv60k_10h")
def _large_lv60k_10h() -> Wav2Vec2AsrConfig:
    """wav2vec2 large arch trained on the LibriVox 60k dataset."""
    return Wav2Vec2AsrConfig(
        encoder_config=_lv60k_encoder(),
        max_temporal_mask_prob=0.80,
        max_spatial_mask_prob=0.30,
    )


@wav2vec2_asr_arch("large_lv60k_100h")
def _large_lv60k_100h() -> Wav2Vec2AsrConfig:
    """wav2vec2 large arch trained on the LibriVox 60k dataset."""
    return Wav2Vec2AsrConfig(
        encoder_config=_lv60k_encoder(),
        max_temporal_mask_prob=0.53,
        max_spatial_mask_prob=0.55,
    )


class Wav2Vec2AsrBuilder:
    """Builds modules of a wav2vec 2.0 ASR model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    _config: Wav2Vec2AsrConfig
    _encoder_builder: Wav2Vec2EncoderBuilder
    _device: Optional[Device]
    _dtype: Optional[DataType]

    def __init__(
        self,
        config: Wav2Vec2AsrConfig,
        encoder_builder: Wav2Vec2EncoderBuilder,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration.
        :param encoder_builder_cls:
            The wav2vec 2.0 encoder builder.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self._config = config

        self._encoder_builder = encoder_builder

        self._device, self._dtype = device, dtype

    def build_model(self) -> Wav2Vec2AsrModel:
        """Build a model."""
        encoder_frontend = self._encoder_builder.build_frontend()

        encoder = self._encoder_builder.build_encoder()

        masker = self.build_masker()

        return Wav2Vec2AsrModel(
            encoder_frontend,
            encoder,
            self._config.final_dim,
            masker=masker,
            final_dropout_p=self._config.final_dropout_p,
            device=self._device,
            dtype=self._dtype,
        )

    def build_masker(self) -> Optional[Wav2Vec2Masker]:
        """Build a feature masker."""
        if not self._config.use_masking:
            return None

        return Wav2Vec2Masker(
            self._config.encoder_config.model_dim,
            self._config.temporal_mask_span_len,
            self._config.max_temporal_mask_prob,
            self._config.min_num_temporal_mask_spans,
            self._config.spatial_mask_span_len,
            self._config.max_spatial_mask_prob,
            self._config.min_num_spatial_mask_spans,
            device=self._device,
            dtype=self._dtype,
        )


def create_wav2vec2_asr_model(
    config: Wav2Vec2AsrConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> Wav2Vec2AsrModel:
    """Create a wav2vec 2.0 ASR model.

    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    encoder_builder = Wav2Vec2EncoderBuilder(
        config.encoder_config, device=device, dtype=dtype
    )

    builder = Wav2Vec2AsrBuilder(config, encoder_builder, device=device, dtype=dtype)

    return builder.build_model().set_family(WAV2VEC2_ASR_FAMILY)
