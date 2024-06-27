# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Final, Optional

from fairseq2.data import VocabularyInfo
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrModel
from fairseq2.models.wav2vec2.factory import (
    Wav2Vec2EncoderBuilder,
    Wav2Vec2EncoderConfig,
)
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker
from fairseq2.typing import DataType, Device

WAV2VEC2_ASR_FAMILY: Final = "wav2vec2_asr"


@dataclass
class Wav2Vec2AsrConfig:
    """Holds the configuration of a wav2vec 2.0 ASR model.

    The default values correspond to the base 10h architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    encoder_config: Wav2Vec2EncoderConfig = field(
        default_factory=lambda: Wav2Vec2EncoderConfig(
            feature_gradient_scale=1.0,
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
            self._config.vocab_info,
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
