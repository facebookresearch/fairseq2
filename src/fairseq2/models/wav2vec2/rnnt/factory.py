# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import torch

from fairseq2.config_registry import ConfigRegistry
from fairseq2.models.factory import model_factories
from fairseq2.models.wav2vec2.asr.archs import (
    _300m_bib61 as _300m_bib61_ctc,
    _5b_bib61 as _5b_bib61_ctc,
    _base_10h as _base_10h_ctc,
)
from fairseq2.models.wav2vec2.asr.factory import Wav2Vec2AsrBuilder, Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.factory import Wav2Vec2EncoderBuilder
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker

from fairseq2.models.wav2vec2.rnnt import beam_search_gpu
from fairseq2.models.wav2vec2.rnnt.model import Wav2Vec2RnntModel
from fairseq2.nn import Linear, StandardEmbedding
from fairseq2.typing import DataType, Device

WAV2VEC2_RNNT_FAMILY: Final = "wav2vec2_rnnt"


@dataclass(kw_only=True)
class RnntBeamSearchConfig:
    nbest: int = field(default=10)
    step_max_symbols: int = field(default=10)
    merge_beam: int = field(default=25)
    length_norm: bool = field(default=True)
    always_merge_blank: bool = field(default=True)


@dataclass(kw_only=True)
class RnntPredictorConfig:
    model_dim: int = field()
    num_layers: int = field()
    dropout: float = field()
    vocab_size: int = field()


@dataclass(kw_only=True)
class Wav2Vec2RnntConfig:
    wav2vec_ctc_config: Wav2Vec2AsrConfig = field()
    predictor_config: RnntPredictorConfig = field()
    beam_search_config: RnntBeamSearchConfig = field()


wav2vec2_rnnt_archs = ConfigRegistry[Wav2Vec2RnntConfig]()

wav2vec2_rnnt_arch = wav2vec2_rnnt_archs.decorator


@wav2vec2_rnnt_arch("base_10h_rnnt")
def _base_10h_rnnt() -> Wav2Vec2RnntConfig:
    # Mainly encoder config, masking
    wav2vec_ctc_config = _base_10h_ctc()

    # Prepare the predictor config
    predictor_config = RnntPredictorConfig(
        model_dim=wav2vec_ctc_config.encoder_config.model_dim,
        num_layers=3,
        dropout=0.1,
        vocab_size=wav2vec_ctc_config.vocab_info.size,
    )
    config = Wav2Vec2RnntConfig(
        wav2vec_ctc_config=wav2vec_ctc_config,
        predictor_config=predictor_config,
        beam_search_config=RnntBeamSearchConfig(),
    )

    return config


@wav2vec2_rnnt_arch("300m_bib61_rnnt")
def _300m_bib61_rnnt() -> Wav2Vec2RnntConfig:
    # Mainly encoder config, masking
    wav2vec_ctc_config = _300m_bib61_ctc()

    # Prepare the predictor config
    predictor_config = RnntPredictorConfig(
        model_dim=wav2vec_ctc_config.encoder_config.model_dim,
        num_layers=2,
        dropout=0.1,
        vocab_size=wav2vec_ctc_config.vocab_info.size,
    )

    config = Wav2Vec2RnntConfig(
        wav2vec_ctc_config=wav2vec_ctc_config,
        predictor_config=predictor_config,
        beam_search_config=RnntBeamSearchConfig(),
    )

    return config


@wav2vec2_rnnt_arch("5b_bib61_rnnt")
def _5b_bib61_rnnt() -> Wav2Vec2RnntConfig:
    # Mainly encoder config, masking
    wav2vec_ctc_config = _5b_bib61_ctc()

    # Prepare the predictor config
    predictor_config = RnntPredictorConfig(
        model_dim=wav2vec_ctc_config.encoder_config.model_dim,
        num_layers=2,
        dropout=0.1,
        vocab_size=wav2vec_ctc_config.vocab_info.size,
    )

    config = Wav2Vec2RnntConfig(
        wav2vec_ctc_config=wav2vec_ctc_config, predictor_config=predictor_config
    )

    return config


class Wav2Vec2RnntBuilder(Wav2Vec2AsrBuilder):
    """
    Builds modules of a wav2vec 2.0 RNN-T model.
    """

    _config: Wav2Vec2RnntConfig
    _encoder_builder: Wav2Vec2EncoderBuilder
    _device: Device | None
    _dtype: DataType | None

    def __init__(
        self,
        config: Wav2Vec2RnntConfig,
        encoder_builder: Wav2Vec2EncoderBuilder,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
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

    def build_masker(self) -> Wav2Vec2Masker | None:
        """Build a feature masker."""
        if not self._config.wav2vec_ctc_config.use_masking:
            return None

        return Wav2Vec2Masker(
            self._config.wav2vec_ctc_config.mask_codebase,
            self._config.wav2vec_ctc_config.encoder_config.model_dim,
            self._config.wav2vec_ctc_config.temporal_mask_span_len,
            self._config.wav2vec_ctc_config.max_temporal_mask_prob,
            self._config.wav2vec_ctc_config.min_num_temporal_mask_spans,
            self._config.wav2vec_ctc_config.spatial_mask_span_len,
            self._config.wav2vec_ctc_config.max_spatial_mask_prob,
            self._config.wav2vec_ctc_config.min_num_spatial_mask_spans,
            device=self._device,
            dtype=self._dtype,
        )

    def build_model(self) -> Wav2Vec2RnntModel:
        # Encoder frontend
        encoder_frontend = self._encoder_builder.build_frontend()

        # The wav2vec 2.0 encoder
        encoder = self._encoder_builder.build_encoder()

        # The wav2vec 2.0 masker
        masker = self.build_masker()

        # The RNN-T decoder, as an RNN
        text_frontend = StandardEmbedding(
            num_embeddings=self._config.predictor_config.vocab_size,
            embedding_dim=self._config.predictor_config.model_dim,
            device=self._device,
            dtype=self._dtype,
        )
        predictor = torch.nn.LSTM(
            input_size=self._config.predictor_config.model_dim,
            hidden_size=self._config.predictor_config.model_dim,
            num_layers=self._config.predictor_config.num_layers,
            batch_first=True,
            dropout=self._config.predictor_config.dropout,
            device=self._device,
            dtype=self._dtype,
        )

        joiner = Linear(
            self._config.predictor_config.model_dim,
            self._config.predictor_config.vocab_size,  # No need to increase for blank
            bias=True,
            device=self._device,
            dtype=self._dtype,
        )

        return Wav2Vec2RnntModel(
            encoder_frontend,
            encoder,
            text_frontend,
            predictor,
            joiner,
            self._config.wav2vec_ctc_config.vocab_info,
            masker=masker,
            final_dropout_p=self._config.wav2vec_ctc_config.final_dropout_p,
            beam_search_config=self._config.beam_search_config,
            device=self._device,
            dtype=self._dtype,
        )


def create_wav2vec2_rnnt_model(
    config: Wav2Vec2RnntConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> Wav2Vec2RnntModel:
    """Create a wav2vec 2.0 ASR model.

    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    encoder_builder = Wav2Vec2EncoderBuilder(
        config.wav2vec_ctc_config.encoder_config, device=device, dtype=dtype
    )

    builder = Wav2Vec2RnntBuilder(config, encoder_builder, device=device, dtype=dtype)

    return builder.build_model().set_family(WAV2VEC2_RNNT_FAMILY)


model_factories.register(
    WAV2VEC2_RNNT_FAMILY,
    create_wav2vec2_rnnt_model,
    Wav2Vec2RnntConfig,
    wav2vec2_rnnt_archs,
)
