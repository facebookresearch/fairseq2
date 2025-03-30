# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.config_registry import ConfigRegistry
from fairseq2.models.factory import model_factories
from fairseq2.models.llama.factory import LLaMABuilder, LLaMAConfig
from fairseq2.models.transformer import init_final_projection
from fairseq2.models.wav2vec2.asr.archs import (
    _300m_bib61 as _300m_bib61_ctc,
    _base_10h as _base_10h_ctc,
)
from fairseq2.models.wav2vec2.asr.factory import Wav2Vec2AsrBuilder, Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.factory import Wav2Vec2EncoderBuilder
from fairseq2.models.wav2vec2.llama.model import Wav2Vec2LlamaModel
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker
from fairseq2.nn import Linear
from fairseq2.typing import DataType, Device

# TODO:
# Fix in RNNT:
# 1) wrong config class in factory,
# 2) `encoder` is not a member of model,
# 3) merge eval units,
# 4) comment in compute_loss
# 5) Make beam search args configurable

WAV2VEC2_LLAMA_FAMILY: Final = "wav2vec2_llama"


@dataclass(kw_only=True)
class Wav2Vec2LlamaConfig:
    wav2vec_ctc_config: Wav2Vec2AsrConfig = field()
    llama_config: LLaMAConfig = field()


wav2vec2_llama_archs = ConfigRegistry[Wav2Vec2LlamaConfig]()

wav2vec2_llama_arch = wav2vec2_llama_archs.decorator


@wav2vec2_llama_arch("base_10h_llama")
def _base_10h_llama() -> Wav2Vec2LlamaConfig:
    # Mainly encoder config, masking
    wav2vec_ctc_config = _base_10h_ctc()

    # Prepare the llama config
    llama_config = LLaMAConfig(
        model_dim=1024,
        # max_seq_len=8192,
        max_seq_len=4096,
        vocab_info=wav2vec_ctc_config.vocab_info,
        num_layers=12,
        num_attn_heads=8,
        num_key_value_heads=8,
        ffn_inner_dim=4096,
        rope_theta=10_000.0,
        dropout_p=0.1,
    )
    config = Wav2Vec2LlamaConfig(
        wav2vec_ctc_config=wav2vec_ctc_config, llama_config=llama_config
    )

    return config


@wav2vec2_llama_arch("300m_bib61_llama")
def _300m_bib61_llama() -> Wav2Vec2LlamaConfig:
    # Mainly encoder config, masking
    wav2vec_ctc_config = _300m_bib61_ctc()

    # Prepare the predictor config
    llama_config = LLaMAConfig(
        model_dim=1024,
        # max_seq_len=8192,
        max_seq_len=4096,
        vocab_info=wav2vec_ctc_config.vocab_info,
        num_layers=12,
        num_attn_heads=8,
        num_key_value_heads=8,
        ffn_inner_dim=4096,
        rope_theta=10_000.0,
        dropout_p=0.1,
    )
    config = Wav2Vec2LlamaConfig(
        wav2vec_ctc_config=wav2vec_ctc_config, llama_config=llama_config
    )

    return config


class Wav2Vec2LlamaBuilder(Wav2Vec2AsrBuilder):
    """
    Builds modules of a wav2vec 2.0 RNN-T model.
    """

    _config: Wav2Vec2LlamaConfig
    _encoder_builder: Wav2Vec2EncoderBuilder
    _llama_builder: LLaMABuilder
    _device: Device | None
    _dtype: DataType | None

    def __init__(
        self,
        config: Wav2Vec2LlamaConfig,
        encoder_builder: Wav2Vec2EncoderBuilder,
        llama_builder: LLaMABuilder,
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
        self._llama_builder = llama_builder

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

    def build_model(self) -> Wav2Vec2LlamaModel:
        # Encoder frontend
        encoder_frontend = self._encoder_builder.build_frontend()

        # The wav2vec 2.0 encoder
        encoder = self._encoder_builder.build_encoder()

        # The wav2vec 2.0 masker
        masker = self.build_masker()

        # Projection from encoder to decoder
        encoder_proj = Linear(
            self._config.wav2vec_ctc_config.encoder_config.model_dim,
            self._config.llama_config.model_dim,
            bias=True,
            device=self._device,
            dtype=self._dtype,
        )

        # The Llama model
        text_frontend = self._llama_builder.build_decoder_frontend()
        llama_decoder = self._llama_builder.build_decoder()
        final_proj = Linear(
            self._config.llama_config.model_dim,
            self._config.llama_config.vocab_info.size,
            bias=False,
            init_fn=init_final_projection,
            device=self._device,
            dtype=self._dtype,
        )

        return Wav2Vec2LlamaModel(
            encoder_frontend,
            encoder,
            encoder_proj,
            text_frontend,
            llama_decoder,
            final_proj,
            self._config.wav2vec_ctc_config.vocab_info,
            masker=masker,
            final_dropout_p=self._config.wav2vec_ctc_config.final_dropout_p,
            max_generation_length=self._config.llama_config.max_seq_len,
            device=self._device,
            dtype=self._dtype,
        )


def create_wav2vec2_llama_model(
    config: Wav2Vec2LlamaConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> Wav2Vec2LlamaModel:
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

    llama_builder = LLaMABuilder(config.llama_config, device=device, dtype=dtype)

    builder = Wav2Vec2LlamaBuilder(
        config, encoder_builder, llama_builder, device=device, dtype=dtype
    )

    return builder.build_model().set_family(WAV2VEC2_LLAMA_FAMILY)


model_factories.register(
    WAV2VEC2_LLAMA_FAMILY,
    create_wav2vec2_llama_model,
    Wav2Vec2LlamaConfig,
    wav2vec2_llama_archs,
)
