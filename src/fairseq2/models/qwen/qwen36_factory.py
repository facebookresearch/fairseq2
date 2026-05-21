# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Factories for Qwen 3.6 VLM models.

Delegates text backbone creation to existing Qwen35Factory/Qwen35MoeFactory.
Adds vision encoder, merger, M-RoPE, and multimodal frontend on top.
"""

from __future__ import annotations

from fairseq2.models.qwen.config import (
    Qwen36Config,
    Qwen36MoeConfig,
    Qwen36VisionConfig,
)
from fairseq2.models.qwen.factory import Qwen35Factory, Qwen35MoeFactory
from fairseq2.models.qwen.frontend import Qwen36Frontend
from fairseq2.models.qwen.mrope import MultimodalRotaryEncoder
from fairseq2.models.qwen.qwen36_model import Qwen36Model
from fairseq2.models.qwen.vision_encoder import (
    QwenVisionEncoder,
    QwenVisionMerger,
)
from fairseq2.nn import Embedding, Projection


def create_qwen36_model(config: Qwen36Config) -> Qwen36Model:
    """Create a Qwen 3.6 dense VLM."""
    return Qwen36Factory(config).create_model()


def create_qwen36_moe_model(config: Qwen36MoeConfig) -> Qwen36Model:
    """Create a Qwen 3.6 MoE VLM."""
    return Qwen36MoeFactory(config).create_model()


class Qwen36Factory:
    """Factory for Qwen 3.6 dense VLM models."""

    def __init__(self, config: Qwen36Config) -> None:
        self._config = config
        # Delegate text backbone to Qwen35Factory
        self._text_factory = Qwen35Factory(config.text_config)

    def create_model(self) -> Qwen36Model:
        config = self._config
        text_config = config.text_config

        # Create text components via Qwen35Factory
        embed = self._text_factory.create_embedding()
        decoder = self._text_factory.create_decoder()
        final_proj = self._text_factory.create_final_projection(embed)

        # Create vision components
        vision_encoder = self.create_vision_encoder()
        vision_merger = self.create_vision_merger()

        # Create multimodal frontend
        frontend = Qwen36Frontend(
            text_config.model_dim,
            embed,
            vision_encoder,
            vision_merger,
            image_token_id=config.image_token_id,
            dropout_p=text_config.dropout_p,
        )

        # Get the M-RoPE encoder (replace the standard RoPE in the decoder)
        pos_encoder = self.create_mrope_encoder()

        # Replace position encoder in all full-attention layers
        self._replace_position_encoders(decoder, pos_encoder)

        return Qwen36Model(
            text_config.model_dim,
            frontend,
            decoder,
            final_proj,
            pos_encoder,
            pad_idx=text_config.pad_idx,
            max_seq_len=text_config.max_seq_len,
            image_token_id=config.image_token_id,
            mrope_section=config.mrope_section,
        )

    def create_vision_encoder(self) -> QwenVisionEncoder:
        return QwenVisionEncoder(self._config.vision_config)

    def create_vision_merger(self) -> QwenVisionMerger:
        return QwenVisionMerger(self._config.vision_config)

    def create_mrope_encoder(self) -> MultimodalRotaryEncoder:
        text_config = self._config.text_config
        encoding_dim = int(text_config.head_dim * text_config.partial_rotary_factor)

        return MultimodalRotaryEncoder(
            encoding_dim,
            text_config.max_seq_len,
            theta=text_config.rope_theta,
            mrope_section=self._config.mrope_section,
        )

    def _replace_position_encoders(
        self, decoder: object, pos_encoder: MultimodalRotaryEncoder
    ) -> None:
        """Replace RoPE encoders in full-attention layers with the M-RoPE encoder.

        The Qwen35Factory creates a shared ReferenceRotaryEncoder for all
        full-attention layers. We replace those references with our shared
        MultimodalRotaryEncoder so M-RoPE position IDs propagate.
        """
        from fairseq2.models.qwen.attention import Qwen35Attention

        for layer in decoder.layers:
            if hasattr(layer, "self_attn") and layer.self_attn is not None:
                attn = layer.self_attn
                if isinstance(attn, Qwen35Attention) and attn.pos_encoder is not None:
                    attn.pos_encoder = pos_encoder


class Qwen36MoeFactory(Qwen36Factory):
    """Factory for Qwen 3.6 MoE VLM models."""

    def __init__(self, config: Qwen36MoeConfig) -> None:
        # Don't call super().__init__ — we need a different text factory
        self._moe_config = config
        self._config = Qwen36Config(
            text_config=config.text_config,
            vision_config=config.vision_config,
            image_token_id=config.image_token_id,
            video_token_id=config.video_token_id,
            vision_start_token_id=config.vision_start_token_id,
            vision_end_token_id=config.vision_end_token_id,
            mrope_section=config.mrope_section,
        )
        # Use MoE text factory
        self._text_factory = Qwen35MoeFactory(config.text_config)
