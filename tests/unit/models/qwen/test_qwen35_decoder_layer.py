# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

from fairseq2.models.qwen.attention import Qwen35Attention
from fairseq2.models.qwen.config import Qwen35Config
from fairseq2.models.qwen.decoder_layer import Qwen35DecoderLayer
from fairseq2.models.qwen.factory import create_qwen35_model
from fairseq2.models.qwen.gated_delta_net import GatedDeltaNet
from fairseq2.models.transformer import GLUFeedForwardNetwork
from fairseq2.models.transformer.attention_bias import (
    AttentionBiasCache,
    IdentityBias,
)
from fairseq2.models.transformer.sdpa.naive import NaiveSDPA
from fairseq2.nn import BatchLayout, RMSNorm
from tests.common import device


class TestQwen35DecoderLayer:
    def test_full_attention_layer_forward(self) -> None:
        """Full attention layer produces correct shape."""
        model_dim = 64
        sdpa = NaiveSDPA(IdentityBias())
        self_attn = Qwen35Attention(
            model_dim, num_heads=4, sdpa=sdpa, head_dim=16
        )
        ffn = GLUFeedForwardNetwork(model_dim, 128, bias=False, inner_dim_scale=1.0)
        layer = Qwen35DecoderLayer(
            "full_attention",
            self_attn=self_attn,
            linear_attn=None,
            ffn=ffn,
            self_attn_layer_norm=RMSNorm(model_dim, bias=False),
            ffn_layer_norm=RMSNorm(model_dim, bias=False),
        ).to(device)

        seqs = torch.randn(2, 8, model_dim, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = layer(seqs, layout, bias_cache)

        assert out.shape == (2, 8, model_dim)

    def test_linear_attention_layer_forward(self) -> None:
        """Linear attention (GatedDeltaNet) layer produces correct shape."""
        model_dim = 64
        gdn = GatedDeltaNet(
            model_dim, num_k_heads=2, num_v_heads=4, head_k_dim=8, head_v_dim=8
        )
        ffn = GLUFeedForwardNetwork(model_dim, 128, bias=False, inner_dim_scale=1.0)
        layer = Qwen35DecoderLayer(
            "linear_attention",
            self_attn=None,
            linear_attn=gdn,
            ffn=ffn,
            self_attn_layer_norm=RMSNorm(model_dim, bias=False),
            ffn_layer_norm=RMSNorm(model_dim, bias=False),
        ).to(device)

        seqs = torch.randn(2, 8, model_dim, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = layer(seqs, layout, bias_cache)

        assert out.shape == (2, 8, model_dim)

    def test_invalid_layer_type_raises(self) -> None:
        """Invalid layer_type raises ValueError."""
        model_dim = 64
        ffn = GLUFeedForwardNetwork(model_dim, 128, bias=False, inner_dim_scale=1.0)

        with pytest.raises(ValueError, match="layer_type"):
            Qwen35DecoderLayer(
                "invalid_type",
                self_attn=None,
                linear_attn=None,
                ffn=ffn,
                self_attn_layer_norm=RMSNorm(model_dim, bias=False),
                ffn_layer_norm=RMSNorm(model_dim, bias=False),
            )


class TestQwen35ModelFactory:
    def test_create_small_model(self) -> None:
        """Factory creates a working model with the correct output shape."""
        config = Qwen35Config(
            model_dim=64,
            vocab_size=128,
            num_layers=4,
            num_attn_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            ffn_inner_dim=128,
            partial_rotary_factor=0.25,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
        )

        model = create_qwen35_model(config).to(device)
        model.eval()

        input_ids = torch.randint(0, 128, (1, 16), device=device)
        layout = BatchLayout.of(input_ids)

        with torch.no_grad():
            logits = model(input_ids, layout)

        assert logits.shape == (1, 16, 128)

    def test_model_has_hybrid_layers(self) -> None:
        """Model should have both full_attention and linear_attention layers."""
        config = Qwen35Config(
            model_dim=64,
            vocab_size=128,
            num_layers=4,
            num_attn_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            ffn_inner_dim=128,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
        )

        with torch.device("meta"):
            model = create_qwen35_model(config)

        layers = list(model.decoder.layers)
        layer_types = [
            l.layer_type for l in layers if isinstance(l, Qwen35DecoderLayer)
        ]
        assert layer_types == [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
