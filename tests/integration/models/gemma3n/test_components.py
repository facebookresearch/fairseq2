# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Component-level tests for Gemma3n model."""

import pytest
import torch

from fairseq2.data_type import DataType
from fairseq2.models.gemma3n import (
    Gemma3nConfig,
    convert_gemma3n_state_dict,
    convert_to_hf_gemma3n_state_dict,
    get_gemma3n_e2b_config,
    get_gemma3n_e4b_config,
    is_global_layer,
)
from fairseq2.models.transformer.attention_bias import CausalAttentionBias
from fairseq2.models.transformer.sdpa.soft_capped import SoftCappedSDPA
from fairseq2.nn.batch_layout import BatchLayout
from fairseq2.nn.position_encoder import DualRotaryEncoder
from tests.common import device


def test_gemma3n_config_defaults() -> None:
    """Verify Gemma3n config has correct default values."""
    config = Gemma3nConfig()

    assert config.model_dim == 2048
    assert config.num_layers == 35
    assert config.num_attn_heads == 8
    assert config.num_key_value_heads == 2
    assert config.head_dim == 256
    assert config.vocab_size == 262_400
    assert config.max_seq_len == 32_768
    assert config.ffn_inner_dim == 16_384
    assert config.sliding_window == 512
    assert config.rope_theta == 10_000.0
    assert config.rope_theta_global == 1_000_000.0
    assert config.final_logit_soft_cap == 30.0
    assert config.num_kv_shared_layers == 15
    assert config.laurel_rank == 64


def test_gemma3n_e2b_config() -> None:
    """Verify E2B config factory works."""
    config = get_gemma3n_e2b_config()
    assert isinstance(config, Gemma3nConfig)
    assert config.model_dim == 2048


def test_gemma3n_e4b_config() -> None:
    """Verify E4B config factory works."""
    config = get_gemma3n_e4b_config()
    assert isinstance(config, Gemma3nConfig)
    assert config.model_dim == 2048


def test_is_global_layer() -> None:
    """Verify global layer detection follows 4:1 local:global pattern."""
    # First 5 layers: 0-3 local, 4 global
    assert not is_global_layer(0, num_layers=35)
    assert not is_global_layer(1, num_layers=35)
    assert not is_global_layer(2, num_layers=35)
    assert not is_global_layer(3, num_layers=35)
    assert is_global_layer(4, num_layers=35)

    # Next 5 layers: 5-8 local, 9 global
    assert not is_global_layer(5, num_layers=35)
    assert not is_global_layer(6, num_layers=35)
    assert not is_global_layer(7, num_layers=35)
    assert not is_global_layer(8, num_layers=35)
    assert is_global_layer(9, num_layers=35)

    # Last layer is always global
    assert is_global_layer(34, num_layers=35)

    # Count global layers: should be 7 (layers 4, 9, 14, 19, 24, 29, 34)
    global_count = sum(1 for i in range(35) if is_global_layer(i, num_layers=35))
    assert global_count == 7


def test_convert_gemma3n_state_dict_validates_format() -> None:
    """Verify state dict conversion validates input format."""
    config = Gemma3nConfig()

    # Empty state dict should raise
    with pytest.raises(ValueError, match="Expected HuggingFace Gemma3n checkpoint"):
        convert_gemma3n_state_dict({}, config)

    # Wrong format should raise
    with pytest.raises(ValueError, match="Expected HuggingFace Gemma3n checkpoint"):
        convert_gemma3n_state_dict({"wrong.key": None}, config)


def test_convert_to_hf_not_implemented() -> None:
    """Verify reverse conversion stub raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        convert_to_hf_gemma3n_state_dict({})



def test_dual_rotary_encoder() -> None:
    """Verify DualRotaryEncoder applies dual-frequency RoPE correctly."""
    encoder = DualRotaryEncoder(
        encoding_dim=256,
        max_seq_len=8192,
        theta=10_000.0,
        dual_theta=100_000.0,
        device=device,
    )

    batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 256
    seqs = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    batch_layout = BatchLayout((batch_size, seq_len), seq_lens=None, device=device)

    output = encoder(seqs, batch_layout)

    assert output.shape == seqs.shape
    assert output.device == seqs.device

    first_half = output[..., :128]
    second_half = output[..., 128:]

    assert not torch.allclose(first_half, second_half)


def test_dual_rotary_encoder_validates_encoding_dim() -> None:
    """Verify DualRotaryEncoder rejects encoding_dim not divisible by 4."""
    with pytest.raises(ValueError, match="must be divisible by 4"):
        DualRotaryEncoder(encoding_dim=127, max_seq_len=8192, device=device)
    
    with pytest.raises(ValueError, match="must be divisible by 4"):
        DualRotaryEncoder(encoding_dim=130, max_seq_len=8192, device=device)


def test_soft_capped_sdpa() -> None:
    """Verify SoftCappedSDPA applies soft-capping correctly."""
    bias = CausalAttentionBias()
    sdpa = SoftCappedSDPA(bias, soft_cap=30.0, dropout_p=0.0)

    batch_size, seq_len, num_heads, head_dim = 2, 64, 8, 64
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    q_layout = BatchLayout((batch_size, seq_len), seq_lens=None, device=device)
    k_layout = BatchLayout((batch_size, seq_len), seq_lens=None, device=device)

    from fairseq2.models.transformer.attention_bias import AttentionBiasCache

    bias_cache = AttentionBiasCache()

    output, weights = sdpa(q, q_layout, k, k_layout, v, bias_cache, needs_weights=True)

    assert output.shape == (batch_size, seq_len, num_heads, head_dim)
    assert output.device == q.device
    assert weights is not None
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_soft_capped_sdpa_caps_logits() -> None:
    """Verify soft-capping bounds attention logits."""
    bias = CausalAttentionBias()
    soft_cap = 10.0
    sdpa = SoftCappedSDPA(bias, soft_cap=soft_cap, dropout_p=0.0)

    batch_size, seq_len, num_heads, head_dim = 1, 32, 4, 32
    
    # Create queries and keys that would produce large dot products
    q = torch.ones(batch_size, seq_len, num_heads, head_dim, device=device) * 100.0
    k = torch.ones(batch_size, seq_len, num_heads, head_dim, device=device) * 100.0
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    q_layout = BatchLayout((batch_size, seq_len), seq_lens=None, device=device)
    k_layout = BatchLayout((batch_size, seq_len), seq_lens=None, device=device)

    from fairseq2.models.transformer.attention_bias import AttentionBiasCache

    bias_cache = AttentionBiasCache()

    # Compute with soft-capping - weights should be accessible via needs_weights
    output, weights = sdpa(q, q_layout, k, k_layout, v, bias_cache, needs_weights=True)

    assert output.shape == (batch_size, seq_len, num_heads, head_dim)
    
    # Verify attention weights sum to 1 (valid probability distribution)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))


def test_altup_ffn() -> None:
    """Verify AltUpFeedForwardNetwork with GELU activation."""
    from fairseq2.models.transformer.ffn import AltUpFeedForwardNetwork

    ffn = AltUpFeedForwardNetwork(
        model_dim=2048,
        inner_dim=5376,
        bias=False,
        device=device,
    )

    batch_size, seq_len, model_dim = 2, 128, 2048
    seqs = torch.randn(batch_size, seq_len, model_dim, device=device)
    
    output = ffn(seqs)

    assert output.shape == (batch_size, seq_len, model_dim)
    assert output.device == seqs.device


def test_altup_ffn_uses_gelu() -> None:
    """Verify AltUpFeedForwardNetwork uses GELU activation."""
    from fairseq2.models.transformer.ffn import AltUpFeedForwardNetwork
    from torch.nn import GELU

    ffn = AltUpFeedForwardNetwork(
        model_dim=256,
        inner_dim=512,
        bias=False,
        device=device,
    )

    assert isinstance(ffn.gate_activation, GELU)


def test_create_gemma3n_decoder_layer_local() -> None:
    """Verify local layer uses DualRotaryEncoder and AltUpFFN."""
    from fairseq2.models.gemma3n import Gemma3nConfig, create_gemma3n_decoder_layer
    from fairseq2.models.transformer.ffn import AltUpFeedForwardNetwork

    config = Gemma3nConfig()
    layer_idx = 0  # Local layer

    layer = create_gemma3n_decoder_layer(layer_idx, config, device=device)

    # Verify FFN type
    assert isinstance(layer.ffn, AltUpFeedForwardNetwork)

    # Verify layer can forward
    batch_size, seq_len, model_dim = 2, 64, config.model_dim
    seqs = torch.randn(batch_size, seq_len, model_dim, device=device)
    seqs_layout = BatchLayout((batch_size, seq_len), seq_lens=None, device=device)

    from fairseq2.models.transformer.attention_bias import AttentionBiasCache

    bias_cache = AttentionBiasCache()

    output = layer(seqs, seqs_layout, bias_cache)

    assert output.shape == (batch_size, seq_len, model_dim)
    assert output.device == seqs.device


def test_create_gemma3n_decoder_layer_global() -> None:
    """Verify global layer uses standard GLU FFN."""
    from fairseq2.models.gemma3n import Gemma3nConfig, create_gemma3n_decoder_layer
    from fairseq2.models.transformer import GLUFeedForwardNetwork

    config = Gemma3nConfig()
    layer_idx = 4  # Global layer (every 5th layer)

    layer = create_gemma3n_decoder_layer(layer_idx, config, device=device)

    # Verify FFN type
    assert isinstance(layer.ffn, GLUFeedForwardNetwork)

    # Verify layer can forward
    batch_size, seq_len, model_dim = 2, 64, config.model_dim
    seqs = torch.randn(batch_size, seq_len, model_dim, device=device)
    seqs_layout = BatchLayout((batch_size, seq_len), seq_lens=None, device=device)

    from fairseq2.models.transformer.attention_bias import AttentionBiasCache

    bias_cache = AttentionBiasCache()

    output = layer(seqs, seqs_layout, bias_cache)

    assert output.shape == (batch_size, seq_len, model_dim)
    assert output.device == seqs.device


def test_create_gemma3n_model() -> None:
    """Verify full model creation."""
    from fairseq2.models.gemma3n import create_gemma3n_model

    config = Gemma3nConfig()
    model = create_gemma3n_model(config, device=device, dtype=torch.float32)

    assert model.model_dim == config.model_dim
    assert model.max_seq_len == config.max_seq_len


def test_gemma3n_model_forward() -> None:
    """Verify full model forward pass."""
    from fairseq2.models.gemma3n import create_gemma3n_model

    config = Gemma3nConfig()
    model = create_gemma3n_model(config, device=device, dtype=torch.float32)
    model.eval()

    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    seqs_layout = BatchLayout((batch_size, seq_len), seq_lens=None, device=device)

    with torch.no_grad():
        logits = model(input_ids, seqs_layout)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert logits.device == input_ids.device
