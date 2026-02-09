# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Component-level tests for Gemma3n model."""

import pytest

from fairseq2.models.gemma3n import (
    Gemma3nConfig,
    convert_gemma3n_state_dict,
    convert_to_hf_gemma3n_state_dict,
    get_gemma3n_e2b_config,
    get_gemma3n_e4b_config,
    is_global_layer,
)


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

