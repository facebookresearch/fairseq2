# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the OLMo HuggingFace converter."""

from __future__ import annotations

import torch

from fairseq2.models.olmo.config import OLMOConfig, YaRNScaleConfig
from fairseq2.models.olmo.factory import create_olmo_model
from fairseq2.models.olmo.interop import (
    _OLMOHuggingFaceConverter,
    convert_olmo_state_dict,
)


class TestOLMOHuggingFaceConverter:
    """Tests for _OLMOHuggingFaceConverter."""

    def test_to_hg_config_olmo2(self) -> None:
        """to_hg_config maps OLMOConfig fields to Olmo2Config attributes."""
        config = OLMOConfig()

        converter = _OLMOHuggingFaceConverter()
        hg_config = converter.to_hg_config(config)

        assert hg_config.kls_name == "Olmo2Config"
        assert hg_config.arch == "Olmo2ForCausalLM"

        data = hg_config.data
        assert data["hidden_size"] == config.model_dim
        assert data["max_position_embeddings"] == config.max_seq_len
        assert data["vocab_size"] == config.vocab_size
        assert data["tie_word_embeddings"] == config.tied_embeddings
        assert data["num_hidden_layers"] == config.num_layers
        assert data["num_attention_heads"] == config.num_attn_heads
        assert data["num_key_value_heads"] == config.num_key_value_heads
        assert data["intermediate_size"] == config.ffn_inner_dim
        assert data["rms_norm_eps"] == config.rms_norm_eps
        assert data["pad_token_id"] == config.pad_idx
        assert data["eos_token_id"] == config.eos_token_id
        assert data["bos_token_id"] == config.bos_token_id

        rope_scaling = data["rope_scaling"]
        assert isinstance(rope_scaling, dict)
        assert rope_scaling["rope_theta"] == config.rope_theta
        assert rope_scaling["rope_type"] == "default"

    def test_to_hg_config_olmo3(self) -> None:
        """to_hg_config maps OLMo3 config to Olmo3Config with sliding window and YaRN."""
        config = OLMOConfig()
        config.vocab_size = 100_278
        config.model_dim = 4096
        config.ffn_inner_dim = 11008
        config.num_layers = 32
        config.num_attn_heads = 32
        config.num_key_value_heads = 32
        config.max_seq_len = 65536
        config.sliding_window = 4096
        config.yarn_scale_config = YaRNScaleConfig(
            scale_factor=8.0,
            original_max_seq_len=8192,
            beta_fast=32.0,
            beta_slow=1.0,
            mscale=1.0,
            mscale_all_dim=0.0,
        )

        converter = _OLMOHuggingFaceConverter()
        hg_config = converter.to_hg_config(config)

        assert hg_config.kls_name == "Olmo3Config"
        assert hg_config.arch == "Olmo3ForCausalLM"

        data = hg_config.data
        assert data["hidden_size"] == 4096
        assert data["max_position_embeddings"] == 65536
        assert data["sliding_window"] == 4096

        rope_scaling = data["rope_scaling"]
        assert isinstance(rope_scaling, dict)
        assert rope_scaling["rope_type"] == "yarn"
        assert rope_scaling["rope_theta"] == config.rope_theta
        assert rope_scaling["factor"] == 8.0
        assert rope_scaling["original_max_position_embeddings"] == 8192
        assert rope_scaling["beta_fast"] == 32.0
        assert rope_scaling["beta_slow"] == 1.0

    def test_to_hg_config_olmo3_sliding_only(self) -> None:
        """to_hg_config uses Olmo3Config when sliding_window is set without YaRN."""
        config = OLMOConfig()
        config.sliding_window = 4096

        converter = _OLMOHuggingFaceConverter()
        hg_config = converter.to_hg_config(config)

        assert hg_config.kls_name == "Olmo3Config"
        assert hg_config.arch == "Olmo3ForCausalLM"

        rope_scaling = hg_config.data["rope_scaling"]
        assert isinstance(rope_scaling, dict)
        assert rope_scaling["rope_type"] == "default"

    def test_state_dict_round_trip(self) -> None:
        """State dict keys survive a fs2 -> HF -> fs2 round trip."""
        config = OLMOConfig()

        with torch.device("meta"):
            model = create_olmo_model(config)

        fs2_keys = set(model.state_dict().keys())
        assert len(fs2_keys) > 0

        fs2_state_dict: dict[str, object] = {k: torch.empty(0) for k in fs2_keys}

        converter = _OLMOHuggingFaceConverter()
        hg_state_dict = converter.to_hg_state_dict(fs2_state_dict, config)
        hg_keys = set(hg_state_dict.keys())

        # Verify HF keys have expected prefixes
        for key in hg_keys:
            assert key.startswith(
                ("model.", "lm_head.")
            ), f"Unexpected HF key prefix: {key}"

        # Verify OLMo-specific HF keys exist (post-norm, Q/K norm)
        hg_keys_str = " ".join(hg_keys)
        assert "post_attention_layernorm" in hg_keys_str
        assert "post_feedforward_layernorm" in hg_keys_str
        assert "q_norm" in hg_keys_str
        assert "k_norm" in hg_keys_str

        # Round-trip: HF -> fs2
        rt_state_dict = convert_olmo_state_dict(dict(hg_state_dict), config)
        rt_keys = set(rt_state_dict.keys())

        assert fs2_keys == rt_keys, (
            f"Round-trip key mismatch.\n"
            f"  Missing in round-trip: {fs2_keys - rt_keys}\n"
            f"  Extra in round-trip:   {rt_keys - fs2_keys}"
        )

    def test_state_dict_tied_embeddings(self) -> None:
        """to_hg_state_dict removes lm_head.weight when tied_embeddings=True."""
        config = OLMOConfig()
        config.tied_embeddings = True

        with torch.device("meta"):
            model = create_olmo_model(config)

        fs2_state_dict: dict[str, object] = {
            k: torch.empty(0) for k in model.state_dict().keys()
        }

        converter = _OLMOHuggingFaceConverter()
        hg_state_dict = converter.to_hg_state_dict(fs2_state_dict, config)

        assert "lm_head.weight" not in hg_state_dict
        assert "model.embed_tokens.weight" in hg_state_dict
