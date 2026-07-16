# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Qwen 3.5 model family (dense + MoE)."""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from fairseq2.models.qwen.attention import Qwen35Attention
from fairseq2.models.qwen.config import Qwen35Config, Qwen35MoeConfig
from fairseq2.models.qwen.decoder_layer import Qwen35DecoderLayer
from fairseq2.models.qwen.factory import create_qwen35_model, create_qwen35_moe_model
from fairseq2.models.qwen.gated_delta_net import (
    GatedDeltaNet,
    GatedDeltaNetState,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from fairseq2.models.qwen.interop import (
    _QWEN35_HG_KEY_MAP,
    _QWEN35_RMSNORM_KEYS,
    _Qwen35HuggingFaceConverter,
    _Qwen35MoeHuggingFaceConverter,
    convert_qwen35_moe_state_dict,
    convert_qwen35_state_dict,
)
from fairseq2.models.qwen.moe import Qwen35MoeBlock
from fairseq2.models.transformer import FeedForwardNetwork
from fairseq2.models.transformer.attention_bias import (
    AttentionBiasCache,
    CausalAttentionBias,
    IdentityBias,
)
from fairseq2.models.transformer.sdpa.naive import NaiveSDPA
from fairseq2.models.utils.checkpoint import convert_state_dict, create_reverse_key_map
from fairseq2.nn import BatchLayout, IncrementalStateBag
from tests.common import assert_close as fs2_assert_close
from tests.common import device

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _small_dense_config() -> Qwen35Config:
    config = Qwen35Config()
    config.model_dim = 64
    config.vocab_size = 128
    config.num_layers = 4
    config.num_attn_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.ffn_inner_dim = 128
    config.partial_rotary_factor = 0.25
    config.linear_num_key_heads = 2
    config.linear_num_value_heads = 4
    config.linear_key_head_dim = 8
    config.linear_value_head_dim = 8
    config.layer_types = None
    config.__post_init__()
    return config


def _small_moe_config() -> Qwen35MoeConfig:
    config = Qwen35MoeConfig()
    config.model_dim = 64
    config.vocab_size = 128
    config.num_layers = 4
    config.num_attn_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.ffn_inner_dim = 128
    config.partial_rotary_factor = 0.25
    config.linear_num_key_heads = 2
    config.linear_num_value_heads = 4
    config.linear_key_head_dim = 8
    config.linear_value_head_dim = 8
    config.num_experts = 4
    config.num_experts_per_tok = 2
    config.moe_intermediate_size = 32
    config.shared_expert_intermediate_size = 32
    config.layer_types = None
    config.__post_init__()
    return config


# ---------------------------------------------------------------------------
# GatedDeltaNet
# ---------------------------------------------------------------------------


class TestGatedDeltaNet:
    def test_forward_shape(self) -> None:
        gdn = GatedDeltaNet(
            hidden_size=64,
            num_k_heads=2,
            num_v_heads=4,
            head_k_dim=16,
            head_v_dim=16,
            conv_kernel_size=4,
        ).to(device)
        out = gdn(torch.randn(2, 8, 64, device=device))
        assert out.shape == (2, 8, 64)

    def test_chunked_vs_recurrent(self) -> None:
        B, S, H, K, V = 1, 16, 4, 16, 16
        q = torch.randn(B, S, H, K, device=device)
        k = torch.randn(B, S, H, K, device=device)
        v = torch.randn(B, S, H, V, device=device)
        g = -torch.rand(B, S, H, device=device).abs()
        beta = torch.rand(B, S, H, device=device)

        c_out, c_st = torch_chunk_gated_delta_rule(
            q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        r_out, r_st = torch_recurrent_gated_delta_rule(
            q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        fs2_assert_close(c_out, r_out, atol=1e-4)
        assert c_st is not None and r_st is not None
        fs2_assert_close(c_st, r_st, atol=1e-4)

    def test_state_reorder(self) -> None:
        conv = torch.randn(3, 8, 3, device=device)
        rec = torch.randn(3, 4, 16, 16, device=device)
        state = GatedDeltaNetState(conv, rec)
        state.reorder(torch.tensor([2, 0, 1], device=device))
        fs2_assert_close(state.conv_state[0], conv[2])
        fs2_assert_close(state.recurrent_state[0], rec[2])

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="causal_conv1d incremental decode requires CUDA",
    )
    def test_incremental_decode(self) -> None:
        gdn = (
            GatedDeltaNet(
                hidden_size=64,
                num_k_heads=2,
                num_v_heads=4,
                head_k_dim=16,
                head_v_dim=16,
            )
            .to(device)
            .eval()
        )

        full_seq = torch.randn(1, 9, 64, device=device)
        with torch.no_grad():
            full_out = gdn(full_seq)

        state_bag = IncrementalStateBag(max_num_steps=9)
        with torch.no_grad():
            gdn(full_seq[:, :8, :], state_bag=state_bag)
        state_bag.increment_step_nr(8)
        with torch.no_grad():
            incr_out = gdn(full_seq[:, 8:, :], state_bag=state_bag)
        fs2_assert_close(incr_out, full_out[:, -1:, :], atol=1e-4)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class TestQwen35Attention:
    def test_forward_shape(self) -> None:
        sdpa = NaiveSDPA(IdentityBias())
        attn = Qwen35Attention(model_dim=64, num_heads=4, sdpa=sdpa, head_dim=16).to(
            device
        )
        seqs = torch.randn(2, 8, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()
        with torch.no_grad():
            out = attn(seqs, layout, seqs, layout, seqs, bias_cache)
        assert out.shape == (2, 8, 64)

    def test_gqa(self) -> None:
        sdpa = NaiveSDPA(IdentityBias())
        attn = Qwen35Attention(
            model_dim=64,
            num_heads=4,
            sdpa=sdpa,
            head_dim=16,
            num_key_value_heads=2,
        ).to(device)
        seqs = torch.randn(2, 6, 64, device=device)
        layout = BatchLayout.of(seqs)
        with torch.no_grad():
            out = attn(seqs, layout, seqs, layout, seqs, AttentionBiasCache())
        assert out.shape == (2, 6, 64)

    def test_incremental_kv_cache(self) -> None:
        sdpa = NaiveSDPA(CausalAttentionBias())
        attn = Qwen35Attention(model_dim=64, num_heads=4, sdpa=sdpa, head_dim=16).to(
            device
        )
        attn.eval()

        seqs = torch.randn(1, 6, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()
        with torch.no_grad():
            full_out = attn(seqs, layout, seqs, layout, seqs, bias_cache)

        state_bag = IncrementalStateBag(max_num_steps=32)
        with torch.no_grad():
            for idx in range(6):
                step = seqs[:, idx : idx + 1, :]
                sl = BatchLayout.of(step)
                out = attn(step, sl, step, sl, step, bias_cache, state_bag=state_bag)
                fs2_assert_close(out, full_out[:, idx : idx + 1, :], atol=1e-5)
                state_bag.increment_step_nr()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


class TestQwen35Factory:
    def test_small_model_forward(self) -> None:
        config = _small_dense_config()
        model = create_qwen35_model(config).to(device).eval()
        ids = torch.randint(0, 128, (1, 16), device=device)
        with torch.no_grad():
            logits = model(ids, BatchLayout.of(ids))
        assert logits.shape == (1, 16, 128)

    def test_hybrid_layer_pattern(self) -> None:
        config = _small_dense_config()
        with torch.device("meta"):
            model = create_qwen35_model(config)
        types = [
            l.layer_type
            for l in model.decoder.layers
            if isinstance(l, Qwen35DecoderLayer)
        ]
        assert types == [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class TestQwen35Moe:
    def test_moe_block_shape(self) -> None:
        moe = Qwen35MoeBlock(
            model_dim=32,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
        ).to(device)
        with torch.no_grad():
            out = moe(torch.randn(2, 8, 32, device=device))
        assert out.shape == (2, 8, 32)

    def test_moe_is_ffn(self) -> None:
        moe = Qwen35MoeBlock(
            model_dim=32,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
        )
        assert isinstance(moe, FeedForwardNetwork)


# ---------------------------------------------------------------------------
# Interop (state dict conversion)
# ---------------------------------------------------------------------------


class TestQwen35Interop:
    def test_key_round_trip(self) -> None:
        config = _small_dense_config()
        with torch.device("meta"):
            model = create_qwen35_model(config)
        fs2_keys = set(model.state_dict().keys())

        sd: dict[str, object] = {k: torch.empty(0) for k in fs2_keys}
        rev = create_reverse_key_map(_QWEN35_HG_KEY_MAP)
        hg_sd = convert_state_dict(sd, rev)
        rt_keys = set(convert_state_dict(dict(hg_sd), _QWEN35_HG_KEY_MAP).keys())
        assert fs2_keys == rt_keys

    def test_rmsnorm_plus_one(self) -> None:
        config = _small_dense_config()
        hf_sd: dict[str, object] = {}
        for i in range(config.num_layers):
            hf_sd[f"model.layers.{i}.input_layernorm.weight"] = torch.zeros(
                config.model_dim
            )
            hf_sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.zeros(
                config.model_dim
            )
        hf_sd["model.norm.weight"] = torch.zeros(config.model_dim)
        hf_sd["model.embed_tokens.weight"] = torch.zeros(
            config.vocab_size, config.model_dim
        )
        hf_sd["lm_head.weight"] = torch.zeros(config.vocab_size, config.model_dim)

        converted = convert_qwen35_state_dict(dict(hf_sd), config)
        for key in converted:
            if any(key.endswith(s) for s in _QWEN35_RMSNORM_KEYS):
                weight = converted[key]
                assert isinstance(weight, torch.Tensor)
                assert_close(weight, torch.ones_like(weight))

    def test_tied_embeddings(self) -> None:
        config = _small_dense_config()
        config.tied_embeddings = True
        weight = torch.randn(config.vocab_size, config.model_dim)
        hf_sd: dict[str, object] = {
            "model.embed_tokens.weight": weight,
            "model.norm.weight": torch.zeros(config.model_dim),
        }
        result = convert_qwen35_state_dict(dict(hf_sd), config)
        assert "decoder_frontend.embed.weight" in result
        assert "final_proj.weight" in result
        assert result["final_proj.weight"] is result["decoder_frontend.embed.weight"]

    def test_vl_keys_filtered(self) -> None:
        config = _small_dense_config()
        config.tied_embeddings = True
        sd: dict[str, object] = {
            "model.language_model.embed_tokens.weight": torch.randn(
                config.vocab_size, config.model_dim
            ),
            "model.language_model.norm.weight": torch.zeros(config.model_dim),
            "model.visual.blocks.0.attn.proj.weight": torch.empty(0),
            "mtp.fc.weight": torch.empty(0),
        }
        result = convert_qwen35_state_dict(dict(sd), config)
        for key in result:
            assert not key.startswith(("model.visual.", "mtp."))


# ---------------------------------------------------------------------------
# HuggingFace converter (bidirectional)
# ---------------------------------------------------------------------------


class TestQwen35HuggingFaceConverter:
    def test_dense_round_trip(self) -> None:
        config = _small_dense_config()
        with torch.device("meta"):
            model = create_qwen35_model(config)
        fs2_keys = set(model.state_dict().keys())
        sd: dict[str, object] = {k: torch.empty(0) for k in fs2_keys}

        converter = _Qwen35HuggingFaceConverter()
        hg_sd = converter.to_hg_state_dict(sd, config)
        rt_keys = set(convert_qwen35_state_dict(dict(hg_sd), config).keys())
        assert fs2_keys == rt_keys

    def test_to_hg_config(self) -> None:
        config = _small_dense_config()
        hg_config = _Qwen35HuggingFaceConverter().to_hg_config(config)
        assert hg_config.kls_name == "Qwen3_5TextConfig"
        assert hg_config.arch == "Qwen3_5ForCausalLM"
        assert hg_config.data["hidden_size"] == config.model_dim

    def test_moe_round_trip(self) -> None:
        config = _small_moe_config()
        with torch.device("meta"):
            model = create_qwen35_moe_model(config)
        fs2_keys = set(model.state_dict().keys())
        sd: dict[str, object] = {k: torch.empty(0) for k in fs2_keys}

        converter = _Qwen35MoeHuggingFaceConverter()
        hg_sd = converter.to_hg_state_dict(sd, config)
        rt_keys = set(convert_qwen35_moe_state_dict(dict(hg_sd), config).keys())
        assert fs2_keys == rt_keys

    def test_moe_to_hg_config(self) -> None:
        config = _small_moe_config()
        hg_config = _Qwen35MoeHuggingFaceConverter().to_hg_config(config)
        assert hg_config.kls_name == "Qwen3_5TextConfig"
        assert hg_config.arch == "Qwen3_5MoeForCausalLM"
        assert hg_config.data["num_experts"] == config.num_experts
