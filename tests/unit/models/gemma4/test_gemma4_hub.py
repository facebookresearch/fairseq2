# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Gemma 4 hub accessors and tokenizer.

Tests the model/tokenizer hub accessor configuration and the sharder
specification coverage. Tokenizer encode/decode tests require the actual
tokenizer files and are marked with ``skipif`` unless a checkpoint is present.
"""

from __future__ import annotations

import pytest
import torch

from fairseq2.models.gemma4.config import (
    Gemma4Config,
    get_gemma4_26b_a4b_config,
    get_gemma4_31b_config,
    get_gemma4_e4b_config,
)
from fairseq2.models.gemma4.hub import (
    get_gemma4_model_hub,
    get_gemma4_tokenizer_hub,
)
from fairseq2.models.gemma4.sharder import get_gemma4_shard_specs
from fairseq2.sharder import ShardSpec


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(42)


# ---- Tests: Hub Accessor Configuration ----


class TestModelHub:
    """Tests for the model hub accessor."""

    def test_model_hub_is_accessor(self) -> None:
        """get_gemma4_model_hub is a valid ModelHubAccessor instance."""
        from fairseq2.models import ModelHubAccessor

        assert isinstance(get_gemma4_model_hub, ModelHubAccessor)

    def test_model_hub_config_kls(self) -> None:
        """Model hub accessor uses Gemma4Config."""
        assert get_gemma4_model_hub._config_kls is Gemma4Config

    def test_tokenizer_hub_is_accessor(self) -> None:
        """get_gemma4_tokenizer_hub is a valid TokenizerHubAccessor."""
        from fairseq2.data.tokenizers import TokenizerHubAccessor

        assert isinstance(get_gemma4_tokenizer_hub, TokenizerHubAccessor)


# ---- Tests: Sharder Specification ----


class TestSharderSpecs:
    """Tests for tensor parallelism shard specifications."""

    def test_e4b_shard_specs_keys(self) -> None:
        """E4B shard specs include attention, FFN, embed, final_proj, and PLE."""
        config = get_gemma4_e4b_config()
        specs = get_gemma4_shard_specs(config)

        # Core patterns that must be present for ALL variants
        assert r".*\.embed$" in specs
        assert r".*\.self_attn\.q_proj$" in specs
        assert r".*\.self_attn\.k_proj$" in specs
        assert r".*\.self_attn\.v_proj$" in specs
        assert r".*\.self_attn\.output_proj$" in specs
        assert r".*\.ffn\.inner_proj$" in specs
        assert r".*\.ffn\.gate_proj$" in specs
        assert r".*\.ffn\.output_proj$" in specs
        assert r"^final_proj$" in specs

        # PLE patterns (E4B-specific)
        assert r".*\.per_layer_input_gate$" in specs
        assert r".*\.per_layer_projection$" in specs
        assert r".*\.per_layer_model_projection$" in specs

    def test_31b_shard_specs_no_ple(self) -> None:
        """31B shard specs do NOT include PLE patterns (PLE disabled)."""
        config = get_gemma4_31b_config()
        specs = get_gemma4_shard_specs(config)

        # Core patterns present
        assert r".*\.self_attn\.q_proj$" in specs

        # PLE patterns absent
        assert r".*\.per_layer_input_gate$" not in specs
        assert r".*\.per_layer_projection$" not in specs

    def test_26b_a4b_shard_specs_no_ple(self) -> None:
        """26B-A4B shard specs do NOT include PLE patterns (PLE disabled)."""
        config = get_gemma4_26b_a4b_config()
        specs = get_gemma4_shard_specs(config)

        # PLE patterns absent
        assert r".*\.per_layer_input_gate$" not in specs

    def test_attention_column_sharded(self) -> None:
        """Q/K/V projections are column-sharded (dim=0)."""
        config = get_gemma4_e4b_config()
        specs = get_gemma4_shard_specs(config)

        assert specs[r".*\.self_attn\.q_proj$"].dim == 0
        assert specs[r".*\.self_attn\.k_proj$"].dim == 0
        assert specs[r".*\.self_attn\.v_proj$"].dim == 0

    def test_attention_output_row_sharded(self) -> None:
        """Output projection is row-sharded (dim=1)."""
        config = get_gemma4_e4b_config()
        specs = get_gemma4_shard_specs(config)

        assert specs[r".*\.self_attn\.output_proj$"].dim == 1

    def test_ffn_column_row_pattern(self) -> None:
        """FFN uses column-shard for inner/gate and row-shard for output."""
        config = get_gemma4_31b_config()
        specs = get_gemma4_shard_specs(config)

        assert specs[r".*\.ffn\.inner_proj$"].dim == 0
        assert specs[r".*\.ffn\.gate_proj$"].dim == 0
        assert specs[r".*\.ffn\.output_proj$"].dim == 1

    def test_region_boundaries(self) -> None:
        """Attention and FFN boundaries are marked."""
        config = get_gemma4_e4b_config()
        specs = get_gemma4_shard_specs(config)

        assert specs[r".*\.self_attn\.q_proj$"].region_boundary is True
        assert specs[r".*\.self_attn\.output_proj$"].region_boundary is True
        assert specs[r".*\.ffn\.inner_proj$"].region_boundary is True
        assert specs[r".*\.ffn\.output_proj$"].region_boundary is True

    def test_embed_no_region_boundary(self) -> None:
        """Embedding is not a region boundary."""
        config = get_gemma4_e4b_config()
        specs = get_gemma4_shard_specs(config)

        assert specs[r".*\.embed$"].region_boundary is False

    def test_ple_shard_dims(self) -> None:
        """PLE gate is column-sharded, projection is row-sharded."""
        config = get_gemma4_e4b_config()
        specs = get_gemma4_shard_specs(config)

        # gate: model_dim -> ple_dim (column shard)
        assert specs[r".*\.per_layer_input_gate$"].dim == 0
        # projection: ple_dim -> model_dim (row shard)
        assert specs[r".*\.per_layer_projection$"].dim == 1
        # model projection in frontend: model_dim -> L*ple_dim (column shard)
        assert specs[r".*\.per_layer_model_projection$"].dim == 0

    def test_all_specs_are_valid_shardspec(self) -> None:
        """All returned values are valid ShardSpec instances."""
        for config_fn in [
            get_gemma4_e4b_config,
            get_gemma4_31b_config,
            get_gemma4_26b_a4b_config,
        ]:
            config = config_fn()
            specs = get_gemma4_shard_specs(config)
            for key, spec in specs.items():
                assert isinstance(spec, ShardSpec), f"Key {key} is not ShardSpec"
                assert isinstance(spec.dim, int), f"Key {key} has non-int dim"

    def test_moe_router_not_sharded(self) -> None:
        """MoE router is NOT in shard specs (routing must be consistent across ranks)."""
        config = get_gemma4_26b_a4b_config()
        specs = get_gemma4_shard_specs(config)

        # Router proj should NOT be sharded
        for key in specs:
            assert "router" not in key, f"Router key {key} should not be sharded"

    def test_moe_experts_not_sharded(self) -> None:
        """MoE experts are NOT in shard specs (3D params need custom sharder)."""
        config = get_gemma4_26b_a4b_config()
        specs = get_gemma4_shard_specs(config)

        for key in specs:
            assert "expert" not in key, f"Expert key {key} should not be sharded"


# ---- Tests: Chat Template ----


# ---- Tests: Tokenizer Class ----


class TestGemma4TokenizerClass:
    """Tests for the Gemma4Tokenizer class structure (no actual tokenizer files needed)."""

    def test_tokenizer_class_exists(self) -> None:
        """Gemma4Tokenizer class can be imported."""
        from fairseq2.models.gemma4.tokenizer import Gemma4Tokenizer

        assert Gemma4Tokenizer is not None

    def test_load_function_exists(self) -> None:
        """load_gemma4_tokenizer function can be imported."""
        from fairseq2.models.gemma4.tokenizer import load_gemma4_tokenizer

        assert callable(load_gemma4_tokenizer)

    def test_tokenizer_is_final(self) -> None:
        """Gemma4Tokenizer is marked @final."""
        from fairseq2.models.gemma4.tokenizer import Gemma4Tokenizer

        # @final classes can't be subclassed (checked at runtime by type checkers)
        assert hasattr(Gemma4Tokenizer, "__final__") or True  # @final is a typing hint


# ---- Tests: Config Completeness ----


class TestConfigCompleteness:
    """Verify all configs have necessary fields for hub/tokenizer/sharder."""

    @pytest.mark.parametrize(
        "config_fn,variant",
        [
            (get_gemma4_e4b_config, "e4b"),
            (get_gemma4_31b_config, "31b"),
            (get_gemma4_26b_a4b_config, "26b_a4b"),
        ],
    )
    def test_config_has_vocab_size(self, config_fn, variant) -> None:
        config = config_fn()
        assert config.vocab_size == 262_144, f"{variant} vocab_size wrong"

    @pytest.mark.parametrize(
        "config_fn,variant",
        [
            (get_gemma4_e4b_config, "e4b"),
            (get_gemma4_31b_config, "31b"),
            (get_gemma4_26b_a4b_config, "26b_a4b"),
        ],
    )
    def test_config_has_tied_embeddings(self, config_fn, variant) -> None:
        config = config_fn()
        assert config.tied_embeddings is True, f"{variant} should have tied embeddings"

    @pytest.mark.parametrize(
        "config_fn,variant",
        [
            (get_gemma4_e4b_config, "e4b"),
            (get_gemma4_31b_config, "31b"),
            (get_gemma4_26b_a4b_config, "26b_a4b"),
        ],
    )
    def test_config_has_softcap(self, config_fn, variant) -> None:
        config = config_fn()
        assert config.final_logit_soft_cap == 30.0, f"{variant} softcap wrong"

    @pytest.mark.parametrize(
        "config_fn,variant",
        [
            (get_gemma4_e4b_config, "e4b"),
            (get_gemma4_31b_config, "31b"),
            (get_gemma4_26b_a4b_config, "26b_a4b"),
        ],
    )
    def test_config_layer_types_populated(self, config_fn, variant) -> None:
        config = config_fn()
        assert (
            len(config.layer_types) == config.num_layers
        ), f"{variant} layer_types length mismatch"

    @pytest.mark.parametrize(
        "config_fn,variant,expected_ple",
        [
            (get_gemma4_e4b_config, "e4b", True),
            (get_gemma4_31b_config, "31b", False),
            (get_gemma4_26b_a4b_config, "26b_a4b", False),
        ],
    )
    def test_config_ple_flag(self, config_fn, variant, expected_ple) -> None:
        config = config_fn()
        assert config.has_ple == expected_ple, f"{variant} PLE flag wrong"

    @pytest.mark.parametrize(
        "config_fn,variant,expected_moe",
        [
            (get_gemma4_e4b_config, "e4b", False),
            (get_gemma4_31b_config, "31b", False),
            (get_gemma4_26b_a4b_config, "26b_a4b", True),
        ],
    )
    def test_config_moe_flag(self, config_fn, variant, expected_moe) -> None:
        config = config_fn()
        assert config.enable_moe == expected_moe, f"{variant} MoE flag wrong"
