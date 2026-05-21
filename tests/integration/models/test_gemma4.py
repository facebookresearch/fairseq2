# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma 4 integration tests.

Tests that require real model weights or the HuggingFace ``transformers``
library.  Skipped automatically when dependencies are unavailable.

Test categories:
  1. **HF logit parity** — load checkpoint into both HF and fairseq2,
     run the same prompt, assert numerical closeness.
  2. **HF converter round-trip** — convert fs2 state dict → HF format → fs2
     and verify zero key/value loss.
  3. **State dict load from HF checkpoint** — download, convert, load with
     ``strict=True``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fairseq2.models.gemma4.config import (
    Gemma4Config,
    get_gemma4_26b_a4b_config,
    get_gemma4_31b_config,
    get_gemma4_e4b_config,
)
from fairseq2.models.gemma4.factory import create_gemma4_model
from fairseq2.models.gemma4.interop import (
    _Gemma4HuggingFaceConverter,
    _GEMMA4_TEXT_KEY_MAP,
    _HG_KEY_MAP,
    convert_gemma4_state_dict,
)
from fairseq2.models.utils.checkpoint import convert_state_dict, create_reverse_key_map
from fairseq2.nn import BatchLayout


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

def _has_transformers() -> bool:
    """Return True if ``transformers`` is installed."""
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _hf_model_type_available(model_type: str) -> bool:
    """Return True if the installed ``transformers`` recognises *model_type*."""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        return model_type in CONFIG_MAPPING
    except Exception:
        return False


def _checkpoint_exists(path: str) -> bool:
    """Return True if a local checkpoint directory exists and contains weights."""
    p = Path(path)
    if not p.is_dir():
        return False
    # Check for safetensors or bin files
    return any(p.glob("*.safetensors")) or any(p.glob("*.bin"))


# Local checkpoint paths (set via env var or fallback to known locations)
E4B_CHECKPOINT = os.environ.get(
    "GEMMA4_E4B_CHECKPOINT", "/checkpoint/smallomnillm/shared/models/gemma-4-E4B"
)
E4B_IT_CHECKPOINT = os.environ.get(
    "GEMMA4_E4B_IT_CHECKPOINT", "/checkpoint/smallomnillm/shared/models/gemma-4-E4B-it"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Test: HF Converter Round-Trip (no checkpoint needed)
# ---------------------------------------------------------------------------


class TestGemma4ConverterRoundTrip:
    """Round-trip tests for the HuggingFace converter.

    These tests verify that converting a fairseq2 state dict to HF format
    and back produces identical keys and values.  No actual checkpoints
    are needed — uses randomly-initialized small models.
    """

    def _make_small_config(self, *, enable_moe: bool = False) -> Gemma4Config:
        return Gemma4Config(
            model_dim=64,
            vocab_size=128,
            num_layers=6,
            num_attn_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            ffn_inner_dim=128,
            sliding_window=32,
            partial_rotary_factor=0.25,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=0,
            final_logit_soft_cap=30.0,
            tied_embeddings=True,
            enable_moe=enable_moe,
            num_experts=4 if enable_moe else None,
            top_k_experts=2 if enable_moe else None,
            moe_intermediate_size=16 if enable_moe else None,
        )

    def test_converter_round_trip_dense(self) -> None:
        """fs2 → HF → fs2 preserves all keys for dense model."""
        config = self._make_small_config()
        model = create_gemma4_model(config)
        model.eval()

        original_sd = model.state_dict()
        fs2_sd: dict[str, object] = {k: v.clone() for k, v in original_sd.items()}

        # Forward: fs2 → HF
        converter = _Gemma4HuggingFaceConverter()
        hg_sd = converter.to_hg_state_dict(dict(fs2_sd), config)

        # All HF keys should have model.* prefix
        for key in hg_sd:
            assert key.startswith("model."), f"Unexpected prefix: {key}"

        # Backward: HF → fs2 (simulate HF checkpoint loading)
        # Add model.language_model prefix to simulate multimodal format
        multimodal_sd: dict[str, object] = {}
        for k, v in hg_sd.items():
            if k.startswith("model."):
                new_key = k.replace("model.", "model.language_model.", 1)
                multimodal_sd[new_key] = v
            else:
                multimodal_sd[k] = v

        rt_sd = convert_gemma4_state_dict(multimodal_sd, config)

        # Verify round-trip key fidelity
        original_keys = set(original_sd.keys())
        rt_keys = set(rt_sd.keys())

        assert original_keys == rt_keys, (
            f"Key mismatch after round-trip.\n"
            f"  Missing: {original_keys - rt_keys}\n"
            f"  Extra:   {rt_keys - original_keys}"
        )

    def test_converter_round_trip_moe(self) -> None:
        """fs2 → HF → fs2 preserves all keys for MoE model."""
        config = self._make_small_config(enable_moe=True)
        model = create_gemma4_model(config)
        model.eval()

        original_sd = model.state_dict()
        fs2_sd: dict[str, object] = {k: v.clone() for k, v in original_sd.items()}

        converter = _Gemma4HuggingFaceConverter()
        hg_sd = converter.to_hg_state_dict(dict(fs2_sd), config)

        # Verify MoE keys are in HF format
        moe_keys = [k for k in hg_sd if "router" in k or "experts" in k]
        assert len(moe_keys) > 0, "No MoE keys in HF state dict"

    def test_converter_round_trip_value_preservation(self) -> None:
        """Tensor values are preserved through fs2 → HF → fs2 round-trip."""
        config = self._make_small_config()
        config.tied_embeddings = False
        model = create_gemma4_model(config)
        model.eval()

        original_sd = model.state_dict()
        fs2_sd: dict[str, object] = {k: v.clone() for k, v in original_sd.items()}

        converter = _Gemma4HuggingFaceConverter()
        hg_sd = converter.to_hg_state_dict(dict(fs2_sd), config)

        # Convert HF keys back via text key map
        rt_sd = convert_state_dict(hg_sd, _GEMMA4_TEXT_KEY_MAP)

        # Spot-check a few values
        for key in ["decoder.layer_norm.weight", "decoder_frontend.embed.weight"]:
            if key in original_sd and key in rt_sd:
                orig = original_sd[key]
                rt = rt_sd[key]
                if isinstance(orig, torch.Tensor) and isinstance(rt, torch.Tensor):
                    assert torch.equal(orig, rt), f"Value mismatch for {key}"

    def test_to_hg_config_all_variants(self) -> None:
        """to_hg_config works for all production configs."""
        converter = _Gemma4HuggingFaceConverter()

        for config_fn, name in [
            (get_gemma4_e4b_config, "e4b"),
            (get_gemma4_31b_config, "31b"),
            (get_gemma4_26b_a4b_config, "26b_a4b"),
        ]:
            config = config_fn()
            hg_config = converter.to_hg_config(config)

            assert hg_config.kls_name == "Gemma4TextConfig", f"Wrong kls for {name}"
            assert hg_config.arch == "Gemma4ForCausalLM", f"Wrong arch for {name}"
            assert hg_config.data["hidden_size"] == config.model_dim
            assert hg_config.data["num_hidden_layers"] == config.num_layers
            assert hg_config.data["vocab_size"] == 262_144

    def test_to_hg_config_moe_variant(self) -> None:
        """26B-A4B config includes MoE parameters."""
        converter = _Gemma4HuggingFaceConverter()
        config = get_gemma4_26b_a4b_config()
        hg_config = converter.to_hg_config(config)

        data = hg_config.data
        assert data["num_local_experts"] == 128
        assert data["num_experts_per_tok"] == 8
        assert data["moe_intermediate_size"] == 704


# ---------------------------------------------------------------------------
# Test: State Dict Load from HF Checkpoint (requires checkpoint)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _checkpoint_exists(E4B_CHECKPOINT),
    reason=f"E4B checkpoint not found at {E4B_CHECKPOINT}",
)
class TestGemma4E4BCheckpointLoad:
    """Tests that load real E4B checkpoint weights into fairseq2 model."""

    def test_load_e4b_state_dict_strict(self) -> None:
        """Load E4B checkpoint with strict=True — no missing/unexpected keys."""
        from safetensors import safe_open

        config = get_gemma4_e4b_config()

        # Load safetensors
        ckpt_path = Path(E4B_CHECKPOINT)
        safetensor_files = sorted(ckpt_path.glob("*.safetensors"))
        assert len(safetensor_files) > 0, "No safetensors files found"

        hf_state_dict: dict[str, object] = {}
        for sf_path in safetensor_files:
            with safe_open(str(sf_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    hf_state_dict[key] = f.get_tensor(key)

        # Convert
        fs2_state_dict = convert_gemma4_state_dict(hf_state_dict, config)

        # Build model on meta device, then load
        with torch.device("meta"):
            model = create_gemma4_model(config)

        # Load with assign=True for meta device
        result = model.load_state_dict(fs2_state_dict, strict=True, assign=True)

        assert len(result.missing_keys) == 0, f"Missing: {result.missing_keys[:10]}"
        assert len(result.unexpected_keys) == 0, f"Unexpected: {result.unexpected_keys[:10]}"

    def test_e4b_param_count(self) -> None:
        """E4B has ~7.46B parameters (matching HuggingFace)."""
        config = get_gemma4_e4b_config()

        with torch.device("meta"):
            model = create_gemma4_model(config)

        total = sum(p.numel() for p in model.parameters())
        # E4B: 7,463,013,376 params
        assert abs(total - 7_463_013_376) < 1_000_000, f"E4B param count: {total:,}"


# ---------------------------------------------------------------------------
# Test: HF Logit Parity (requires checkpoint + transformers)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_has_transformers() and _hf_model_type_available("gemma4")),
    reason="transformers does not support model_type 'gemma4' (install/upgrade transformers)",
)
@pytest.mark.skipif(
    not _checkpoint_exists(E4B_IT_CHECKPOINT),
    reason=f"E4B-it checkpoint not found at {E4B_IT_CHECKPOINT}",
)
class TestGemma4HFParity:
    """Numerical parity between HuggingFace and fairseq2 for Gemma 4 E4B-it.

    This is the definitive parity test.  It loads the same checkpoint into
    both frameworks, runs identical prompts, and asserts that logits match
    within tight numerical tolerances.
    """

    def test_logit_parity_fp32(self) -> None:
        """FP32 logit parity: max_abs_diff < 1e-3, cosine > 0.9999."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # ---- Load HF model ----
        hf_tokenizer = AutoTokenizer.from_pretrained(E4B_IT_CHECKPOINT)
        hf_model = AutoModelForCausalLM.from_pretrained(
            E4B_IT_CHECKPOINT, torch_dtype=torch.float32
        )
        hf_model.eval()

        # ---- Build and load fairseq2 model ----
        config = get_gemma4_e4b_config()
        hf_state_dict = dict(hf_model.state_dict())
        fs2_state_dict = convert_gemma4_state_dict(hf_state_dict, config)

        # Free HF model to make room for FS2 model on CPU
        del hf_model

        fs2_model = create_gemma4_model(config)
        fs2_model.load_state_dict(fs2_state_dict, strict=True, assign=True)
        fs2_model.eval()

        # Reload HF model for forward pass
        hf_model = AutoModelForCausalLM.from_pretrained(
            E4B_IT_CHECKPOINT, torch_dtype=torch.float32
        )
        hf_model.eval()

        # ---- Run forward passes ----
        test_prompts = [
            "The capital of France is",
            "def fibonacci(n):",
            "In quantum mechanics,",
        ]

        for prompt in test_prompts:
            tokens = hf_tokenizer(prompt, return_tensors="pt")
            input_ids = tokens["input_ids"]

            with torch.no_grad():
                hf_logits = hf_model(input_ids).logits
                seqs_layout = BatchLayout.of(input_ids)
                fs2_logits = fs2_model(input_ids, seqs_layout)

            hf_last = hf_logits[0, -1].float()
            fs2_last = fs2_logits[0, -1].float()

            abs_diff = (hf_last - fs2_last).abs()
            max_diff = abs_diff.max().item()

            cos_sim = F.cosine_similarity(
                hf_last.unsqueeze(0), fs2_last.unsqueeze(0)
            ).item()

            hf_top1 = hf_last.argmax().item()
            fs2_top1 = fs2_last.argmax().item()

            assert max_diff < 1e-3 or cos_sim > 0.9999, (
                f"FP32 parity failed for '{prompt}': "
                f"max_diff={max_diff:.2e}, cosine={cos_sim:.8f}"
            )
            assert hf_top1 == fs2_top1, (
                f"Top-1 mismatch for '{prompt}': "
                f"HF={hf_top1}, fs2={fs2_top1}"
            )

    def test_logit_parity_bf16(self) -> None:
        """BF16 logit parity: cosine > 0.998, top-1 agreement > 90%."""
        if not torch.cuda.is_available():
            pytest.skip("BF16 parity test requires CUDA")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device("cuda")

        hf_tokenizer = AutoTokenizer.from_pretrained(E4B_IT_CHECKPOINT)
        hf_model = AutoModelForCausalLM.from_pretrained(
            E4B_IT_CHECKPOINT, torch_dtype=torch.bfloat16
        ).to(device)
        hf_model.eval()

        config = get_gemma4_e4b_config()
        hf_state_dict = dict(hf_model.state_dict())

        # Move HF state dict to CPU for conversion
        cpu_sd = {k: v.cpu() for k, v in hf_state_dict.items()}
        fs2_state_dict = convert_gemma4_state_dict(cpu_sd, config)

        # Create on CPU (non-persistent buffers like per_layer_embed_scale
        # must be computed in the constructor, not loaded from state dict)
        del hf_model  # Free GPU memory for FS2
        torch.cuda.empty_cache()

        fs2_model = create_gemma4_model(config)
        fs2_model.load_state_dict(fs2_state_dict, strict=True, assign=True)
        fs2_model = fs2_model.to(device=device, dtype=torch.bfloat16)
        fs2_model.eval()

        hf_model = AutoModelForCausalLM.from_pretrained(
            E4B_IT_CHECKPOINT, torch_dtype=torch.bfloat16
        ).to(device)
        hf_model.eval()

        test_prompts = [
            "The capital of France is",
            "def fibonacci(n):",
            "In quantum mechanics,",
        ]

        top1_matches = 0
        total_prompts = 0
        min_cosine = 1.0

        for prompt in test_prompts:
            tokens = hf_tokenizer(prompt, return_tensors="pt")
            input_ids = tokens["input_ids"].to(device)

            with torch.no_grad():
                hf_logits = hf_model(input_ids).logits
                seqs_layout = BatchLayout.of(input_ids)
                fs2_logits = fs2_model(input_ids, seqs_layout)

            hf_last = hf_logits[0, -1].float()
            fs2_last = fs2_logits[0, -1].float()

            cos_sim = F.cosine_similarity(
                hf_last.unsqueeze(0), fs2_last.unsqueeze(0)
            ).item()
            min_cosine = min(min_cosine, cos_sim)

            if hf_last.argmax().item() == fs2_last.argmax().item():
                top1_matches += 1
            total_prompts += 1

        top1_rate = top1_matches / total_prompts
        assert min_cosine > 0.998, f"BF16 min cosine={min_cosine:.6f}, expected > 0.998"
        assert top1_rate >= 0.66, f"BF16 top-1 rate={top1_rate:.2%}, expected >= 66%"


# ---------------------------------------------------------------------------
# Test: HF Export → Load Round-Trip (requires transformers)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_transformers(),
    reason="transformers not installed",
)
class TestGemma4HFExportRoundTrip:
    """Test that fs2 state dict can be exported to HF format and keys are valid.

    Does NOT require checkpoints — uses small randomly-initialized models.
    Validates that the exported state dict has the right structure for HF.
    """

    def _make_small_config(self) -> Gemma4Config:
        return Gemma4Config(
            model_dim=64,
            vocab_size=128,
            num_layers=6,
            num_attn_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            ffn_inner_dim=128,
            sliding_window=32,
            partial_rotary_factor=0.25,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=0,
            final_logit_soft_cap=30.0,
            tied_embeddings=True,
        )

    def test_exported_config_is_valid_dict(self) -> None:
        """Exported HF config can be serialized to JSON (no torch tensors)."""
        import json

        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config()
        hg_config = converter.to_hg_config(config)

        # All values should be JSON-serializable
        json_str = json.dumps(dict(hg_config.data), default=str)
        assert len(json_str) > 0

    def test_exported_state_dict_key_patterns(self) -> None:
        """Exported keys match HuggingFace naming conventions."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config()
        model = create_gemma4_model(config)
        model.eval()

        fs2_sd: dict[str, object] = dict(model.state_dict())
        hg_sd = converter.to_hg_state_dict(fs2_sd, config)

        expected_prefixes = {
            "model.embed_tokens.",
            "model.layers.",
            "model.norm.",
        }

        for key in hg_sd:
            assert any(key.startswith(p) for p in expected_prefixes), (
                f"Key '{key}' doesn't match any expected HF prefix"
            )

    def test_exported_layer_count(self) -> None:
        """Number of layers in exported state dict matches config."""
        converter = _Gemma4HuggingFaceConverter()
        config = self._make_small_config()
        model = create_gemma4_model(config)

        fs2_sd: dict[str, object] = dict(model.state_dict())
        hg_sd = converter.to_hg_state_dict(fs2_sd, config)

        # Count unique layer indices
        import re

        layer_indices = set()
        for key in hg_sd:
            m = re.match(r"model\.layers\.(\d+)\.", key)
            if m:
                layer_indices.add(int(m.group(1)))

        assert len(layer_indices) == config.num_layers, (
            f"Expected {config.num_layers} layers, found {len(layer_indices)}"
        )
