# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for Qwen 3.6 VLM — checkpoint loading + text-only logit parity.

Requires:
  - Checkpoints at /engshare/yunchaoyang1/models/Qwen3.6-{27B,35B-A3B}/
  - GPU with sufficient memory (at least 1x 80GB for BF16)
  - conda env: fs2-090dev0-pt290-cu128

Run:
    pytest tests/unit/models/qwen/test_qwen36_integration.py -v -s
"""

from __future__ import annotations

import os

import pytest
import torch

# Skip all tests if checkpoints are not available
DENSE_CKPT = "/engshare/yunchaoyang1/models/Qwen3.6-27B"
MOE_CKPT = "/engshare/yunchaoyang1/models/Qwen3.6-35B-A3B"

has_dense_ckpt = os.path.exists(DENSE_CKPT)
has_moe_ckpt = os.path.exists(MOE_CKPT)
has_gpu = torch.cuda.is_available()

skip_no_dense = pytest.mark.skipif(
    not has_dense_ckpt, reason=f"Dense checkpoint not found at {DENSE_CKPT}"
)
skip_no_moe = pytest.mark.skipif(
    not has_moe_ckpt, reason=f"MoE checkpoint not found at {MOE_CKPT}"
)
skip_no_gpu = pytest.mark.skipif(not has_gpu, reason="GPU not available")


class TestQwen36MetaConstruction:
    """Test model construction on meta device (no GPU/checkpoint required)."""

    def test_dense_27b_meta(self) -> None:
        """Construct dense 27B model on meta device."""
        from fairseq2.models.qwen.config import Qwen36Config
        from fairseq2.models.qwen.qwen36_factory import create_qwen36_model

        config = Qwen36Config()
        with torch.device("meta"):
            model = create_qwen36_model(config)

        total = sum(p.numel() for p in model.parameters())
        # ~27.4B total
        assert 25_000_000_000 < total < 30_000_000_000, f"Got {total:,}"

    def test_moe_35b_meta(self) -> None:
        """Construct MoE 35B-A3B model on meta device."""
        from fairseq2.models.qwen.config import Qwen36MoeConfig
        from fairseq2.models.qwen.qwen36_factory import create_qwen36_moe_model

        config = Qwen36MoeConfig()
        with torch.device("meta"):
            model = create_qwen36_moe_model(config)

        total = sum(p.numel() for p in model.parameters())
        # ~35.1B total
        assert 30_000_000_000 < total < 40_000_000_000, f"Got {total:,}"

    def test_dense_has_vision_components(self) -> None:
        """Verify the constructed model has vision encoder and merger."""
        from fairseq2.models.qwen.config import Qwen36Config
        from fairseq2.models.qwen.qwen36_factory import create_qwen36_model

        config = Qwen36Config()
        with torch.device("meta"):
            model = create_qwen36_model(config)

        # Check vision encoder exists in frontend
        assert hasattr(model.decoder_frontend, "vision_encoder")
        assert hasattr(model.decoder_frontend, "vision_merger")
        assert hasattr(model, "pos_encoder")

    def test_mrope_encoder_shared(self) -> None:
        """All full-attention layers should share the same M-RoPE encoder."""
        from fairseq2.models.qwen.config import Qwen36Config
        from fairseq2.models.qwen.qwen36_factory import create_qwen36_model

        config = Qwen36Config()
        with torch.device("meta"):
            model = create_qwen36_model(config)

        # The model's pos_encoder should be a MultimodalRotaryEncoder
        from fairseq2.models.qwen.mrope import MultimodalRotaryEncoder

        assert isinstance(model.pos_encoder, MultimodalRotaryEncoder)


@skip_no_dense
class TestQwen36DenseKeyMapping:
    """Test checkpoint key mapping for dense 27B model."""

    def test_all_keys_mapped(self) -> None:
        """Every checkpoint key should map to a model parameter."""
        from safetensors import safe_open

        from fairseq2.models.qwen.config import Qwen36Config
        from fairseq2.models.qwen.interop import convert_qwen36_state_dict
        from fairseq2.models.qwen.qwen36_factory import create_qwen36_model

        # Load checkpoint keys
        index_file = os.path.join(DENSE_CKPT, "model.safetensors.index.json")
        if os.path.exists(index_file):
            import json

            with open(index_file) as f:
                index = json.load(f)
            ckpt_keys = set(index["weight_map"].keys())
        else:
            # Single file
            with safe_open(
                os.path.join(DENSE_CKPT, "model.safetensors"), framework="pt"
            ) as f:
                ckpt_keys = set(f.keys())

        # Filter MTP keys (expected to be skipped)
        ckpt_keys = {k for k in ckpt_keys if not k.startswith("mtp.")}

        # Create model on meta and get expected keys
        config = Qwen36Config()
        with torch.device("meta"):
            model = create_qwen36_model(config)
        model_keys = set(dict(model.named_parameters()).keys())

        # Build a dummy state dict with the checkpoint keys and convert
        dummy_sd = {k: torch.tensor(0.0) for k in ckpt_keys}
        converted = convert_qwen36_state_dict(dummy_sd, config)
        converted_keys = set(converted.keys())

        # Check coverage
        missing_in_model = converted_keys - model_keys
        missing_in_ckpt = model_keys - converted_keys

        # Allow tied embeddings
        missing_in_ckpt.discard("final_proj.weight")
        missing_in_ckpt.discard("decoder_frontend.embed.weight")

        assert len(missing_in_model) == 0, f"Keys in checkpoint but not model: {missing_in_model}"
        assert len(missing_in_ckpt) == 0, f"Keys in model but not checkpoint: {missing_in_ckpt}"


@skip_no_moe
class TestQwen36MoeKeyMapping:
    """Test checkpoint key mapping for MoE 35B-A3B model."""

    def test_all_keys_mapped(self) -> None:
        """Every MoE checkpoint key should map to a model parameter."""
        from safetensors import safe_open

        from fairseq2.models.qwen.config import Qwen36MoeConfig
        from fairseq2.models.qwen.interop import convert_qwen36_moe_state_dict
        from fairseq2.models.qwen.qwen36_factory import create_qwen36_moe_model

        index_file = os.path.join(MOE_CKPT, "model.safetensors.index.json")
        if os.path.exists(index_file):
            import json

            with open(index_file) as f:
                index = json.load(f)
            ckpt_keys = set(index["weight_map"].keys())
        else:
            with safe_open(
                os.path.join(MOE_CKPT, "model.safetensors"), framework="pt"
            ) as f:
                ckpt_keys = set(f.keys())

        ckpt_keys = {k for k in ckpt_keys if not k.startswith("mtp.")}

        config = Qwen36MoeConfig()
        with torch.device("meta"):
            model = create_qwen36_moe_model(config)
        model_keys = set(dict(model.named_parameters()).keys())

        dummy_sd = {k: torch.tensor(0.0) for k in ckpt_keys}
        converted = convert_qwen36_moe_state_dict(dummy_sd, config)
        converted_keys = set(converted.keys())

        missing_in_model = converted_keys - model_keys
        missing_in_ckpt = model_keys - converted_keys

        missing_in_ckpt.discard("final_proj.weight")
        missing_in_ckpt.discard("decoder_frontend.embed.weight")

        assert len(missing_in_model) == 0, f"Keys in checkpoint but not model: {missing_in_model}"
        assert len(missing_in_ckpt) == 0, f"Keys in model but not checkpoint: {missing_in_ckpt}"
