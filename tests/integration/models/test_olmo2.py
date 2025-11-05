# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests comparing OLMO2 model output between fairseq2 and HuggingFace transformers.

Tests verify that forward pass outputs are consistent between implementations.
"""

import pytest
import torch

# Skip if transformers not available
transformers = pytest.importorskip("transformers")

from fairseq2.models.olmo2 import (
    OLMO2Config,
    create_olmo2_model,
    convert_olmo2_state_dict,
)
from fairseq2.nn import BatchLayout


# Path to local model checkpoint
MODEL_PATH = "/storage/home/yunchaoyang1/Olmo-2/Olmo-2-0425-1B"


@pytest.fixture(scope="module")
def hf_model():
    """Load HuggingFace OLMO2 model from local checkpoint."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cuda",
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def hf_tokenizer():
    """Load HuggingFace tokenizer from local checkpoint."""
    return transformers.AutoTokenizer.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="module")
def fs2_model(hf_model):
    """Load fairseq2 OLMO2 model with converted weights."""
    hf_config = hf_model.config

    # Create fairseq2 config from HuggingFace config
    config = OLMO2Config(
        model_dim=hf_config.hidden_size,
        max_seq_len=hf_config.max_position_embeddings,
        vocab_size=hf_config.vocab_size,
        num_layers=hf_config.num_hidden_layers,
        num_attn_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        ffn_inner_dim=hf_config.intermediate_size,
        rope_theta=hf_config.rope_theta,
        rms_norm_eps=hf_config.rms_norm_eps,
        tied_embeddings=hf_config.tie_word_embeddings,
    )

    # Convert state dict
    hf_state_dict = hf_model.state_dict()
    fs2_state_dict = convert_olmo2_state_dict(hf_state_dict.copy(), config)

    # Create and load model
    model = create_olmo2_model(config)
    model.load_state_dict(fs2_state_dict, strict=False)
    model = model.cuda()
    model.eval()

    return model


class TestOLMO2ForwardPass:
    """Test forward pass output consistency between fairseq2 and HuggingFace."""

    def test_simple_forward_pass(self, hf_model, fs2_model, hf_tokenizer):
        """Test simple forward pass with short text."""
        text = "Hello, world!"

        # Tokenize
        hf_inputs = hf_tokenizer(text, return_tensors="pt", padding=False)
        input_ids = hf_inputs["input_ids"].cuda()

        # HuggingFace forward pass
        with torch.no_grad():
            hf_outputs = hf_model(input_ids=input_ids)
            hf_logits = hf_outputs.logits.float()

        # fairseq2 forward pass
        seqs = input_ids
        seqs_layout = BatchLayout.of(seqs)

        with torch.no_grad():
            fs2_logits = fs2_model(seqs, seqs_layout).float()

        # Check shapes match
        assert hf_logits.shape == fs2_logits.shape

        # Check outputs are close
        assert torch.allclose(hf_logits, fs2_logits, atol=1e-4, rtol=1e-4)

    def test_multiple_texts(self, hf_model, fs2_model, hf_tokenizer):
        """Test forward pass with multiple different texts."""
        test_texts = [
            "The capital of France is",
            "Once upon a time",
            "Machine learning is",
        ]

        for text in test_texts:
            # Tokenize
            hf_inputs = hf_tokenizer(text, return_tensors="pt", padding=False)
            input_ids = hf_inputs["input_ids"].cuda()

            # HuggingFace forward pass
            with torch.no_grad():
                hf_outputs = hf_model(input_ids=input_ids)
                hf_logits = hf_outputs.logits.float()

            # fairseq2 forward pass
            seqs = input_ids
            seqs_layout = BatchLayout.of(seqs)

            with torch.no_grad():
                fs2_logits = fs2_model(seqs, seqs_layout).float()

            # Check outputs are close
            max_diff = (hf_logits - fs2_logits).abs().max().item()
            assert max_diff < 1e-4, f"Mismatch for '{text}': max_diff={max_diff}"

    def test_longer_sequence(self, hf_model, fs2_model, hf_tokenizer):
        """Test forward pass with longer sequence."""
        text = (
            "In a world where artificial intelligence has become increasingly "
            "sophisticated, researchers continue to push the boundaries of what "
            "machines can accomplish."
        )

        # Tokenize with truncation
        hf_inputs = hf_tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=128,
        )
        input_ids = hf_inputs["input_ids"].cuda()

        # HuggingFace forward pass
        with torch.no_grad():
            hf_outputs = hf_model(input_ids=input_ids)
            hf_logits = hf_outputs.logits.float()

        # fairseq2 forward pass
        seqs = input_ids
        seqs_layout = BatchLayout.of(seqs)

        with torch.no_grad():
            fs2_logits = fs2_model(seqs, seqs_layout).float()

        # Check outputs are close
        assert torch.allclose(hf_logits, fs2_logits, atol=1e-4, rtol=1e-4)
