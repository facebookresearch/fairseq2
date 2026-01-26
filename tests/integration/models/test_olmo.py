# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for all OLMO model variants."""

import os
from typing import Any

import pytest
import torch

from fairseq2.data.tokenizers import Tokenizer, load_tokenizer
from fairseq2.models.hub import ModelHub
from fairseq2.models.olmo import get_olmo_model_hub
from fairseq2.models.olmo.config import OLMOConfig
from fairseq2.models.transformer_lm import TransformerLM
from fairseq2.nn import BatchLayout
from tests import common

transformers = pytest.importorskip("transformers")

# List of OLMO models to test: (model_name, local_path)
OLMO_MODELS = [
    ("olmo-2-0425-1b", "/datasets/pretrained-llms/OLMo-2-0425-1B"),
    ("olmo-2-1124-7b", "/datasets/pretrained-llms/OLMo-2-1124-7B"),
    # ("olmo-2-1124-13b", "/datasets/pretrained-llms/OLMo-2-1124-13B"),  # OOM
    ("olmo-3-1025-7b", "/datasets/pretrained-llms/Olmo-3-1025-7B"),
    # ("olmo-3-1125-32b", "/datasets/pretrained-llms/Olmo-3-1125-32B"),  # OOM
]


@pytest.fixture(scope="module")
def hub() -> ModelHub[TransformerLM, OLMOConfig]:
    """Get the OLMO model hub."""
    return get_olmo_model_hub()


@pytest.mark.parametrize("model_name,local_path", OLMO_MODELS)
def test_consistency(model_name: str, local_path: str, hub: Any) -> None:
    """Test end-to-end inference consistency between fairseq2 and HuggingFace."""
    if not os.path.exists(local_path):
        pytest.skip(f"Model path {local_path} does not exist (not on fair cluster)")
    # Load HuggingFace model
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.float32,
    )
    hf_model.to(common.device)
    hf_model.eval()

    # Load HuggingFace tokenizer
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(local_path)

    # Load fairseq2 model
    fs2_model: TransformerLM = hub.load_model(
        model_name, device=common.device, dtype=torch.float32
    )
    fs2_model.eval()

    # Load fairseq2 tokenizer
    fs2_tokenizer: Tokenizer = load_tokenizer(model_name)

    test_texts = [
        "The capital of Germany is",
        "Once upon a time",
        "Machine learning is",
    ]

    for text in test_texts:
        # fairseq2 pipeline
        fs2_encoder = fs2_tokenizer.create_encoder(mode="prompt", device=common.device)
        fs2_token_ids = fs2_encoder(text).unsqueeze(0)

        # HuggingFace pipeline
        hf_inputs = hf_tokenizer(text, return_tensors="pt", padding=False)
        hf_token_ids = hf_inputs["input_ids"].to(common.device)

        with torch.no_grad():
            fs2_logits = fs2_model(fs2_token_ids, BatchLayout.of(fs2_token_ids)).float()
            hf_logits = hf_model(input_ids=hf_token_ids).logits.float()

        common.assert_close(fs2_logits, hf_logits, atol=1e-4, rtol=1e-5)

    # Clean up to free memory
    del hf_model
    del fs2_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
