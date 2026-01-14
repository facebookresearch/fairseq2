# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import pytest
import torch
from fairseq2.data.tokenizers import Tokenizer, load_tokenizer
from fairseq2.models.olmo2 import get_olmo2_model_hub
from fairseq2.models.transformer_lm import TransformerLM
from fairseq2.nn import BatchLayout
from tests import common

transformers = pytest.importorskip("transformers")


MODEL_NAME = "olmo2-0425-1b"


@pytest.fixture(scope="module")
def hf_model() -> Any:
    """Load the HuggingFace OLMO2 model."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "allenai/OLMo-2-0425-1B",
        torch_dtype=torch.float32,
    )
    model.to(common.device)
    model.eval()
    return model


@pytest.fixture(scope="module")
def hf_tokenizer() -> Any:
    """Load the HuggingFace OLMO2 tokenizer."""
    return transformers.AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")


@pytest.fixture(scope="module")
def fs2_model() -> TransformerLM:
    """Load the fairseq2 OLMO2 model."""
    hub = get_olmo2_model_hub()
    model = hub.load_model(
        MODEL_NAME, device=common.device, dtype=torch.float32
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def fs2_tokenizer() -> Tokenizer:
    """Load the fairseq2 OLMO tokenizer."""
    return load_tokenizer(MODEL_NAME)


def test_consistency(
    fs2_model: TransformerLM,
    fs2_tokenizer: Tokenizer,
    hf_model: Any,
    hf_tokenizer: Any,
) -> None:
    """
    Test end-to-end inference consistency between fairseq2 and HuggingFace.
    """
    test_texts = [
        "The capital of Germany is",
        "Once upon a time",
        "Machine learning is",
    ]

    for text in test_texts:
        # fairseq2 pipeline
        fs2_encoder = fs2_tokenizer.create_encoder(
            mode="prompt", device=common.device
        )
        fs2_token_ids = fs2_encoder(text).unsqueeze(0)

        # HuggingFace pipeline
        hf_inputs = hf_tokenizer(text, return_tensors="pt", padding=False)
        hf_token_ids = hf_inputs["input_ids"].to(common.device)

        with torch.no_grad():
            fs2_logits = fs2_model(
                fs2_token_ids, BatchLayout.of(fs2_token_ids)
            ).float()
            hf_logits = hf_model(input_ids=hf_token_ids).logits.float()

        common.assert_close(fs2_logits, hf_logits, atol=1e-4, rtol=1e-5)
