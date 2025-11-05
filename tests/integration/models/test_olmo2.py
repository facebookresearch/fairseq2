# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

transformers = pytest.importorskip("transformers")

from fairseq2.device import Device
from fairseq2.models.olmo2 import get_olmo2_model_hub
from fairseq2.nn import BatchLayout


MODEL_NAME = "olmo2-0425-1B"


@pytest.fixture(scope="module")
def hf_model():
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "allenai/OLMo-2-0425-1B",
        dtype=torch.float32,
        device_map="cuda",
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def hf_tokenizer():
    return transformers.AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")


@pytest.fixture(scope="module")
def fs2_model():
    hub = get_olmo2_model_hub()
    model = hub.load_model(MODEL_NAME, device=Device("cuda"), dtype=torch.float32)
    model.eval()
    return model


def test_consistency(hf_model, fs2_model, hf_tokenizer):
    test_texts = [
        "The capital of Germany is",
        "Once upon a time",
        "Machine learning is",
    ]

    for text in test_texts:
        hf_inputs = hf_tokenizer(text, return_tensors="pt", padding=False)
        input_ids = hf_inputs["input_ids"].cuda()

        with torch.no_grad():
            hf_logits = hf_model(input_ids=input_ids).logits.float()
            fs2_logits = fs2_model(input_ids, BatchLayout.of(input_ids)).float()

        assert torch.allclose(fs2_logits, hf_logits, atol=1e-4, rtol=1e-5), \
            f"Logits mismatch for '{text}'"
