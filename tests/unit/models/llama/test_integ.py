# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.llama import get_llama_model_hub
from fairseq2.models.llama.integ import convert_to_huggingface_config


def test_intermediate_size_is_correct() -> None:
    model_hub = get_llama_model_hub()

    model_config = model_hub.load_config("llama3_2_1b")

    hg_config = convert_to_huggingface_config(model_config)

    # `intermediate_size` is a required parameter in the Hugging Face LLaMA
    # configuration that is computed dynamically in fairseq2.
    assert hg_config["intermediate_size"] == 8192
