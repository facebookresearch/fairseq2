# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.llama import get_llama_model_hub
from fairseq2.models.llama.integ import convert_to_huggingface_config


def test_intermediate_size_is_correct() -> None:
    # "intermediate_size" is an expected parameter in the HF Llama config
    # and is computed dynamically from other parameters in Fairseq2
    # We only check it for one arch
    arch = "llama3_2_1b"

    model_config = get_llama_model_hub().load_config(arch)

    config_json = convert_to_huggingface_config(arch, model_config)

    assert config_json["intermediate_size"] == 8192
