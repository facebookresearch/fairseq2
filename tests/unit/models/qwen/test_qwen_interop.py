# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

import pytest
from transformers import AutoConfig

from fairseq2.models.hg.converter import save_hugging_face_model
from fairseq2.models.qwen.config import QwenConfig
from fairseq2.models.qwen.interop import _QwenHuggingFaceConverter


@pytest.mark.parametrize(
    "qkv_proj_bias,kls_name", [(True, "Qwen2Config"), (False, "Qwen3Config")]
)
def test_save_hg_config_keeps_rope_theta(
    tmp_path: Path, qkv_proj_bias: bool, kls_name: str
) -> None:
    config = QwenConfig()
    config.qkv_proj_bias = qkv_proj_bias
    config.rope_theta = 1_000_000.0

    hg_config = _QwenHuggingFaceConverter().to_hg_config(config)

    assert hg_config.kls_name == kls_name

    save_hugging_face_model(tmp_path, {}, hg_config)

    native_config = AutoConfig.from_pretrained(tmp_path)

    assert native_config.num_hidden_layers == config.num_layers
    assert native_config.layer_types == ["full_attention"] * config.num_layers
    assert native_config.rope_parameters == {
        "rope_theta": 1_000_000.0,
        "rope_type": "default",
    }
