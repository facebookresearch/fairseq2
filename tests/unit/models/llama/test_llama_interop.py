# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

from transformers import AutoConfig

from fairseq2.models.hg.converter import save_hugging_face_model
from fairseq2.models.llama.config import LLaMAConfig
from fairseq2.models.llama.interop import _LLaMAHuggingFaceConverter


def test_save_hg_config_keeps_rope_theta(tmp_path: Path) -> None:
    config = LLaMAConfig()
    config.rope_theta = 500_000.0

    hg_config = _LLaMAHuggingFaceConverter().to_hg_config(config)

    save_hugging_face_model(tmp_path, {}, hg_config)

    native_config = AutoConfig.from_pretrained(tmp_path)

    assert native_config.rope_parameters == {
        "rope_theta": 500_000.0,
        "rope_type": "default",
    }


def test_save_hg_config_keeps_rope_theta_with_scaled_rope(tmp_path: Path) -> None:
    config = LLaMAConfig()
    config.max_seq_len = 131_072
    config.rope_theta = 500_000.0
    config.use_scaled_rope = True

    hg_config = _LLaMAHuggingFaceConverter().to_hg_config(config)

    save_hugging_face_model(tmp_path, {}, hg_config)

    native_config = AutoConfig.from_pretrained(tmp_path)

    assert native_config.rope_parameters == {
        "rope_theta": 500_000.0,
        "rope_type": "llama3",
        "factor": config.rope_scale.factor,
        "low_freq_factor": config.rope_scale.frequency_factors[0],
        "high_freq_factor": config.rope_scale.frequency_factors[1],
        "original_max_position_embeddings": config.rope_scale.original_context_length,
    }
