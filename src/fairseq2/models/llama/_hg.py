# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

from torch import Tensor
import torch

from fairseq2.models.utils.checkpoint import convert_checkpoint, get_converted_key

try:
    import transformers.models as transformers_models  # type: ignore[import-not-found]
    from transformers import PretrainedConfig  # type: ignore[import-not-found]
except ImportError:
    raise ImportError(
        "transformers package is required to fetch Qwen Config for export purpose, run `pip install transformers`"
    )

# isort: split

from fairseq2.models.llama._config import LLaMAConfig


def export_llama_checkpoint(
    checkpoint: dict[str, object], config: LLaMAConfig
) -> tuple[dict[str, object], PretrainedConfig]:
    hg_config = _convert_config(config)

    hg_checkpoint = _convert_checkpoint(checkpoint, config)

    return hg_checkpoint, hg_config


def _convert_config(config: LLaMAConfig) -> PretrainedConfig:
    multiplier = config.ffn_inner_dim_multiplier

    multiple_of = config.ffn_inner_dim_multiple_of

    intermediate_size = multiple_of * ((int(multiplier * int(8 * config.model_dim / 3)) + multiple_of - 1) // multiple_of)  # fmt: skip

    if config.rope_scale is not None:
        rope_scale = {
            "factor": config.rope_scale.factor,
            "low_freq_factor": config.rope_scale.frequency_factors[0],
            "high_freq_factor": config.rope_scale.frequency_factors[1],
            "original_max_position_embeddings": config.rope_scale.original_context_length,
            "rope_type": "llama3",
        }
    else:
        rope_scale = None

    if config.vocab_size == 32_000:  # LLaMA 1 and 2
        bos_idx = 1
        eos_idx = 2
    else:
        bos_idx = 128_000
        eos_idx = 128_001

    config_cls = getattr(transformers_models, config.hg_config_class)

    config_mapped_to_hg = {
        "bos_token_id": bos_idx,
        "eos_token_id": eos_idx,
        "hidden_size": config.model_dim,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": config.max_seq_len,
        "model_type": "llama",
        "num_attention_heads": config.num_attn_heads,
        "num_hidden_layers": config.num_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "rms_norm_eps": 1e-5,
        "rope_scaling": rope_scale,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": config.tied_embeddings,
        "vocab_size": config.vocab_size,
    }

    hg_config = config_cls()

    for k, v in config_mapped_to_hg.items():
        if getattr(hg_config, k, None) is not None:
            setattr(hg_config, k, v)

    # always add architectures in the end since its used by vllm
    setattr(hg_config, "architectures", config.hg_architectures)

    return hg_config


def _convert_checkpoint(
    checkpoint: dict[str, object], config: LLaMAConfig
) -> dict[str, object]:
    head_dim = config.model_dim // config.num_attn_heads

    def permute_rotary(w: Tensor, num_heads: int) -> Tensor:
        # (H, M) -> (H_d, D / 2, 2, M)
        w = w.view(num_heads, head_dim // 2, 2, config.model_dim)

        # (H_d, D / 2, 2, M) -> (H_d, 2, D / 2, m)
        w = w.transpose(1, 2)

        # (H_d, 2, D / 2, M) -> (H, M)
        return w.reshape(-1, config.model_dim)

    for idx in range(config.num_layers):
        q_key = f"decoder.layers.{idx}.self_attn.q_proj.weight"
        k_key = f"decoder.layers.{idx}.self_attn.k_proj.weight"

        q_proj = cast(Tensor, checkpoint[q_key])
        k_proj = cast(Tensor, checkpoint[k_key])

        q_proj = permute_rotary(q_proj, config.num_attn_heads)
        k_proj = permute_rotary(k_proj, config.num_key_value_heads)

        checkpoint[q_key] = q_proj
        checkpoint[k_key] = k_proj

    key_map = {
        # fmt: off
        r"decoder\.layers\.([0-9]+)\.self_attn\.q_proj.":      r"model.layers.\1.self_attn.q_proj.",
        r"decoder\.layers\.([0-9]+)\.self_attn\.k_proj.":      r"model.layers.\1.self_attn.k_proj.",
        r"decoder\.layers\.([0-9]+)\.self_attn\.v_proj.":      r"model.layers.\1.self_attn.v_proj.",
        r"decoder\.layers\.([0-9]+)\.self_attn\.output_proj.": r"model.layers.\1.self_attn.o_proj.",
        r"decoder\.layers\.([0-9]+)\.ffn_layer_norm\.":        r"model.layers.\1.post_attention_layernorm.",
        r"decoder\.layers\.([0-9]+)\.ffn.gate_proj\.":         r"model.layers.\1.mlp.gate_proj.",
        r"decoder\.layers\.([0-9]+)\.ffn.output_proj\.":       r"model.layers.\1.mlp.down_proj.",
        r"decoder\.layers\.([0-9]+)\.ffn.inner_proj\.":        r"model.layers.\1.mlp.up_proj.",
        r"decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":  r"model.layers.\1.input_layernorm.",
        r"decoder\.layer_norm\.":                              r"model.norm.",
        r"decoder_frontend.embed\.":                           r"model.embed_tokens.",
        r"final_proj\.":                                       r"lm_head.",
        # fmt: on
    }

    checkpoint = convert_checkpoint(checkpoint, key_map)

    if config.tied_embeddings:
        del checkpoint["lm_head.weight"]

    return checkpoint


def _convert_parameter(name: str,
    parameter: torch.nn.Parameter, config: LLaMAConfig
) -> dict[str, object]:
    head_dim = config.model_dim // config.num_attn_heads

    def permute_rotary(w: Tensor, num_heads: int) -> Tensor:
        # (H, M) -> (H_d, D / 2, 2, M)
        w = w.view(num_heads, head_dim // 2, 2, config.model_dim)

        # (H_d, D / 2, 2, M) -> (H_d, 2, D / 2, m)
        w = w.transpose(1, 2)

        # (H_d, 2, D / 2, M) -> (H, M)
        return w.reshape(-1, config.model_dim)

    if "q_proj" in name:
        parameter = permute_rotary(parameter, config.num_attn_heads)

    if "k_proj" in name:
        parameter = permute_rotary(parameter, config.num_key_value_heads)

    key_map = {
        # fmt: off
        r"decoder\.layers\.([0-9]+)\.self_attn\.q_proj.":      r"model.layers.\1.self_attn.q_proj.",
        r"decoder\.layers\.([0-9]+)\.self_attn\.k_proj.":      r"model.layers.\1.self_attn.k_proj.",
        r"decoder\.layers\.([0-9]+)\.self_attn\.v_proj.":      r"model.layers.\1.self_attn.v_proj.",
        r"decoder\.layers\.([0-9]+)\.self_attn\.output_proj.": r"model.layers.\1.self_attn.o_proj.",
        r"decoder\.layers\.([0-9]+)\.ffn_layer_norm\.":        r"model.layers.\1.post_attention_layernorm.",
        r"decoder\.layers\.([0-9]+)\.ffn.gate_proj\.":         r"model.layers.\1.mlp.gate_proj.",
        r"decoder\.layers\.([0-9]+)\.ffn.output_proj\.":       r"model.layers.\1.mlp.down_proj.",
        r"decoder\.layers\.([0-9]+)\.ffn.inner_proj\.":        r"model.layers.\1.mlp.up_proj.",
        r"decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":  r"model.layers.\1.input_layernorm.",
        r"decoder\.layer_norm\.":                              r"model.norm.",
        r"decoder_frontend.embed\.":                           r"model.embed_tokens.",
        r"final_proj\.":                                       r"lm_head.",
        # fmt: on
    }

    converted_name = get_converted_key(name, key_map)

    return converted_name, parameter
