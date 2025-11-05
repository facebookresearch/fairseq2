# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from typing import cast

import torch

from fairseq2.models.llama4.config import Llama4Config
from fairseq2.models.utils.checkpoint import convert_state_dict


def _get_indices_to_split_wqkv(
    config: Llama4Config, actual_wqkv_dim_0: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    Reference Llama 4 checkpoints have merged WQKV matrices.
    This method returns three tuples, (q_range, k_range, v_range). They can be used as:
        - Q: [0, k_end)
        - K: [k_start, v_start)
        - V: [v_start, -1)
    """
    # compute the expected unsharded combined wqkv dimension
    n_kv_heads = config.num_key_value_heads or config.num_attn_heads
    n_local_heads = config.num_attn_heads
    head_dim = config.model_dim // config.num_attn_heads
    n_local_kv_heads = n_kv_heads
    expected_wqkv_dim_0 = n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim
    # deduce the tensor-parallelism size
    tp_size = expected_wqkv_dim_0 // actual_wqkv_dim_0
    # recompute local kv head count
    n_local_heads = config.num_attn_heads // tp_size
    n_local_kv_heads = max(1, n_kv_heads // tp_size)
    # return sharded ranges
    return (
        (0, n_local_heads * head_dim),
        (
            n_local_heads * head_dim,
            n_local_heads * head_dim + n_local_kv_heads * head_dim,
        ),
        (
            n_local_heads * head_dim + n_local_kv_heads * head_dim,
            n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim,
        ),
    )


def convert_llama4_state_dict(
    state_dict: dict[str, object], config: Llama4Config
) -> dict[str, object]:
    # Check if we have a reference or Hugging Face checkpoint.
    if "lm_head.weight" in state_dict:  # HG
        raise ValueError("Llama 4 Huggingface checkpoint is not supported yet.")

    checkpoint: dict[str, object] = {}

    for k, v in state_dict.items():
        if ".moe_w_" in k:
            if config.experts is None:
                raise ValueError(
                    f"State dict contains MoE weights ({k}) but the Llama 4 config had a `experts` attribute set to `None`."
                )
            # Shard weights appropriately here depending on EP strategy.
            # The current strategies don't shard weights across their expert dimension.
            checkpoint[k] = v

        elif re.match(r"layers\.([0-9]+)\.attention\.wqkv\.weight", k):
            wqkv = cast(torch.Tensor, v)
            # split fused QKV weights (along the output dimension)
            q_range, k_range, v_range = _get_indices_to_split_wqkv(
                config, wqkv.shape[0]
            )

            wq = wqkv[q_range[0] : q_range[1]]
            wk = wqkv[k_range[0] : k_range[1]]
            wv = wqkv[v_range[0] : v_range[1]]

            wq_key = k.replace(".wqkv.", ".wq.")
            wk_key = k.replace(".wqkv.", ".wk.")
            wv_key = k.replace(".wqkv.", ".wv.")

            checkpoint[wq_key] = wq
            checkpoint[wk_key] = wk
            checkpoint[wv_key] = wv

        elif "rope.freqs" in k:
            # We do not need the pre-computed 'rope.freqs' buffers.
            pass

        elif "._extra_state" in k:
            # This might contain FP8 quantization info
            pass

        elif k.startswith("vision_"):
            # Disable vision encoder in this version
            pass

        elif k.endswith(".expert_activation_DE"):
            # This doesn't seem to be used in Scout or Maverick
            pass

        elif k.endswith("_stats_3E"):
            # Skip MoE running stat buffers
            pass

        else:
            checkpoint[k] = v

    key_map = {
        # fmt: off
        r"^layers\.([0-9]+)\.attention\.wq\.": r"decoder.layers.\1.self_attn.q_proj.",
        r"^layers\.([0-9]+)\.attention\.wk\.": r"decoder.layers.\1.self_attn.k_proj.",
        r"^layers\.([0-9]+)\.attention\.wv\.": r"decoder.layers.\1.self_attn.v_proj.",
        r"^layers\.([0-9]+)\.attention\.wo\.": r"decoder.layers.\1.self_attn.output_proj.",
        r"^layers\.([0-9]+)\.attention.wqkv.layer_norm_weight$": r"decoder.layers.\1.self_attn_layer_norm.weight",
        r"^layers\.([0-9]+)\.feed_forward\.w1\.": r"decoder.layers.\1.ffn.gate_proj.",
        r"^layers\.([0-9]+)\.feed_forward\.mlp\.fc2_weight$": r"decoder.layers.\1.ffn.output_proj.weight",
        r"^layers\.([0-9]+)\.feed_forward\.w3\.": r"decoder.layers.\1.ffn.inner_proj.",
        r"^layers\.([0-9]+)\.feed_forward\.mlp\.layer_norm_weight$": r"decoder.layers.\1.ffn_layer_norm.weight",
        # MoE: router
        r"^layers\.([0-9]+)\.feed_forward\.router_DE$": r"decoder.layers.\1.ffn.router",
        r"^layers\.([0-9]+)\.feed_forward\.expert_activation_DE$": r"decoder.layers.\1.ffn.router.expert_activation",
        r"^layers\.([0-9]+)\.feed_forward\.norm\.": r"decoder.layers.\1.ffn_layer_norm.",
        # MoE: shared expert
        r"^layers\.([0-9]+)\.feed_forward\.w_in_shared_FD\.": r"decoder.layers.\1.ffn.shared_expert.gate_proj.",
        r"^layers\.([0-9]+)\.feed_forward\.w_swiglu_FD\.": r"decoder.layers.\1.ffn.shared_expert.inner_proj.",
        r"^layers\.([0-9]+)\.feed_forward\.w_out_shared_DF\.": r"decoder.layers.\1.ffn.shared_expert.output_proj.",
        # MoE: grouped experts
        r"^layers\.([0-9]+)\.feed_forward\.experts\.moe_w_in_eD_F$": r"decoder.layers.\1.ffn.experts.gate_proj",
        r"^layers\.([0-9]+)\.feed_forward\.experts\.moe_w_swiglu_eD_F$": r"decoder.layers.\1.ffn.experts.inner_proj",
        r"^layers\.([0-9]+)\.feed_forward\.experts\.moe_w_out_eF_D$": r"decoder.layers.\1.ffn.experts.output_proj",
        # Initial layer norm
        r"^norm\.": r"decoder.layer_norm.",
        # Embeddings and input projections
        r"^tok_embeddings\.": r"decoder_frontend.embed.",
        r"^vision_embeddings\.": r"decoder_frontend.vision_embed.",
        r"^vision_projection\.": r"decoder_frontend.vision_proj.",
        r"^speech_embeddings\.embeddings\.": r"decoder_frontend.speech_embed.",
        # Output projections
        r"^output\.": r"final_proj.",
        # fmt: on
    }

    return convert_state_dict(checkpoint, key_map)
