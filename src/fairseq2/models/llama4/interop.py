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


def get_indices_to_split_wqkv(
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
    # recompute local dimensions
    n_kv_heads = config.num_key_value_heads or config.num_attn_heads
    n_local_heads = config.num_attn_heads // tp_size
    head_dim = config.model_dim // config.num_attn_heads
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
    try:
        state_dict = cast(dict[str, object], state_dict["model"])  # legacy
    except KeyError:
        pass

    loaded_checkpoint = state_dict

    # Check if we have a reference or Hugging Face checkpoint.
    if "lm_head.weight" in loaded_checkpoint:  # HG
        raise ValueError("Llama 4 Huggingface checkpoint is not supported yet.")
    else:
        checkpoint: dict[str, object] = {}  # type: ignore

        for k, v in loaded_checkpoint.items():
            if ".moe_w_" in k:
                assert isinstance(v, torch.Tensor)
                assert config.experts is not None

                num_experts = config.experts.num_experts

                checkpoint[k] = v.unflatten(0, (num_experts, -1))

            elif re.match(r"layers\.([0-9]+)\.attention\.wqkv\.weight", k):
                assert isinstance(v, torch.Tensor)
                # split fused QKV weights (along the output dimension)
                q_range, k_range, v_range = get_indices_to_split_wqkv(
                    config, v.shape[0]
                )

                wq = v[q_range[0] : q_range[1]]
                wk = v[k_range[0] : k_range[1]]
                wv = v[v_range[0] : v_range[1]]

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

            elif k.startswith("vision_") and not config.vision_config:
                # Skip vision layers if vision config is disabled
                pass

            elif k.endswith(".expert_activation_DE"):
                # This doesn't seem to be used in Scout or Maverick
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
            r"^layers\.([0-9]+)\.feed_forward\.router_DE$": r"decoder.layers.\1.ffn.router.gate",
            r"^layers\.([0-9]+)\.feed_forward\.expert_activation_DE$": r"decoder.layers.\1.ffn.router.expert_activation",
            r"^layers\.([0-9]+)\.feed_forward\.norm\.": r"decoder.layers.\1.ffn_layer_norm.",
            r"^layers\.([0-9]+)\.feed_forward\.running_gate_stats_3E$": r"decoder.layers.\1.ffn.running_gate_stats",
            r"^layers\.([0-9]+)\.feed_forward\.global_gate_stats_3E$": r"decoder.layers.\1.ffn.global_gate_stats",
            # MoE: shared expert
            r"^layers\.([0-9]+)\.feed_forward\.w_in_shared_FD\.": r"decoder.layers.\1.ffn.shared_expert.gate_proj.",
            r"^layers\.([0-9]+)\.feed_forward\.w_swiglu_FD\.": r"decoder.layers.\1.ffn.shared_expert.inner_proj.",
            r"^layers\.([0-9]+)\.feed_forward\.w_out_shared_DF\.": r"decoder.layers.\1.ffn.shared_expert.output_proj.",
            # MoE: grouped experts
            r"^layers\.([0-9]+)\.feed_forward\.experts\.moe_w_in_eD_F$": r"decoder.layers.\1.ffn.experts.gate_proj.weight",
            r"^layers\.([0-9]+)\.feed_forward\.experts\.moe_w_swiglu_eD_F$": r"decoder.layers.\1.ffn.experts.inner_proj.weight",
            r"^layers\.([0-9]+)\.feed_forward\.experts\.moe_w_out_eF_D$": r"decoder.layers.\1.ffn.experts.output_proj.weight",
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

    state_dict = convert_state_dict(checkpoint, key_map)

    return state_dict


# 'vision_projection.weight',
# 'tok_embeddings.weight',
# 'norm.weight',
# 'output.weight',
# 'layers.0.feed_forward.norm.weight', 'layers.0.attention.wo.weight', 'layers.0.feed_forward.global_gate_stats_3E', 'layers.0.feed_forward.expert_activation_DE', 'layers.0.feed_forward.running_gate_stats_3E', 'layers.0.feed_forward.router_DE', 'layers.0.feed_forward.w_in_shared_FD.weight', 'layers.0.feed_forward.w_swiglu_FD.weight', 'layers.0.feed_forward.w_out_shared_DF.weight', 'layers.0.feed_forward.experts.moe_w_in_eD_F', 'layers.0.feed_forward.experts.moe_w_swiglu_eD_F', 'layers.0.feed_forward.experts.moe_w_out_eF_D', 'layers.0.attention.wqkv.weight', 'layers.0.attention.wqkv.layer_norm_weight',
# 'layers.1.feed_forward.norm.weight', 'layers.1.attention.wo.weight', 'layers.1.feed_forward.expert_activation_DE', 'layers.1.feed_forward.running_gate_stats_3E', 'layers.1.feed_forward.global_gate_stats_3E', 'layers.1.feed_forward.router_DE', 'layers.1.feed_forward.w_in_shared_FD.weight', 'layers.1.feed_forward.w_swiglu_FD.weight', 'layers.1.feed_forward.w_out_shared_DF.weight', 'layers.1.feed_forward.experts.moe_w_in_eD_F', 'layers.1.feed_forward.experts.moe_w_swiglu_eD_F', 'layers.1.feed_forward.experts.moe_w_out_eF_D', 'layers.1.attention.wqkv.weight', 'layers.1.attention.wqkv.layer_norm_weight',
#
# 'vision_embeddings.vision_adapter.mlp.c_fc.weight',
# 'vision_embeddings.vision_adapter.mlp.c_proj.weight',
# 'vision_embeddings.vision_encoder.class_embedding',
# 'vision_embeddings.vision_encoder.conv1._linear.weight',
# 'vision_embeddings.vision_encoder.ln_post.bias',
# 'vision_embeddings.vision_encoder.ln_post.weight',
# 'vision_embeddings.vision_encoder.ln_pre.bias',
# 'vision_embeddings.vision_encoder.ln_pre.weight',
# 'vision_embeddings.vision_encoder.positional_embedding_vlm',
# 'vision_embeddings.vision_encoder.transformer.resblocks.0.attn.wk.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.0.attn.wk.weight', 'vision_embeddings.vision_encoder.transformer.resblocks.0.attn.wo.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.0.attn.wo.weight', 'vision_embeddings.vision_encoder.transformer.resblocks.0.attn.wq.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.0.attn.wq.weight', 'vision_embeddings.vision_encoder.transformer.resblocks.0.attn.wv.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.0.attn.wv.weight', 'vision_embeddings.vision_encoder.transformer.resblocks.0.ln_1.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.0.ln_1.weight', 'vision_embeddings.vision_encoder.transformer.resblocks.0.ln_2.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.0.ln_2.weight', 'vision_embeddings.vision_encoder.transformer.resblocks.0.mlp.c_fc.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.0.mlp.c_fc.weight', 'vision_embeddings.vision_encoder.transformer.resblocks.0.mlp.c_proj.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.0.mlp.c_proj.weight',
# 'vision_embeddings.vision_encoder.transformer.resblocks.1.attn.wk.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.1.attn.wk.weight', 'vision_embeddings.vision_encoder.transformer.resblocks.1.attn.wo.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.1.attn.wo.weight', 'vision_embeddings.vision_encoder.transformer.resblocks.1.attn.wq.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.1.attn.wq.weight', 'vision_embeddings.vision_encoder.transformer.resblocks.1.attn.wv.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.1.attn.wv.weight', 'vision_embeddings.vision_encoder.transformer.resblocks.1.ln_1.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.1.ln_1.weight', 'vision_embeddings.vision_encoder.transformer.resblocks.1.ln_2.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.1.ln_2.weight', 'vision_embeddings.vision_encoder.transformer.resblocks.1.mlp.c_fc.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.1.mlp.c_fc.weight', 'vision_embeddings.vision_encoder.transformer.resblocks.1.mlp.c_proj.bias', 'vision_embeddings.vision_encoder.transformer.resblocks.1.mlp.c_proj.weight',
