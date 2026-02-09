# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


# TODO(Phase 5): Implement tensor parallelism sharding specs
# Gemma3n sharding strategy should follow these patterns:
#
# Embeddings: Replicated
# - decoder_frontend.embed.weight
#
# Attention: Column/Row sharding
# - decoder.layers.*.self_attn.q_proj.weight: Column
# - decoder.layers.*.self_attn.k_proj.weight: Column
# - decoder.layers.*.self_attn.v_proj.weight: Column
# - decoder.layers.*.self_attn.output_proj.weight: Row
#
# FFN: Column/Row sharding
# - decoder.layers.*.ffn.gate_proj.weight: Column
# - decoder.layers.*.ffn.inner_proj.weight: Column
# - decoder.layers.*.ffn.output_proj.weight: Row
#
# AltUp FFN (local layers): Same as above
# - decoder.layers.*.altup_ffn.*: Same pattern
#
# PLE: Expert-parallel sharding
# - ple_modules.*.experts_up.*.weight: Expert-parallel
# - ple_modules.*.experts_down.*.weight: Expert-parallel
#
# Normalization: Replicated
# - decoder.layers.*.self_attn_layer_norm.weight
# - decoder.layers.*.ffn_layer_norm.weight
#
# Output: Column sharding
# - final_proj.weight: Column

__all__: list[str] = []
