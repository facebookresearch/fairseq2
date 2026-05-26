# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor-parallelism shard specifications for Gemma 4 models.

Defines how each module's parameters should be partitioned across TP ranks.
Following the same pattern as :mod:`fairseq2.models.llama.sharder` and
:mod:`fairseq2.models.qwen.sharder`.

Sharding strategy:
    - **Embedding**: column-sharded along vocab dim (dim=0)
    - **Q/K/V projections**: column-sharded along head dim (dim=0)
    - **Output projection**: row-sharded along input dim (dim=1)
    - **FFN inner/gate**: column-sharded (dim=0)
    - **FFN output**: row-sharded (dim=1)
    - **Final projection**: column-sharded (dim=0)
    - **PLE projections**: column/row-sharded following FFN pattern
    - **MoE router proj**: NOT sharded (routing must be consistent across ranks)
    - **MoE experts**: NOT sharded at spec level (Gemma4 uses fused 3-D Parameters,
      not ``GroupedExpertNetwork``; sharding would require a custom ``ModuleSharder``)

Limitations:
    - **Audio tower** is not sharded -- it is replicated on every TP rank.
      Acceptable for E2B/E4B multimodal (tower is small), but explicit
      coverage would be needed to TP the full multimodal model.
    - **MoE experts** (26B-A4B) are not sharded.  Use Expert Parallelism (EP)
      or a custom ``ModuleSharder`` analogous to LLaMA 4's ``MoESharder`` --
      adapted for ``Gemma4Experts``' fused 3-D ``Parameter`` layout
      (``(E, 2*I, D)`` and ``(E, D, I)``).
    - **CONSUMER layers** (KV sharing) lack ``k_proj`` / ``v_proj`` / ``k_norm`` /
      ``v_norm`` modules altogether.  The regexes above target those names but
      safely produce zero matches on consumer layers, so no special handling
      is required here.
"""

from __future__ import annotations

from fairseq2.models.gemma4.config import Gemma4Config
from fairseq2.sharder import ShardSpec


def get_gemma4_shard_specs(config: Gemma4Config) -> dict[str, ShardSpec]:
    """Return tensor-parallelism shard specifications for a Gemma 4 model.

    :param config: The Gemma 4 configuration.
    :returns: A mapping from module-name regex to :class:`ShardSpec`.
    """
    specs: dict[str, ShardSpec] = {
        # fmt: off
        # ---- Embedding ----
        r".*\.embed$":                      ShardSpec(dim=0),

        # ---- Self-attention ----
        r".*\.self_attn\.q_proj$":          ShardSpec(dim=0, region_boundary=True),
        r".*\.self_attn\.k_proj$":          ShardSpec(dim=0, region_boundary=True),
        r".*\.self_attn\.v_proj$":          ShardSpec(dim=0, region_boundary=True),
        r".*\.self_attn\.output_proj$":     ShardSpec(dim=1, region_boundary=True),

        # ---- Dense FFN (GLU) ----
        r".*\.ffn\.inner_proj$":            ShardSpec(dim=0, region_boundary=True),
        r".*\.ffn\.gate_proj$":             ShardSpec(dim=0, region_boundary=True),
        r".*\.ffn\.output_proj$":           ShardSpec(dim=1, region_boundary=True),

        # ---- Final projection ----
        r"^final_proj$":                    ShardSpec(dim=0),
        # fmt: on
    }

    # ---- PLE projections (E4B only) ----
    if config.has_ple:
        specs.update(
            {
                # fmt: off
            # per_layer_input_gate: (model_dim -> ple_dim) — column shard
            r".*\.per_layer_input_gate$":       ShardSpec(dim=0, region_boundary=True),
            # per_layer_projection: (ple_dim -> model_dim) — row shard
            r".*\.per_layer_projection$":       ShardSpec(dim=1, region_boundary=True),
            # per_layer_model_projection in frontend: (model_dim -> L*ple_dim) — column shard
            r".*\.per_layer_model_projection$": ShardSpec(dim=0, region_boundary=True),
                # fmt: on
            }
        )

    # Note: MoE router.proj and experts (gate_up_proj, down_proj) are NOT
    # sharded via ShardSpec because:
    #   1. Router proj must produce identical routing decisions across TP ranks.
    #   2. Gemma4Experts uses fused 3-D Parameters (not GroupedExpertNetwork),
    #      so they cannot be handled by the standard LinearSharder.
    #   Expert parallelism for 26B-A4B would require a custom ModuleSharder
    #   similar to LLaMA4's MoESharder, but adapted for Gemma4Experts' 3-D layout.

    return specs
