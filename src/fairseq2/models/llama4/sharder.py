# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing_extensions import override

from torch.nn import Module

import fairseq2.runtime.dependency
from fairseq2.models.llama4.config import Llama4Config
from fairseq2.models.llama4.model.moe.moe import MoE
from fairseq2.sharder import ShardSpec, ModuleSharder
from fairseq2.gang import Gangs
from fairseq2.nn import (
    BatchColumnShardedLinear,
    BatchLinear,
    BatchRowShardedLinear,
    ColumnShardedLinear,
    Linear,
    RowShardedLinear,
)
from fairseq2.models.transformer import GLUFeedForwardNetwork


def get_llama4_shard_specs(config: Llama4Config) -> dict[str, ShardSpec]:
    embed_dim = 1 if config.shard_embed_dim else 0
    
    # Register the MoE sharder
    resolver = fairseq2.runtime.dependency._resolver
    assert (
        resolver is not None
        and isinstance(resolver, fairseq2.runtime.dependency.DependencyContainer)
    )
    resolver.collection.register_type(ModuleSharder, MoESharder)

    return {
        # fmt: off
        r".*\.embed$":                 ShardSpec(dim=embed_dim),
        # TODO(mgleize): add vision embed sharding
        # TODO(mgleize): add vision proj sharding
        r".*\.self_attn.q_proj$":      ShardSpec(dim=0, region_boundary=True),
        r".*\.self_attn.k_proj$":      ShardSpec(dim=0, region_boundary=True),
        r".*\.self_attn.v_proj$":      ShardSpec(dim=0, region_boundary=True),
        r".*\.self_attn.output_proj$": ShardSpec(dim=1, region_boundary=True),
        # FFN sharding: regular GLU FFN
        r".*\.ffn.inner_proj$":        ShardSpec(dim=0, region_boundary=True),
        r".*\.ffn.gate_proj$":         ShardSpec(dim=0, region_boundary=True),
        r".*\.ffn.output_proj$":       ShardSpec(dim=1, region_boundary=True),
        # FFN sharding: MoE
        r".*\.ffn$":                   ShardSpec(dim=-1),
        r"^final_proj$":               ShardSpec(dim=0),
        # fmt: on
    }


class MoESharder(ModuleSharder):
    @override
    def shard(self, module: Module, gangs: Gangs, spec: ShardSpec) -> Module:
        if not isinstance(module, MoE):
            raise TypeError(
                f"`module` must be of type `{MoE}`, but is of type `{type(module)}` instead."
            )
        
        tp_gang = gangs.tp
        
        def shard_glu_ffn(
            m: GLUFeedForwardNetwork,
            reduce_output: bool = True,
        ) -> None:
            for proj in (m.gate_proj, m.inner_proj, m.output_proj):
                if not isinstance(proj, Linear):
                    return

            # Scatter.
            m.gate_proj = ColumnShardedLinear.from_linear(
                m.gate_proj, tp_gang, gather_output=False
            )

            m.inner_proj = ColumnShardedLinear.from_linear(
                m.inner_proj, tp_gang, gather_output=False
            )

            # Gather.
            m.output_proj = RowShardedLinear.from_linear(
                m.output_proj, tp_gang, scatter_input=False, reduce_output=reduce_output
            )
        
        def shard_moe(m: MoE) -> None:
            for proj in (m.experts.gate_proj, m.experts.inner_proj, m.experts.output_proj):
                if not isinstance(proj, BatchLinear):
                    return

            # Shard shared expert without reducing its output
            shard_glu_ffn(m.shared_expert, reduce_output=False)

            # Shard expert layers on TP gang
            m.experts.gate_proj = BatchColumnShardedLinear.from_batch_linear(
                m.experts.gate_proj, tp_gang, gather_output=False
            )

            m.experts.inner_proj = BatchColumnShardedLinear.from_batch_linear(
                m.experts.inner_proj, tp_gang, gather_output=False
            )

            m.experts.output_proj = BatchRowShardedLinear.from_batch_linear(
                m.experts.output_proj, tp_gang, scatter_input=False, reduce_output=False
            )

            # Set the gang of the MoE module, to force output reduce at the end
            m.tp_gang = tp_gang
        
        shard_moe(module)
        
        return module
        
    @property
    @override
    def supported_module_kls(self) -> type[Module]:
        return MoE
