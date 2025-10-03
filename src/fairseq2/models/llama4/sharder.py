# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module
from typing_extensions import override

from fairseq2.gang import Gangs
from fairseq2.models.llama4.config import Llama4Config
from fairseq2.models.llama4.moe import MoE
from fairseq2.models.transformer.experts import (
    GroupedExpertNetwork,
    TPShardedExpertNetwork,
)
from fairseq2.nn import ColumnShardedLinear, RowShardedLinear
from fairseq2.sharder import ModuleSharder, ShardSpec


def get_llama4_shard_specs(config: Llama4Config) -> dict[str, ShardSpec]:
    embed_dim = 1 if config.shard_embed_dim else 0

    return {
        # fmt: off
        r".*\.embed$":                         ShardSpec(dim=embed_dim),
        # Attention sharding
        r".*\.self_attn.q_proj$":              ShardSpec(dim=0, region_boundary=True),
        r".*\.self_attn.k_proj$":              ShardSpec(dim=0, region_boundary=True),
        r".*\.self_attn.v_proj$":              ShardSpec(dim=0, region_boundary=True),
        r".*\.self_attn.output_proj$":         ShardSpec(dim=1, region_boundary=True),
        # FFN sharding: GLU FFN
        r".*\.ffn.inner_proj$":                ShardSpec(dim=0, region_boundary=True),
        r".*\.ffn.gate_proj$":                 ShardSpec(dim=0, region_boundary=True),
        r".*\.ffn.output_proj$":               ShardSpec(dim=1, region_boundary=True),
        # FFN sharding: MoE
        r".*\.ffn$":                           ShardSpec(dim=-1),  # MoESharder
        # Final proj sharding
        r"^final_proj$":                       ShardSpec(dim=0),
        # fmt: on
    }


class MoESharder(ModuleSharder):
    @override
    def shard(self, module: Module, gangs: Gangs, spec: ShardSpec) -> Module:
        if not isinstance(module, MoE):
            raise TypeError(
                f"`module` must be of type `{MoE}`, but is of type `{type(module)}` instead."
            )

        # make the MoE layer aware of the TP gang, for the reduce of the output
        module.tp_gang = gangs.tp

        # shard the shared expert
        if module.shared_expert is not None:
            module.shared_expert.gate_proj = ColumnShardedLinear.from_linear(  # type: ignore[assignment]
                module.shared_expert.gate_proj,  # type: ignore[arg-type]
                gangs.tp,
                gather_output=False,
            )
            module.shared_expert.inner_proj = ColumnShardedLinear.from_linear(  # type: ignore[assignment]
                module.shared_expert.inner_proj,  # type: ignore[arg-type]
                gangs.tp,
                gather_output=False,
            )
            module.shared_expert.output_proj = RowShardedLinear.from_linear(  # type: ignore[assignment]
                module.shared_expert.output_proj,  # type: ignore[arg-type]
                gangs.tp,
                reduce_output=False,
            )

        # shard the experts
        if not isinstance(module.experts, GroupedExpertNetwork):
            raise TypeError(
                f"Expected `module.experts` to be of type `GroupedExpertNetwork`, got {type(module.experts)}."
            )

        module.experts = TPShardedExpertNetwork.from_grouped_expert_network(
            module.experts,
            gangs.tp,
            reduce_output=False,
        )

        return module

    @property
    @override
    def supported_module_kls(self) -> type[Module]:
        return MoE
