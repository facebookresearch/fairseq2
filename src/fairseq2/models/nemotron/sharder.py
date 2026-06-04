# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor parallelism shard specifications for NemotronH.

NOTE: The sharder/shard_specs API is deprecated in fairseq2 v0.6 and will be
removed in v0.12. The preferred approach is to handle parallelism in the model
factory (see factory.py). These specs are provided for backward compatibility
with the deprecated ModelSharder API.

For new code, use NemotronHFactory which handles TP sharding internally:
- Attention: via StandardMultiheadAttention(gangs=...)
- MoE: via factory._shard_moe() with ColumnShardedLinear/RowShardedLinear
- Embedding/final_proj: via VocabShardedEmbedding/ColumnShardedLinear
"""

from __future__ import annotations

from torch.nn import Module
from typing_extensions import override

from fairseq2.gang import Gangs
from fairseq2.models.nemotron.config import NemotronHConfig
from fairseq2.models.nemotron.moe import NemotronHMoE
from fairseq2.nn import ColumnShardedLinear, RowShardedLinear
from fairseq2.sharder import ModuleSharder, ShardSpec


def get_nemotron_h_shard_specs(config: NemotronHConfig) -> dict[str, ShardSpec]:
    """Get tensor parallelism shard specifications for NemotronH.

    These are used by the deprecated ModelSharder API for backward compatibility.
    The factory handles TP natively when gangs are available.
    """
    return {
        # fmt: off
        # Embedding — vocab-sharded
        r".*\.embed$":                                  ShardSpec(dim=0),

        # Attention blocks — GQA column/row sharding
        # These match the Linear submodules of StandardMultiheadAttention
        r".*\.mixer\.q_proj$":                          ShardSpec(dim=0, region_boundary=True),
        r".*\.mixer\.k_proj$":                          ShardSpec(dim=0, region_boundary=True),
        r".*\.mixer\.v_proj$":                          ShardSpec(dim=0, region_boundary=True),
        r".*\.mixer\.output_proj$":                     ShardSpec(dim=1, region_boundary=True),

        # MoE blocks — shared expert sharding (individual expert sharding
        # via the custom NemotronHMoESharder is not practical with the
        # deprecated regex-based API due to 128 experts; use the factory instead)
        r".*\.mixer\.shared_experts\.up_proj$":         ShardSpec(dim=0, region_boundary=True),
        r".*\.mixer\.shared_experts\.down_proj$":       ShardSpec(dim=1, region_boundary=True),

        # Final projection
        r"^final_proj$":                                ShardSpec(dim=0),
        # fmt: on
    }


class NemotronHMoESharder(ModuleSharder):
    """Custom sharder for NemotronH MoE blocks (deprecated API).

    Sets tp_gang on the MoE module and shards shared + routed expert
    projections for tensor parallelism.

    NOTE: The factory handles this natively; this sharder exists only for
    backward compatibility with the deprecated ModelSharder framework.
    """

    @override
    def shard(self, module: Module, gangs: Gangs, spec: ShardSpec) -> Module:
        if not isinstance(module, NemotronHMoE):
            raise TypeError(
                f"`module` must be of type `{NemotronHMoE}`, but is of type "
                f"`{type(module)}` instead."
            )

        # Set tp_gang for the all-reduce in forward()
        module.tp_gang = gangs.tp

        # Shard the shared expert
        module.shared_experts.up_proj = ColumnShardedLinear.from_linear(  # type: ignore[assignment]
            module.shared_experts.up_proj,  # type: ignore[arg-type]
            gangs.tp,
            gather_output=False,
        )
        module.shared_experts.down_proj = RowShardedLinear.from_linear(  # type: ignore[assignment]
            module.shared_experts.down_proj,  # type: ignore[arg-type]
            gangs.tp,
            reduce_output=False,
        )

        # Shard each routed expert
        for expert in module.experts:
            expert.up_proj = ColumnShardedLinear.from_linear(  # type: ignore[assignment]
                expert.up_proj,  # type: ignore[arg-type]
                gangs.tp,
                gather_output=False,
            )
            expert.down_proj = RowShardedLinear.from_linear(  # type: ignore[assignment]
                expert.down_proj,  # type: ignore[arg-type]
                gangs.tp,
                reduce_output=False,
            )

        return module

    @property
    @override
    def supported_module_kls(self) -> type[Module]:
        return NemotronHMoE
