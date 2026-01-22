# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod

from torch.nn import Module, Parameter


class Sharded(ABC):
    """
    Indicates that a PyTorch module has tensor-sharded parameters.

    Modules implementing this interface must specify which of their parameters
    are sharded and along which dimensions the sharding occurs.

    See :class:`fairseq2.nn.ColumnShardedLinear` for an example.
    """

    @abstractmethod
    def get_shard_dims(self) -> list[tuple[Parameter, int]]:
        """
        Returns the sharding information for this module's parameters.

        This function returns a list of tuples where each tuple contains the
        sharded parameter within this module and the tensor dimension along
        which the parameter is sharded.
        """


def get_shard_dims(module: Module) -> dict[str, int]:
    """
    Recursively traverses ``module`` and collects sharding information from all
    :class:`Sharded` modules.

    Returns a dictionary mapping fully qualified parameter names to their
    sharded dimensions.

    .. code:: python
        :caption: Example usage with sharded embedding layers

        # Given a model with sharded embeddings
        model = TransformerModel(...)

        # Get all sharding information
        shard_dims = get_shard_dims(model)

        # Returns: {
        #   "encoder.embed_tokens.weight": 0,
        #   "decoder.embed_tokens.weight": 0,
        #   "encoder.layers.0.self_attn.q_proj.weight": 1,
        #   "encoder.layers.0.self_attn.q_proj.bias": 1,
        #   ...
        # }
    """
    shard_dims = {}

    for m in module.modules():
        if isinstance(m, Sharded):
            for param, shard_dim in m.get_shard_dims():
                if param not in shard_dims:
                    shard_dims[param] = shard_dim

    output = {}

    for param_name, param in module.named_parameters():
        dim = shard_dims.get(param)
        if dim is not None:
            output[param_name] = dim

    return output
