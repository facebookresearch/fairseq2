# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import embedding
from torch.nn.parameter import Parameter
from typing_extensions import override

from fairseq2.gang import Gang
from fairseq2.nn.utils.module import to_empty
from fairseq2.tensor_parallel import gather, reduce, reduce_on_backward
from fairseq2.typing import META, DataType, Device


class Embedding(Module, ABC):
    """Stores embeddings of a fixed dictionary and size."""

    num_embeddings: int
    embedding_dim: int
    pad_idx: int | None
    padding_idx: int | None  # compat

    def __init__(
        self, num_embeddings: int, embedding_dim: int, pad_idx: int | None = None
    ) -> None:
        """
        :param num_embeddings:
            The size of the embedding table.
        :param embedding_dim:
            The dimensionality of returned embeddings.
        :param pad_idx:
            If not ``None``, entries at ``pad_idx`` do not contribute to the
            gradient; therefore, the embedding at ``pad_idx`` is not updated
            during training.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.pad_idx = pad_idx

        # Alias field for compatibility with `torch.nn.Embedding`.
        self.padding_idx = pad_idx

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:
            The embedding indices. *Shape:* Any.

        :returns:
            The embeddings corresponding to the specified indices. *Shape:*
            :math:`(*,E)`, where :math:`*` is the input shape and :math:`E` is
            the dimensionality of the embeddings.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        s = f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"

        if self.pad_idx is not None:
            s = f"{s}, pad_idx={self.pad_idx}"

        return s


@final
class StandardEmbedding(Embedding):
    """Stores embeddings of a fixed dictionary and size in an in-memory table."""

    weight: Parameter
    init_fn: Callable[[StandardEmbedding], None] | None

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        pad_idx: int | None = None,
        *,
        init_fn: Callable[[StandardEmbedding], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param num_embeddings:
            The size of the embedding table.
        :param embedding_dim:
            The dimensionality of returned embeddings.
        :param pad_idx:
            If not ``None``, entries at ``pad_idx`` do not contribute to the
            gradient; therefore, the embedding at ``pad_idx`` is not updated
            during training.
        :param init_fn:
            The callable to initialize the embedding table.
        """
        super().__init__(num_embeddings, embedding_dim, pad_idx)

        self.weight = Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.init_fn is not None:
            self.init_fn(self)

            return

        nn.init.normal_(self.weight)

        if self.pad_idx is not None:
            with torch.no_grad():
                self.weight[self.pad_idx].fill_(0.0)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return embedding(x, self.weight, self.pad_idx)

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.init_fn is not None:
            init_fn = getattr(self.init_fn, "__name__", self.init_fn)

            s = f"{s}, init_fn={init_fn}"

        return s


@final
class VocabShardedEmbedding(Embedding):
    """Represents a :class:`StandardEmbedding` that is sharded across its
    vocabulary dimension."""

    gang: Gang
    sharded_num_embeddings: int
    weight: Parameter
    init_fn: Callable[[StandardEmbedding], None] | None

    @staticmethod
    def from_embedding(embed: StandardEmbedding, gang: Gang) -> VocabShardedEmbedding:
        """Construct a :class:`VocabShardedEmbedding` by sharding ``embed``.

        :param embed:
            The embedding to shard.
        :param gang:
            The gang over which to shard ``embed``.
        """
        device = embed.weight.device

        if device != gang.device and device.type != "meta":
            raise ValueError(
                "The device of `embed` must either match `gang.device` or must be of type `meta`."
            )

        sharded = VocabShardedEmbedding(
            gang,
            embed.num_embeddings,
            embed.embedding_dim,
            embed.pad_idx,
            init_fn=embed.init_fn,
            device=META,
            dtype=embed.weight.dtype,
        )

        if device.type != "meta":
            to_empty(sharded, device)

        sharded._copy_weight(embed)

        return sharded

    def __init__(
        self,
        gang: Gang,
        num_embeddings: int,
        embedding_dim: int,
        pad_idx: int | None = None,
        *,
        init_fn: Callable[[StandardEmbedding], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param gang:
            The gang over which to shard the embedding table.
        :param num_embeddings:
            The size of the embedding table.
        :param embedding_dim:
            The dimensionality of returned embeddings.
        :param pad_idx:
            If not ``None``, entries at ``pad_idx`` do not contribute to the
            gradient; therefore, the embedding at ``pad_idx`` is not updated
            during training.
        :param init_fn:
            The callable to initialize the embedding table.
        """
        super().__init__(num_embeddings, embedding_dim, pad_idx)

        if num_embeddings % gang.size != 0:
            raise ValueError(
                f"`num_embeddings` must be a multiple of `gang.size` ({gang.size}), but is {num_embeddings} instead."
            )

        self.gang = gang

        self.sharded_num_embeddings = num_embeddings // gang.size

        if device is None:
            device = gang.device
        elif device != gang.device and device.type != "meta":
            raise ValueError(
                "`device` must either match `gang.device` or must be of type `meta`."
            )

        weight = torch.empty(
            (self.sharded_num_embeddings, embedding_dim), device=device, dtype=dtype
        )

        self.weight = Parameter(weight)

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        embed = self._embedding_like(self.gang.device)

        self._copy_weight(embed)

    def _copy_weight(self, embed: StandardEmbedding) -> None:
        with torch.no_grad():
            weight_shards = embed.weight.split(self.sharded_num_embeddings, dim=0)

            self.weight.copy_(weight_shards[self.gang.rank])

    @override
    def forward(self, x: Tensor) -> Tensor:
        num_embeds = self.sharded_num_embeddings

        vocab_begin_idx, vocab_end_idx = (
            self.gang.rank * num_embeds, (self.gang.rank + 1) * num_embeds  # fmt: skip
        )

        if self.pad_idx is None:
            pad_idx = None
        elif self.pad_idx >= vocab_begin_idx and self.pad_idx < vocab_end_idx:
            pad_idx = self.pad_idx - vocab_begin_idx
        else:
            pad_idx = None

        # (N, S)
        mask = (x < vocab_begin_idx) | (x >= vocab_end_idx)

        x = x - vocab_begin_idx

        x = torch.where(mask, 0, x)

        # (N, S, E)
        x = embedding(x, self.weight, pad_idx)

        # (N, S, 1)
        mask = mask.unsqueeze(-1)

        x = torch.where(mask, 0.0, x)

        # (N, S, E)
        x = reduce(x, self.gang)

        return x

    def to_embedding(self, device: Device | None = None) -> StandardEmbedding:
        """Convert this instance to a :class:`StandardEmbedding`."""
        embed = self._embedding_like(META)

        to_empty(embed, device or self.gang.device)

        with torch.no_grad():
            weight = gather(self.weight, self.gang, dim=0)

            embed.weight.copy_(weight)

        return embed

    def _embedding_like(self, device: Device) -> StandardEmbedding:
        return StandardEmbedding(
            self.num_embeddings,
            self.embedding_dim,
            self.pad_idx,
            init_fn=self.init_fn,
            device=device,
            dtype=self.weight.dtype,
        )

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        s = f"{s}, rank={self.gang.rank}, world_size={self.gang.size}"

        if self.init_fn is not None:
            init_fn = getattr(self.init_fn, "__name__", self.init_fn)

            s = f"{s}, init_fn={init_fn}"

        return s


@final
class ShardedEmbedding(Embedding):
    """Represents a :class:`StandardEmbedding` that is sharded across its
    embedding dimension."""

    gang: Gang
    sharded_embedding_dim: int
    weight: Parameter
    init_fn: Callable[[StandardEmbedding], None] | None

    @staticmethod
    def from_embedding(embed: StandardEmbedding, gang: Gang) -> ShardedEmbedding:
        """Construct a :class:`ShardedEmbedding` by sharding ``embed``.

        :param embed:
            The embedding to shard.
        :param gang:
            The gang over which to shard ``embed``.
        """
        device = embed.weight.device

        if device != gang.device and device.type != "meta":
            raise ValueError(
                "The device of `embed` must either match `gang.device` or must be of type `meta`."
            )

        sharded = ShardedEmbedding(
            gang,
            embed.num_embeddings,
            embed.embedding_dim,
            embed.pad_idx,
            init_fn=embed.init_fn,
            device=META,
            dtype=embed.weight.dtype,
        )

        if device.type != "meta":
            to_empty(sharded, device)

        sharded._copy_weight(embed)

        return sharded

    def __init__(
        self,
        gang: Gang,
        num_embeddings: int,
        embedding_dim: int,
        pad_idx: int | None = None,
        *,
        init_fn: Callable[[StandardEmbedding], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param gang:
            The gang over which to shard the embedding table.
        :param num_embeddings:
            The size of the embedding table.
        :param embedding_dim:
            The dimensionality of returned embeddings.
        :param pad_idx:
            If not ``None``, entries at ``pad_idx`` do not contribute to the
            gradient; therefore, the embedding at ``pad_idx`` is not updated
            during training.
        :param init_fn:
            The callable to initialize the embedding table.
        """
        super().__init__(num_embeddings, embedding_dim, pad_idx)

        if embedding_dim % gang.size != 0:
            raise ValueError(
                f"`embedding_dim` must be a multiple of `gang.size` ({gang.size}), but is {embedding_dim} instead."
            )

        self.gang = gang

        self.sharded_embedding_dim = embedding_dim // gang.size

        if device is None:
            device = gang.device
        elif device != gang.device and device.type != "meta":
            raise ValueError(
                "`device` must either match `gang.device` or must be of type `meta`."
            )

        weight = torch.empty(
            (num_embeddings, self.sharded_embedding_dim), device=device, dtype=dtype
        )

        self.weight = Parameter(weight)

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        embed = self._embedding_like(self.gang.device)

        self._copy_weight(embed)

    def _copy_weight(self, embed: StandardEmbedding) -> None:
        with torch.no_grad():
            weight_shards = embed.weight.split(self.sharded_embedding_dim, dim=1)

            self.weight.copy_(weight_shards[self.gang.rank])

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = reduce_on_backward(x, self.gang)

        x = embedding(x, self.weight, self.pad_idx)

        x = gather(x, self.gang)

        return x

    def to_embedding(self, device: Device | None = None) -> StandardEmbedding:
        """Convert this instance to a :class:`StandardEmbedding`."""
        embed = self._embedding_like(META)

        to_empty(embed, device or self.gang.device)

        with torch.no_grad():
            weight = gather(self.weight, self.gang, dim=1)

            embed.weight.copy_(weight)

        return embed

    def _embedding_like(self, device: Device) -> StandardEmbedding:
        return StandardEmbedding(
            self.num_embeddings,
            self.embedding_dim,
            self.pad_idx,
            init_fn=self.init_fn,
            device=device,
            dtype=self.weight.dtype,
        )

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        s = f"{s}, rank={self.gang.rank}, world_size={self.gang.size}"

        if self.init_fn is not None:
            init_fn = getattr(self.init_fn, "__name__", self.init_fn)

            s = f"{s}, init_fn={init_fn}"

        return s


def init_scaled_embedding(embed: StandardEmbedding) -> None:
    """Initialize ``embed`` from
    :math:`\\mathcal{N}(0, \\frac{1}{\\text{embedding_dim}})`."""
    nn.init.normal_(embed.weight, std=embed.embedding_dim**-0.5)

    if embed.pad_idx is not None:
        with torch.no_grad():
            embed.weight[embed.pad_idx].fill_(0.0)
