# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import embedding
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import META_DEVICE, Device
from fairseq2.gang import Gang
from fairseq2.nn.sharded import Sharded
from fairseq2.nn.utils.module import get_name_or_self, to_empty
from fairseq2.ops.tensor_parallel import gather, reduce, reduce_on_backward


class Embedding(Module, ABC):
    """Stores embeddings of a fixed dictionary and size."""

    def __init__(
        self, num_embeddings: int, embed_dim: int, pad_idx: int | None
    ) -> None:
        """
        If ``pad_idx`` is provided, the embedding at the specified index won't
        contribute to the gradient and therefore won't be updated during training.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Returns the embeddings corresponding to the specified indices.

        ``x`` can have any shape.

        The return value will be of shape :math:`(*,E)`, where :math:`*` is the
        input shape and :math:`E` is the dimensionality of the embeddings.
        """

    if TYPE_CHECKING:
        __call__ = forward


@final
class StandardEmbedding(Embedding):
    """Represents the standard implementation of :class:`Embedding`."""

    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        pad_idx: int | None = None,
        *,
        init_fn: Callable[[StandardEmbedding], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        If ``init_fn`` is provided, it will be used to initialize the embedding
        table in :meth:`reset_parameters`.
        """
        super().__init__(num_embeddings, embed_dim, pad_idx)

        self.weight = Parameter(
            torch.empty((num_embeddings, embed_dim), device=device, dtype=dtype)
        )

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_fn is not None:
            self.init_fn(self)
        else:
            nn.init.normal_(self.weight)

        if self.pad_idx is not None:
            with torch.no_grad():
                self.weight[self.pad_idx].fill_(0.0)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return embedding(x, self.weight, self.pad_idx)

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        s = f"num_embeddings={self.num_embeddings}, embed_dim={self.embed_dim}"

        if self.pad_idx is not None:
            s = f"{s}, pad_idx={self.pad_idx}"

        if self.init_fn is not None:
            init_fn = get_name_or_self(self.init_fn)

            s = f"{s}, init_fn={init_fn}"

        return s


@final
class VocabShardedEmbedding(Embedding, Sharded):
    """
    Represents an :class:`Embedding` sharded across its vocabulary dimension.
    """

    @staticmethod
    def from_embedding(embed: StandardEmbedding, gang: Gang) -> VocabShardedEmbedding:
        """
        Creates a :class:`VocabShardedEmbedding` by sharding ``embed`` over its
        vocabulary dimension using ``gang``.
        """
        device = embed.weight.device

        if device != gang.device and device.type != "meta":
            raise ValueError(
                f"Device of `embed` must match `gang.device` or must be of type `meta`, but is `{device}` instead."
            )

        sharded_embed = VocabShardedEmbedding(
            gang,
            embed.num_embeddings,
            embed.embed_dim,
            embed.pad_idx,
            init_fn=embed.init_fn,
            device=META_DEVICE,
            dtype=embed.weight.dtype,
        )

        if device.type != "meta":
            to_empty(sharded_embed, device)

        sharded_embed._copy_weight(embed)

        return sharded_embed

    def __init__(
        self,
        gang: Gang,
        num_embeddings: int,
        embed_dim: int,
        pad_idx: int | None = None,
        *,
        init_fn: Callable[[StandardEmbedding], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__(num_embeddings, embed_dim, pad_idx)

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
                f"`device` must match `gang.device` or must be of type `meta`, but is `{device}` instead."
            )

        weight = torch.empty(
            (self.sharded_num_embeddings, embed_dim), device=device, dtype=dtype
        )

        self.weight = Parameter(weight)

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        embed = self._embedding_like(device=self.gang.device)

        self._copy_weight(embed)

    def _copy_weight(self, embed: StandardEmbedding) -> None:
        with torch.no_grad():
            weight_shards = embed.weight.split(self.sharded_num_embeddings, dim=0)

            weight = weight_shards[self.gang.rank]

            self.weight.copy_(weight, non_blocking=True)

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
        """Unshards this instance to a :class:`StandardEmbedding`."""
        embed = self._embedding_like(device=META_DEVICE)

        to_empty(embed, device or self.gang.device)

        with torch.no_grad():
            weight = gather(self.weight, self.gang, dim=0)

            embed.weight.copy_(weight, non_blocking=True)

        return embed

    def _embedding_like(self, device: Device) -> StandardEmbedding:
        return StandardEmbedding(
            self.num_embeddings,
            self.embed_dim,
            self.pad_idx,
            init_fn=self.init_fn,
            device=device,
            dtype=self.weight.dtype,
        )

    @override
    def get_shard_dims(self) -> list[tuple[Parameter, int]]:
        return [(self.weight, 0)]

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        s = (
            f"tp_rank={self.gang.rank}, "
            f"tp_size={self.gang.size}, "
            f"num_embeddings={self.num_embeddings}, "
            f"sharded_num_embeddings={self.sharded_num_embeddings}, "
            f"embed_dim={self.embed_dim}"
        )

        if self.pad_idx is not None:
            s = f"{s}, pad_idx={self.pad_idx}"

        if self.init_fn is not None:
            init_fn = get_name_or_self(self.init_fn)

            s = f"{s}, init_fn={init_fn}"

        return s


@final
class ShardedEmbedding(Embedding, Sharded):
    """
    Represents an :class:`Embedding` sharded across its embedding dimension.
    """

    @staticmethod
    def from_embedding(embed: StandardEmbedding, gang: Gang) -> ShardedEmbedding:
        """
        Creates a :class:`ShardedEmbedding` by sharding ``embed`` over its
        embedding dimension using ``gang``.
        """
        device = embed.weight.device

        if device != gang.device and device.type != "meta":
            raise ValueError(
                f"Device of `embed` must match `gang.device` or must be of type `meta`, but is `{device}` instead."
            )

        sharded_embed = ShardedEmbedding(
            gang,
            embed.num_embeddings,
            embed.embed_dim,
            embed.pad_idx,
            init_fn=embed.init_fn,
            device=META_DEVICE,
            dtype=embed.weight.dtype,
        )

        if device.type != "meta":
            to_empty(sharded_embed, device)

        sharded_embed._copy_weight(embed)

        return sharded_embed

    def __init__(
        self,
        gang: Gang,
        num_embeddings: int,
        embed_dim: int,
        pad_idx: int | None = None,
        *,
        init_fn: Callable[[StandardEmbedding], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__(num_embeddings, embed_dim, pad_idx)

        if embed_dim % gang.size != 0:
            raise ValueError(
                f"`embed_dim` must be a multiple of `gang.size` ({gang.size}), but is {embed_dim} instead."
            )

        self.gang = gang

        self.sharded_embed_dim = embed_dim // gang.size

        if device is None:
            device = gang.device
        elif device != gang.device and device.type != "meta":
            raise ValueError(
                f"`device` must match `gang.device` or must be of type `meta`, but is `{device}` instead."
            )

        weight = torch.empty(
            (num_embeddings, self.sharded_embed_dim), device=device, dtype=dtype
        )

        self.weight = Parameter(weight)

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        embed = self._embedding_like(self.gang.device)

        self._copy_weight(embed)

    def _copy_weight(self, embed: StandardEmbedding) -> None:
        with torch.no_grad():
            weight_shards = embed.weight.split(self.sharded_embed_dim, dim=1)

            weight = weight_shards[self.gang.rank]

            self.weight.copy_(weight, non_blocking=True)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = reduce_on_backward(x, self.gang)

        x = embedding(x, self.weight, self.pad_idx)

        x = gather(x, self.gang)

        return x

    def to_embedding(self, device: Device | None = None) -> StandardEmbedding:
        """Unshards this instance to a :class:`StandardEmbedding`."""
        embed = self._embedding_like(device=META_DEVICE)

        to_empty(embed, device or self.gang.device)

        with torch.no_grad():
            weight = gather(self.weight, self.gang, dim=1)

            embed.weight.copy_(weight, non_blocking=True)

        return embed

    def _embedding_like(self, device: Device) -> StandardEmbedding:
        return StandardEmbedding(
            self.num_embeddings,
            self.embed_dim,
            self.pad_idx,
            init_fn=self.init_fn,
            device=device,
            dtype=self.weight.dtype,
        )

    @override
    def get_shard_dims(self) -> list[tuple[Parameter, int]]:
        return [(self.weight, 1)]

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        s = (
            f"tp_rank={self.gang.rank}, "
            f"tp_size={self.gang.size}, "
            f"num_embeddings={self.num_embeddings}, "
            f"embed_dim={self.embed_dim}, "
            f"sharded_embed_dim={self.sharded_embed_dim}"
        )

        if self.init_fn is not None:
            init_fn = get_name_or_self(self.init_fn)

            s = f"{s}, init_fn={init_fn}"

        return s


def init_scaled_embedding(embed: StandardEmbedding) -> None:
    """
    Initializes ``embed`` from :math:`\\mathcal{N}(0, \\frac{1}{\\text{embed_dim}})`.
    """
    nn.init.normal_(embed.weight, std=embed.embed_dim**-0.5)

    if embed.pad_idx is not None:
        with torch.no_grad():
            embed.weight[embed.pad_idx].fill_(0.0)
