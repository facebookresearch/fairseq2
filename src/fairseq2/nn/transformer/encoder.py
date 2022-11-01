# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Tuple, final

import torch.nn as nn
import torch.nn.functional as F
from overrides import final as finaloverride
from overrides import override
from torch import Tensor
from torch import dtype as DataType
from torch.nn import LayerNorm, Module

from fairseq2.nn.embedding import Embedding
from fairseq2.nn.module_list import ModuleList
from fairseq2.nn.positional_embedding import PositionalEmbedding
from fairseq2.nn.projection import Projection, ResettableProjection
from fairseq2.nn.transformer.attention_mask import AttentionMaskGenerator
from fairseq2.nn.transformer.encoder_layer import TransformerEncoderLayer
from fairseq2.nn.transformer.norm_order import TransformerNormOrder
from fairseq2.nn.utils import to_float_mask


class TransformerEncoder(Module, ABC):
    """Represents a Transformer encoder."""

    model_dim: int
    """The dimensionality of the model (i.e. inputs and outputs)."""

    batch_first: bool
    """If ``True``, the first dimension of batched inputs and outputs represents
    the batch; otherwise, the sequence."""

    def __init__(self, model_dim: int, batch_first: bool) -> None:
        """
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
        :param batch_first:
            If ``True``, the first dimension of batched inputs and outputs
            represents the batch; otherwise, the sequence.
        """
        super().__init__()

        self.model_dim = model_dim

        self.batch_first = batch_first

    @abstractmethod
    def forward(self, seq: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param seq:
            The source sequences. *Shape:* :math:`(S)` when unbatched,
            :math:`(N,S)` when :attr:`batch_first` is ``True``, or :math:`(S,N)`
            when :attr:`batch_first` is ``False``, where :math:`N` is the batch
            size and :math:`S` is the source sequence length.

        :returns:
            - The output. *Shape:* :math:`(S,M)` when unbatched,
              :math:`(N,S,M)` when :attr:`batch_first` is ``True``, or
              :math:`(S,N,M)` when :attr:`batch_first` is ``False``, where
              :math:`N` is the batch size, :math:`S` is the source sequence
              length, and :math:`M` is the model size.
            - The key padding mask used by the self attention. *Shape:*
              :math:`(S)` when unbatched, :math:`(N,S)` when :attr:`batch_first`
              is ``True``, or :math:`(S,N)` when :attr:`batch_first` is
              ``False``, where :math:`N` is the batch size and :math:`S` is the
              source sequence length.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


class InternalDimProjection(ResettableProjection):
    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        device: Any,
        dtype: Optional[DataType],
    ) -> None:
        super().__init__(
            inp_dim,
            out_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

    @override
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)


@final
class StandardTransformerEncoder(TransformerEncoder):
    """Represents a Transformer encoder layer as described in
    :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`."""

    embed: Embedding
    embed_scale: float
    positional_embed: Optional[PositionalEmbedding]
    embed_norm: Optional[LayerNorm]
    embed_dropout_p: float
    inp_dim_proj: Optional[Projection]
    self_attn_mask_gen: Optional[AttentionMaskGenerator]
    layers: ModuleList
    layer_norm: Optional[LayerNorm]
    out_dim_proj: Optional[Projection]

    def __init__(
        self,
        embed: Embedding,
        positional_embed: Optional[PositionalEmbedding],
        layers: Iterable[TransformerEncoderLayer],
        *,
        no_scale_embed: bool = False,
        norm_embed: bool = False,
        embed_dropout_p: float = 0.1,
        self_attn_mask_gen: Optional[AttentionMaskGenerator] = None,
        layer_drop_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        norm_eps: float = 1e-5,
        device: Any = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param embed:
            The input embedding dictionary.
        :param positional_embed:
            The positional embedding dictionary.
        :param layers:
            The encoder layers.
        :param no_scale_embed:
            If ``True``, input embeddings won't be scaled by the square root of
            the embedding size.
        :param norm_embed:
            If ``True``, applies Layer Normalization to the sum of input and
            positional embeddings.
        :param embed_dropout_p:
            The dropout probability on input embeddings.
        :param self_attn_mask_gen:
            The attention mask generator. If ``None``, no mask will be used.
        :param layer_drop_p:
            If greater than zero, applies LayerDrop to the encoder layers as
            described in :cite:t:`DBLP:journals/corr/abs-1909-11556`.
        :param norm_order:
            The Layer Normalization order to use.
        :param norm_eps:
            The epsilon value to add to the denominator of the
            :class:`~torch.nn.LayerNorm` modules for numerical stability.
        """
        fct_kwargs: Dict = {"device": device, "dtype": dtype}

        embedding_dim = embed.embedding_dim

        layer_list = ModuleList(layers, layer_drop_p)
        if not layer_list:
            raise ValueError("`layers` must contain at least one encoder layer.")

        model_dim, batch_first = layer_list[0].model_dim, layer_list[0].batch_first

        for idx, layer in enumerate(layers):
            if layer.model_dim != model_dim:
                raise ValueError(
                    f"`model_dim` of the encoder layer {idx} ({layer.model_dim}) does not match `model_dim` ({model_dim})."
                )

            if layer.batch_first != batch_first:
                raise ValueError(
                    f"`batch_first` of the encoder layer {idx} ({layer.batch_first}) does not match `batch_first` ({batch_first})."
                )

        super().__init__(model_dim, batch_first)

        self.embed = embed

        self.embed_scale = 1.0 if no_scale_embed else math.sqrt(embedding_dim)

        if positional_embed is not None:
            if positional_embed.embedding_dim != embedding_dim:
                raise ValueError(
                    f"`embedding_dim` of `positional_embed` ({positional_embed.embedding_dim}) does not match `embedding_dim` of `embed` ({embedding_dim})."
                )

            self.positional_embed = positional_embed
        else:
            self.register_module("positional_embed", None)

        if norm_embed:
            self.embed_norm = LayerNorm(embedding_dim, norm_eps, **fct_kwargs)
        else:
            self.register_module("embed_norm", None)

        self.embed_dropout_p = embed_dropout_p

        if embedding_dim != model_dim:
            self.inp_dim_proj = InternalDimProjection(
                embedding_dim, model_dim, **fct_kwargs
            )
        else:
            self.register_module("inp_dim_proj", None)

        self.self_attn_mask_gen = self_attn_mask_gen

        self.layers = layer_list

        if norm_order != TransformerNormOrder.POST:
            self.layer_norm = LayerNorm(model_dim, norm_eps, **fct_kwargs)
        else:
            self.register_module("layer_norm", None)

        if embedding_dim != model_dim:
            self.out_dim_proj = InternalDimProjection(
                model_dim, embedding_dim, **fct_kwargs
            )
        else:
            self.register_module("out_dim_proj", None)

    @finaloverride
    def forward(self, seq: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        self_attn_padding_mask = self._get_self_attn_padding_mask(seq)

        x = self._forward_embed(seq)

        if self.inp_dim_proj is not None:
            x = self.inp_dim_proj(x)

        x = self._forward_encoder_layers(x, self_attn_padding_mask)

        if self.out_dim_proj is not None:
            x = self.out_dim_proj(x)

        return x, self_attn_padding_mask

    def _get_self_attn_padding_mask(self, seq: Tensor) -> Optional[Tensor]:
        if self.embed.padding_idx is not None:
            mask = seq.eq(self.embed.padding_idx)

            # Applying a reduction (i.e. `any()`) and returning `None` if the
            # mask does not contain any padding sounds like a smart idea, but
            # doing so causes a device-to-host transfer which costs more time
            # than applying the mask in a fused op (i.e. `baddbmm`).

            return to_float_mask(mask, dtype=self.embed.weight.dtype)
        else:
            return None

    def _forward_embed(self, seq: Tensor) -> Tensor:
        x = self.embed(seq)

        if self.embed_scale != 1.0:
            x = x * self.embed_scale

        # TODO: quant noise?

        if self.positional_embed is not None:
            x = x + self.positional_embed(seq)

        if self.embed_norm is not None:
            x = self.embed_norm(x)

        if self.embed_dropout_p > 0.0:
            x = F.dropout(x, self.embed_dropout_p, self.training)

        return x  # type: ignore[no-any-return]

    def _forward_encoder_layers(
        self, x: Tensor, self_attn_padding_mask: Optional[Tensor]
    ) -> Tensor:
        self_attn_mask: Optional[Tensor] = None

        if self.self_attn_mask_gen is not None:
            self_attn_mask = self.self_attn_mask_gen(x, self.batch_first)

        for layer in self.layers:
            x = layer(x, self_attn_mask, self_attn_padding_mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x
