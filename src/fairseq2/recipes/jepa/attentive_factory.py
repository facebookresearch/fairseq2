# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Sequence, final

from torch import Tensor
from torch.nn import Module, ModuleList

from fairseq2.models.jepa.factory import JepaEncoderConfig
from fairseq2.models.jepa.model import JepaModel
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.transformer.factory import TransformerConfig
from fairseq2.nn.projection import Linear
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.transformer.encoder import TransformerEncoder
from fairseq2.nn.transformer.encoder_layer import StandardTransformerEncoderLayer
from fairseq2.nn.transformer.layer_norm import (
    LayerNormFactory,
    make_standard_layer_norm,
)
from fairseq2.typing import Device, DataType


@dataclass(kw_only=True)
class AttentivePoolerConfig:
    """Holds the configuration for attentive classifier
    
    The default value is from the AttentiveClassifier
    (https://github.com/facebookresearch/jepa/blob/main/src/models/attentive_pooler.py)
    """

    model_dim: int = 768
    """Embedding dimension"""

    depth: int = 1
    """The depth of attention layers. The first one is a thin cross-attention module and all
    others are standard multi-head attention"""
    
    num_queries: int = 1
    """Number of queries for the cross attention layer"""

    num_encoder_attn_heads: int = 12
    """The number of attention heads in encoder layers."""

    attn_dropout_p: float = 0.0
    """The dropout probability on attention weights."""

    ffn_inner_dim_ratio: float = 4.0
    """
    The ratio of the dimensionality of the inner projection layers in
    feed-forward networks to :attr:`embed_dim`.
    """

    dropout_p: float = 0.0
    """The dropout probability on outputs of Transformer layers."""
    
    init_std: float = 0.02
    """std to initialize the weights and bias for linear and LayerNorm layers"""


@final
class AttentivePoolerBuilder:
    """Build an attentive pooler used for attentive probing evaluation"""
    
    _config: AttentivePoolerConfig
    _device: Device | None
    _dtype: DataType | None
    
    def __init__(
        self,
        config: AttentivePoolerConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        self._config = config

        self._device, self._dtype = device, dtype
        
    def build_model(self) -> AttentivePooler:
        config = self._config
        
        cross_attn = self.build_cross_attention()
        
        if config.depth > 1:
            attn_layers = [self.build_encoder_layer(i) for i in range(config.depth - 1)]
        else:
            attn_layers = None
        
        return AttentivePooler(
            cross_attn,
            attn_layers,
            num_queries=config.num_queries,
            layer_norm_factory=self.build_layer_norm,
            device=self._device,
            dtype=self._dtype,
        )


@final
class AttentivePooler(TransformerEncoder):
    def __init__(
        self,
        config: AttentivePoolerConfig,
        *,
        depth: int = 1,
        dropout_p: float = 0.0,
        layer_norm_factory: LayerNormFactory | None = None,
        layernorm_init_fn: Callable[[LayerNorm], None] | None = None,
        proj_init_fn: Callable[[Linear], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param layers:
            The encoder layers.
        :param dropout_p:
            The dropout probability on encoder outputs.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization module.
        """
        model_dim = layer_list[0].model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = make_standard_layer_norm
        
        layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)
    
        cross_attn_layer = StandardTransformerEncoderLayer(
            model_dim, bias=True, eps=1e-6, init_fn=layernorm_init_fn, device=device, dtype=dtype
        )


        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

        self.norm_order = norm_order

    @override
    def forward(
        self, seqs: Tensor, padding_mask: PaddingMask | None
    ) -> tuple[Tensor, PaddingMask | None]:
        if self._layer_output_hooks and self.layer_drop_p > 0.0 and self.training:
            raise InvalidOperationError(
                "The layer output hooks cannot be run when LayerDrop is enabled."
            )

        num_layers = len(self.layers)

        if self.self_attn_mask_factory is None:
            self_attn_mask = None
        else:
            self_attn_mask = self.self_attn_mask_factory(
                seqs, keys=seqs, training=self.training
            )

        for layer_idx, (layer, drop) in enumerate(self._drop_iter()):
            layer_output, layer_padding_mask = layer(seqs, padding_mask, self_attn_mask)

            if drop:
                seqs = _record_drop_for_backward(seqs, layer_output)

                continue

            seqs, padding_mask = layer_output, layer_padding_mask

            for hook in self._layer_output_hooks.values():
                if not hook(layer_idx, seqs, padding_mask, num_layers):
                    break

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs, padding_mask

    def _drop_iter(self) -> Iterator[tuple[Module, bool]]:
        if self.training and self.layer_drop_p > 0.0:
            prob_dist = torch.rand(
                len(self.layers), generator=self.generator, device=CPU
            )
        else:
            prob_dist = None

        for idx, m in enumerate(self.layers):
            drop = prob_dist is not None and float(prob_dist[idx]) <= self.layer_drop_p

            yield m, drop

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.self_attn_mask_factory is not None:
            self_attn_mask_factory = getattr(
                self.self_attn_mask_factory, "__name__", self.self_attn_mask_factory
            )

            s = f"{s}, self_attn_mask_factory={self_attn_mask_factory}"

        if self.layer_drop_p > 0.0:
            s = f"{s}, layer_drop_p={self.layer_drop_p:G}"

        return f"{s}, norm_order={self.norm_order.name}"


@final
class JepaForClassification(Module):
    """
    Represents a pretrained Jepa model, with an attentive probing layer for
    classfication tasks. See 
        * :cite:t:`https://doi.org/10.48550/arXiv.2301.08243`
        * :cite:t:`https://doi.org/10.48550/arXiv.2404.08471`
    """
    jepa: JepaModel
    attentive_pooler: AttentivePooler
    head: Linear
    
    def __init__(
        self,
        jepa: JepaModel,
        attentive_pooler: TransformerEncoder,
        head: Linear,
    ) -> None:
        super().__init__()
        
        self.model_dim = jepa.model_dim
        
        self.jepa = jepa
        self.attentive_pooler = attentive_pooler
        self.head = head
                
        # TODO: Move to builder
        # normalize_truncate(self.query_tokens, std=init_std)
    
    def forward(self, batch: SequenceBatch) -> Tensor:
        seqs = self.jepa(batch)
        seqs = self.attentive_pooler(seqs)
        output = self.head(self)
        return output
        

@dataclass(kw_only=True)
class JepaProbeConfig:
    """
    Holds the configuration of a probing model
    
    TODO: Move to fairseq2.models.jepa
    """
    
    encoder_config: JepaEncoderConfig = field(
        default_factory=lambda: JepaEncoderConfig()
    )
    """The configuration of the Vision Transformer encoder."""
    
    attentive_config: TransformerConfig = field(
        default_factory=lambda: TransformerConfig()
    )