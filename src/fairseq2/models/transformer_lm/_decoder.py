# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from typing import Protocol, final

from torch import Generator, Tensor
from torch.nn import Dropout
from torch.utils.hooks import RemovableHandle
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.transformer import (
    AttentionBiasCache,
    LayerNormFactory,
    TransformerNormOrder,
    create_standard_layer_norm,
)
from fairseq2.nn import BatchLayout, IncrementalStateBag, LayerNorm, LayerStack

# isort: split

from fairseq2.models.transformer_lm._decoder_layer import TransformerLMDecoderLayer


class TransformerLMDecoder(LayerStack, ABC):
    """Represents a Transformer-based language model decoder."""

    model_dim: int

    _layer_hooks: dict[int, TransformerLMDecoderLayerHook]

    def __init__(
        self, model_dim: int, layers: Sequence[TransformerLMDecoderLayer]
    ) -> None:
        """
        :param model_dim: The dimensionality of the model.
        """
        super().__init__(layers)

        self.model_dim = model_dim

        self._layer_hooks = OrderedDict()

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """
        :param seqs: The sequences to decode. *Shape:* :math:`(N,S,M)`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`M` is the dimensionality of the model.
        :param state_bag: The state bag to use for incremental decoding.

        :returns: The decoder output. *Shape:* Same as ``seqs``.
        """

    def register_layer_hook(
        self, hook: TransformerLMDecoderLayerHook
    ) -> RemovableHandle:
        """
        Registers a layer hook on the module.

        The hook will be called every time after a layer in the decoder stack
        has computed an output.

        :param hook: The hook to register.

        :returns: A handle that can be used to remove the added hook by calling
            ``handle.remove()``.
        """
        handle = RemovableHandle(self._layer_hooks)

        self._layer_hooks[handle.id] = hook

        return handle

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


class TransformerLMDecoderLayerHook(Protocol):
    """
    Represents a hook to pass to :meth:`~TransformerLMDecoder.register_layer_hook`.
    """

    def __call__(
        self,
        layer_idx: int,
        layer_output: Tensor,
        layer_output_layout: BatchLayout,
        num_layers: int,
    ) -> bool:
        """
        :param layer_idx: The index of the layer in the decoder stack.
        :param layer_output: The decoded output of the layer.
        :param num_layers: The number of layers in the decoder stack.

        :returns: ``True`` if the decoder should continue executing the
            remaining layers in the stack; ``False`` if the decoder should treat
            this layer as the final layer in the stack.
        """


@final
class StandardTransformerLMDecoder(TransformerLMDecoder):
    layer_drop_p: float
    generator: Generator | None
    layer_norm: LayerNorm | None
    dropout_p: float
    norm_order: TransformerNormOrder

    def __init__(
        self,
        layers: Sequence[TransformerLMDecoderLayer],
        *,
        generator: Generator | None = None,
        dropout_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        layer_norm_factory: LayerNormFactory | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param layers: The decoder layers.
        :param generator: The random number generator for LayerDrop
            probabilities.
        :param dropout_p: The dropout probability on decoder outputs.
        :param norm_order: The Layer Normalization order.
        :param layer_norm_factory: The factory to construct the Layer
            Normalization module.
        """
        if not layers:
            raise ValueError("`layers` must be non-empty.")

        model_dim = layers[0].model_dim

        super().__init__(model_dim, layers)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self.generator = generator

        if norm_order != TransformerNormOrder.POST:
            self.layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

        self.norm_order = norm_order

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        attn_bias_cache = AttentionBiasCache()

        num_layers = len(self.layers)

        for layer_idx, layer in enumerate(self.layers):
            seqs = layer(seqs, seqs_layout, attn_bias_cache, state_bag=state_bag)

            for hook in self._layer_hooks.values():
                if not hook(layer_idx, seqs, seqs_layout, num_layers):
                    break

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, norm_order={self.norm_order.name}"
