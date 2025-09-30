# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Protocol, final

import torch
from torch import Generator, Tensor
from torch.nn import Dropout, Module, ModuleList
from torch.utils.hooks import RemovableHandle
from typing_extensions import override

from fairseq2.device import CPU
from fairseq2.error import InvalidOperationError
from fairseq2.models.transformer.attention_bias import AttentionBiasCache
from fairseq2.models.transformer.decoder_layer import TransformerDecoderLayer
from fairseq2.models.transformer.encoder import _record_drop_for_backward
from fairseq2.nn import BatchLayout, IncrementalStateBag, LayerNorm


class TransformerDecoder(Module):
    """Represents a Transformer decoder."""

    layers: ModuleList

    def __init__(self) -> None:
        super().__init__()

        self._layer_hooks: dict[int, TransformerDecoderLayerHook] = OrderedDict()

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        encoder_output: Tensor,
        encoder_output_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """
        :param seqs: The sequences to decode. *Shape:* :math:`(N,S,M)`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`M` is the dimensionality of the model.
        :param encoder_output: The encoder output to use in encoder-decoder
            attention. *Shape:* :math:`(N,S_{enc},M_{enc})`, where :math:`N` is
            the batch size, :math:`S_{enc}` is the encoder output sequence
            length, and :math:`M_{enc}` is the dimensionality of the encoder.
        :param state_bag: The state bag to use for incremental decoding.

        :returns: The decoder output. *Shape:* Same as ``seqs``.
        """

    if TYPE_CHECKING:
        __call__ = forward

    def register_layer_hook(self, hook: TransformerDecoderLayerHook) -> RemovableHandle:
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


class TransformerDecoderLayerHook(Protocol):
    """
    Represents a hook to pass to :meth:`~TransformerDecoder.register_layer_hook`.
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
class StandardTransformerDecoder(TransformerDecoder):
    """
    Represents a Transformer decoder as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.
    """

    def __init__(
        self,
        layers: Sequence[TransformerDecoderLayer],
        layer_norm: LayerNorm | None = None,
        *,
        layer_drop_p: float = 0.0,
        generator: Generator | None = None,
        dropout_p: float = 0.0,
    ) -> None:
        """
        :param layer_drop_p: If greater than zero, applies LayerDrop to the
            decoder layers as described in
            :cite:t:`https://doi.org/10.48550/arxiv.1909.11556`.
        :param generator: The random number generator for LayerDrop.
        """
        super().__init__()

        self.layers = ModuleList(layers)

        self.layer_drop_p = layer_drop_p

        self.generator = generator

        self.layer_norm: LayerNorm | None

        self.register_module("layer_norm", layer_norm)

        if dropout_p > 0.0:
            dropout = Dropout(dropout_p)
        else:
            dropout = None

        self.dropout: Dropout | None

        self.register_module("dropout", dropout)

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        encoder_output: Tensor,
        encoder_output_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        if self._layer_hooks:
            if self.training and self.layer_drop_p > 0.0:
                raise InvalidOperationError(
                    "The layer output hooks cannot be run when LayerDrop is enabled."
                )

        attn_bias_cache = AttentionBiasCache()

        num_layers = len(self.layers)

        for layer_idx, (layer, drop) in enumerate(self._drop_iter()):
            layer_output = layer(
                seqs,
                seqs_layout,
                encoder_output,
                encoder_output_layout,
                attn_bias_cache,
                state_bag=state_bag,
            )

            if drop:
                seqs = _record_drop_for_backward(seqs, layer_output)

                continue

            seqs = layer_output

            for hook in self._layer_hooks.values():
                if not hook(layer_idx, seqs, seqs_layout, num_layers):
                    break

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs

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

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        if self.layer_drop_p > 0.0:
            return f"layer_drop_p={self.layer_drop_p:G}"

        return ""
