# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, final

from torch import Tensor
from torch.nn import Dropout, Module, ModuleList
from torch.utils.hooks import RemovableHandle
from typing_extensions import override

from fairseq2.models.transformer import AttentionBiasCache
from fairseq2.models.transformer_lm.decoder_layer import TransformerLMDecoderLayer
from fairseq2.nn import BatchLayout, IncrementalStateBag, LayerNorm


class TransformerLMDecoder(Module):
    """Represents a decoder-only Transformer decoder."""

    layers: ModuleList

    def __init__(self) -> None:
        super().__init__()

        self._layer_hooks: dict[int, TransformerLMDecoderLayerHook] = OrderedDict()

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

    if TYPE_CHECKING:
        __call__ = forward

    @abstractmethod
    def compile_layerwise(self, *args: Any, **kwargs: Any) -> None: ...

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
    def __init__(
        self,
        layers: Sequence[TransformerLMDecoderLayer],
        layer_norm: LayerNorm | None = None,
        *,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        self.layers = ModuleList(layers)

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

    @override
    def compile_layerwise(self, *args: Any, **kwargs: Any) -> None:
        for layer in self.layers:
            layer.compile(*args, **kwargs)

        if self.layer_norm is not None:
            self.layer_norm.compile(*args, **kwargs)
