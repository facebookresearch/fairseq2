# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Iterable, Optional, Protocol, Tuple, final

from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.module_list import ModuleList
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer.attention_mask import (
    AttentionMaskFactory,
    CausalAttentionMaskFactory,
)
from fairseq2.nn.transformer.decoder_layer import TransformerDecoderLayer
from fairseq2.nn.transformer.layer_norm import (
    LayerNormFactory,
    create_standard_layer_norm,
)
from fairseq2.nn.transformer.norm_order import TransformerNormOrder
from fairseq2.typing import DataType, Device, finaloverride


class TransformerDecoder(Module, ABC):
    """Represents a Transformer decoder."""

    model_dim: int
    layers: ModuleList

    _layer_output_hooks: Dict[int, DecoderLayerOutputHook]

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

        self._layer_output_hooks = OrderedDict()

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Optional[Tensor] = None,
        encoder_padding_mask: Optional[PaddingMask] = None,
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """
        :param seqs:
            The sequences to decode. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param encoder_output:
            The encoder output to use in encoder-decoder attention. *Shape:*
            :math:`(N,S_{enc},M_{enc})`, where :math:`N` is the batch size,
            :math:`S_{enc}` is the encoder output sequence length, and
            :math:`M_{enc}` is the dimensionality of the encoder.
        :param encoder_padding_mask:
            The padding mask of ``encoder_output``. *Shape:* :math:`(N,S_{enc})`,
            where :math:`N` is the batch size and :math:`S_{enc}` is the encoder
            output sequence length.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            - The decoder output. *Shape:* Same as ``seqs``.
            - The padding mask of the decoder output. *Shape:* Same as
              ``padding_mask``.
        """

    def register_layer_output_hook(
        self, hook: DecoderLayerOutputHook
    ) -> RemovableHandle:
        """Register a layer output hook on the module.

        The hook will be called every time after a layer in the decoder stack
        has computed an output.

        :param hook:
            The hook to register.

        :returns:
            A handle that can be used to remove the added hook by calling
            ``handle.remove()``.
        """
        handle = RemovableHandle(self._layer_output_hooks)

        self._layer_output_hooks[handle.id] = hook

        return handle

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


class DecoderLayerOutputHook(Protocol):
    """Represents a hook to pass to :meth:`~TransformerDecoder.forward`."""

    def __call__(
        self,
        layer_idx: int,
        layer_output: Tensor,
        layer_padding_mask: Optional[PaddingMask],
        num_layers: int,
    ) -> bool:
        """
        :param layer_idx:
            The index of the layer in the decoder stack.
        :param layer_output:
            The decoded output of the layer.
        :param layer_padding_mask:
            The padding mask of ``layer_output``.
        :param num_layers:
            The number of layers in the decoder stack.

        :returns:
            ``True`` if the decoder should continue executing the remaining
            layers in the stack; ``False`` if the decoder should treat this
            layer as the final layer in the stack.
        """


@final
class StandardTransformerDecoder(TransformerDecoder):
    """Represents a Transformer decoder as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    self_attn_mask_factory: Optional[AttentionMaskFactory]
    layer_norm: Optional[LayerNorm]
    norm_order: TransformerNormOrder

    def __init__(
        self,
        layers: Iterable[TransformerDecoderLayer],
        *,
        self_attn_mask_factory: Optional[AttentionMaskFactory] = None,
        use_causal_attn_mask: bool = True,
        layer_drop_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param layers:
            The decoder layers.
        :param self_attn_mask_factory:
            The self attention mask factory.
        :param use_causal_attn_mask:
            If ``True``, passes a full :class:`CausalAttentionMask` to the
            decoder layers; otherwise, passes ``None``. Ignored if
            ``self_attn_mask_factory`` is specified.
        :param layer_drop_p:
            If greater than zero, applies LayerDrop to the decoder layers as
            described in :cite:t:`https://doi.org/10.48550/arxiv.1909.11556`.
        :param norm_order:
            The Layer Normalization order.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization module.
        """
        layer_list = ModuleList(layers, drop_p=layer_drop_p)
        if not layer_list:
            raise ValueError("`layers` must be non-empty.")

        model_dim = layer_list[0].model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        if self_attn_mask_factory is not None:
            self.self_attn_mask_factory = self_attn_mask_factory
        elif use_causal_attn_mask:
            self.self_attn_mask_factory = CausalAttentionMaskFactory()
        else:
            self.self_attn_mask_factory = None

        self.layers = layer_list

        if norm_order != TransformerNormOrder.POST:
            self.layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        self.norm_order = norm_order

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Optional[Tensor] = None,
        encoder_padding_mask: Optional[PaddingMask] = None,
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        if self._layer_output_hooks and self.layers.drop_p > 0.0:
            raise RuntimeError(
                "The layer output hooks cannot be run when LayerDrop is enabled."
            )

        num_layers = len(self.layers)

        if self.self_attn_mask_factory is None:
            self_attn_mask = None
        else:
            self_attn_mask = self.self_attn_mask_factory(
                seqs, keys=seqs, training=self.training, state_bag=state_bag
            )

        for layer_idx, layer in enumerate(self.layers.drop_iter()):
            seqs, padding_mask = layer(
                seqs,
                padding_mask,
                self_attn_mask,
                encoder_output,
                encoder_padding_mask,
                state_bag=state_bag,
            )

            for hook in self._layer_output_hooks.values():
                if not hook(layer_idx, seqs, padding_mask, num_layers):
                    break

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        return seqs, padding_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.self_attn_mask_factory is not None:
            self_attn_mask_factory = getattr(
                self.self_attn_mask_factory, "__name__", self.self_attn_mask_factory
            )

            s = f"{s}, self_attn_mask_factory={self_attn_mask_factory}"

        return f"{s}, norm_order={self.norm_order}"
