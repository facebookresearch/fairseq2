# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Literal, final, overload

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.models.clm import CausalLM
from fairseq2.models.gemma3n.decoder import Gemma3nDecoderBase
from fairseq2.models.gemma3n.frontend import Gemma3nFrontendBase
from fairseq2.nn import BatchLayout, IncrementalStateBag, Projection
from fairseq2.nn.functional import cross_entropy


@final
class Gemma3nModel(CausalLM):
    """Gemma3n decoder-only transformer with multimodal support."""

    model_dim: int
    decoder_frontend: Gemma3nFrontendBase
    decoder: Gemma3nDecoderBase
    final_proj: Projection
    pad_idx: int | None
    audio_tower: "Module | None"

    def __init__(
        self,
        model_dim: int,
        decoder_frontend: Gemma3nFrontendBase,
        decoder: Gemma3nDecoderBase,
        final_proj: Projection,
        pad_idx: int | None,
        max_seq_len: int,
        *,
        audio_tower: "Module | None" = None,
    ) -> None:
        """
        :param model_dim: The model dimensionality.
        :param decoder_frontend: The decoder frontend.
        :param decoder: The decoder.
        :param final_proj: The projection to apply to decoder outputs.
        :param pad_idx: The index of the pad symbol in the vocabulary.
        :param max_seq_len: The maximum sequence length.
        :param audio_tower: Optional audio tower for mel → text-space encoding.
        """
        super().__init__(max_seq_len)

        self.model_dim = model_dim
        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.final_proj = final_proj
        self.pad_idx = pad_idx

        self.audio_tower: Module | None
        self.register_module("audio_tower", audio_tower)

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = ...,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: Literal[False],
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: bool = ...,
    ) -> Tensor | tuple[Tensor, Tensor]: ...

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor | None = None,
        *,
        state_bag: IncrementalStateBag | None = None,
        audio_features: Tensor | None = None,
        vision_features: Tensor | None = None,
        label_smoothing: float = 0.0,
        target_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
        return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        :param seqs: Input token IDs. *Shape:* :math:`(N,S)`.
        :param seqs_layout: Layout information.
        :param targets: Target token IDs for loss computation.
        :param state_bag: Incremental decoding state.
        :param audio_features: Log-mel spectrogram. *Shape:* :math:`(N,T,128)`.
            Requires Gemma3n-specific preprocessing (HTK preemphasis, 32ms
            frames, 10ms hop, overdrive FFT). Use HuggingFace's
            ``Gemma3nAudioFeatureExtractor`` rather than fairseq2's built-in
            audio pipeline, which uses incompatible defaults.
        :param vision_features: Image pixels. *Shape:* :math:`(N,C,H,W)`.
        :param label_smoothing: Label smoothing factor.
        :param target_mask: Mask for targets.
        :param reduction: Loss reduction method.
        :param return_logits: If True, return both loss and logits.
        :returns: Logits or loss (or both if return_logits=True).
        """
        # Encode audio through tower before frontend
        audio_embeds: Tensor | None = None
        if audio_features is not None and self.audio_tower is not None:
            audio_embeds = self.audio_tower(audio_features)

        seqs, seqs_layout, per_layer_embeds = self.decoder_frontend(
            seqs,
            seqs_layout,
            state_bag=state_bag,
            audio_embeds=audio_embeds,
            vision_features=vision_features,
        )

        decoder_output = self.decoder(
            seqs,
            seqs_layout,
            state_bag=state_bag,
            per_layer_embeds=per_layer_embeds,
        )

        del seqs

        if targets is None:
            return self.final_proj(decoder_output)

        if not return_logits:
            return self.compute_fused_loss(
                decoder_output,
                targets,
                label_smoothing=label_smoothing,
                target_mask=target_mask,
                reduction=reduction,
            )

        logits = self.final_proj(decoder_output)

        del decoder_output

        loss = self.compute_loss(
            logits,
            targets,
            label_smoothing=label_smoothing,
            target_mask=target_mask,
            reduction=reduction,
        )

        return loss, logits

    def compute_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        *,
        label_smoothing: float = 0.0,
        target_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
    ) -> Tensor:
        return cross_entropy(
            logits,
            targets,
            self.pad_idx,
            label_smoothing=label_smoothing,
            target_mask=target_mask,
            reduction=reduction,
        )

    def compute_fused_loss(
        self,
        decoder_output: Tensor,
        targets: Tensor,
        *,
        label_smoothing: float = 0.0,
        target_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
    ) -> Tensor:
        logits = self.final_proj(decoder_output)

        return cross_entropy(
            logits,
            targets,
            self.pad_idx,
            label_smoothing=label_smoothing,
            target_mask=target_mask,
            reduction=reduction,
        )

    def compile_loss(self, *args: Any, **kwargs: Any) -> None:
        self.compute_fused_loss = torch.compile(  # type: ignore[method-assign]
            self.compute_fused_loss, *args, **kwargs
        )

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"pad_idx={self.pad_idx}, "
            f"max_seq_len={self.max_seq_len}"
        )
