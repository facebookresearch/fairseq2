# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Literal, final, overload

from torch import Tensor
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.seq2seq import Seq2SeqModel
from fairseq2.models.transformer.decoder import TransformerDecoder
from fairseq2.models.transformer.encoder import TransformerEncoder
from fairseq2.models.transformer.frontend import TransformerFrontend
from fairseq2.nn import BatchLayout, IncrementalState, IncrementalStateBag, Projection
from fairseq2.nn.functional import cross_entropy


@final
class TransformerModel(Seq2SeqModel):
    """Represents a Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    def __init__(
        self,
        model_dim: int,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        pad_idx: int | None,
        max_source_seq_len: int,
        max_target_seq_len: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param encoder_frontend: The encoder frontend.
        :param encoder: The encoder.
        :param decoder_frontend: The decoder frontend.
        :param decoder: The decoder.
        :param final_proj: The projection to apply to decoder outputs.
        :param max_target_seq_len: The maximum length of produced sequences.
        """
        super().__init__(max_source_seq_len, max_target_seq_len)

        self.model_dim = model_dim
        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.final_proj = final_proj
        self.pad_idx = pad_idx

    @override
    @overload
    def forward(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = ...,
    ) -> Tensor: ...

    @override
    @overload
    def forward(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
    ) -> Tensor: ...

    @override
    @overload
    def forward(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: Literal[False],
    ) -> Tensor: ...

    @override
    @overload
    def forward(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    @override
    @overload
    def forward(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
        targets: Tensor,
        *,
        label_smoothing: float = ...,
        target_mask: Tensor | None = ...,
        reduction: Literal["sum", "mean"] = ...,
        return_logits: bool = ...,
    ) -> Tensor | tuple[Tensor, Tensor]: ...

    def forward(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
        targets: Tensor | None = None,
        *,
        state_bag: IncrementalStateBag | None = None,
        label_smoothing: float = 0.0,
        target_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
        return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if not self.training and state_bag is not None:
            state = state_bag.maybe_get_state(self, _TransformerModelState)
        else:
            state = None

        if state is None:
            source_seqs, source_seqs_layout = self.encoder_frontend(
                source_seqs, source_seqs_layout
            )

            encoder_output = self.encoder(source_seqs, source_seqs_layout)

            encoder_output_layout = source_seqs_layout

            if not self.training and state_bag is not None:
                state = _TransformerModelState(encoder_output, encoder_output_layout)

                state_bag.set_state(self, state)
        else:
            encoder_output = state.encoder_output

            encoder_output_layout = state.encoder_output_layout

        del source_seqs

        target_seqs, target_seqs_layout = self.decoder_frontend(
            target_seqs, target_seqs_layout, state_bag=state_bag
        )

        decoder_output = self.decoder(
            target_seqs,
            target_seqs_layout,
            encoder_output,
            encoder_output_layout,
            state_bag=state_bag,
        )

        del target_seqs

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

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"pad_idx={self.pad_idx}, "
            f"max_source_seq_len={self.max_source_seq_len}, "
            f"max_target_seq_len={self.max_target_seq_len}"
        )


class _TransformerModelState(IncrementalState):
    def __init__(
        self, encoder_output: Tensor, encoder_output_layout: BatchLayout
    ) -> None:
        self.encoder_output = encoder_output
        self.encoder_output_layout = encoder_output_layout

    @override
    def reorder(self, new_order: Tensor) -> None:
        self.encoder_output = self.encoder_output.index_select(0, new_order)

        seq_lens = self.encoder_output_layout.seq_lens

        seq_lens = [seq_lens[i] for i in new_order.tolist()]

        self.encoder_output_layout = BatchLayout.of(self.encoder_output, seq_lens)

    @override
    def size_bytes(self) -> int:
        return self.capacity_bytes()

    @override
    def capacity_bytes(self) -> int:
        return 2 * self.encoder_output.numel() * self.encoder_output.itemsize
