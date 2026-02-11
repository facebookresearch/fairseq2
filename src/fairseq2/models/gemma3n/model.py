# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from typing import Literal, final

from torch import Tensor
from typing_extensions import override

from fairseq2.models.transformer_lm.model import TransformerLM
from fairseq2.nn import BatchLayout, IncrementalStateBag, Projection
from fairseq2.models.transformer import TransformerDecoder, TransformerFrontend


@final
class Gemma3nLM(TransformerLM):
    """Gemma3n language model with state_bag management for PLE."""

    final_logit_softcapping: float | None

    def __init__(
        self,
        model_dim: int,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        pad_idx: int | None,
        max_seq_len: int,
        *,
        final_logit_softcapping: float | None = None,
    ) -> None:
        """
        :param final_logit_softcapping:
            Scaling factor for tanh softcapping on output logits. If ``None``,
            no softcapping is applied.
        """
        super().__init__(model_dim, decoder_frontend, decoder, final_proj, pad_idx, max_seq_len)

        self.final_logit_softcapping = final_logit_softcapping

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor | None = None,
        *,
        state_bag: IncrementalStateBag | None = None,
        label_smoothing: float = 0.0,
        target_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
        return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass that ensures state_bag is created for PLE.

        :param seqs: Input token IDs.
        :param seqs_layout: Batch layout information.
        :param targets: Target token IDs (for loss computation).
        :param state_bag: Incremental state bag (created if None).
        :param label_smoothing: Label smoothing factor for loss.
        :param target_mask: Mask for targets.
        :param reduction: Loss reduction method.
        :param return_logits: If True, return both loss and logits.
        :returns: Logits or loss (or both if return_logits=True).
        """
        # Create state_bag if None (required for PLE)
        if state_bag is None:
            state_bag = IncrementalStateBag(max_num_steps=seqs.size(1))

        # Frontend embeds tokens and computes PLE, stores in state_bag
        seqs, seqs_layout = self.decoder_frontend(
            seqs, seqs_layout, state_bag=state_bag
        )

        # Decoder processes with 4D AltUp and retrieves PLE from state_bag
        decoder_output = self.decoder(seqs, seqs_layout, state_bag=state_bag)

        del seqs

        # Compute loss or return logits
        if targets is None:
            logits = self.final_proj(decoder_output)
            return self._apply_logit_softcapping(logits)

        if not return_logits:
            return self.compute_fused_loss(
                decoder_output,
                targets,
                label_smoothing=label_smoothing,
                target_mask=target_mask,
                reduction=reduction,
            )

        logits = self.final_proj(decoder_output)
        logits = self._apply_logit_softcapping(logits)

        del decoder_output

        loss = self.compute_loss(
            logits,
            targets,
            label_smoothing=label_smoothing,
            target_mask=target_mask,
            reduction=reduction,
        )

        return loss, logits

    def _apply_logit_softcapping(self, logits: Tensor) -> Tensor:
        """Apply tanh softcapping to logits if configured.

        :param logits: Raw logits from final projection.
        :returns: Softcapped logits if final_logit_softcapping is set, else unchanged.
        """
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping

        return logits
