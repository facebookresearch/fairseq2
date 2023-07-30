# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Tuple, final

import torch
from overrides import final as finaloverride
from torch import Tensor


class BeamSearch(ABC):
    """Represents a beam search algorithm.

    An implementation is expected to return the best 2 x beam size predictions.
    The sequence generator will choose the first beam size of these which don't
    predict EOS to continue with.
    """

    @abstractmethod
    def step(
        self, step_nr: int, is_start_step: bool, lprobs: Tensor, scores: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Take a single search step.

        :param step_nr:
            The current search step number.
        :param is_start_step:
            If ``True``, this is the initial step in the search. Note that the
            start step is not necessarily the first step in the sequence. When
            the user specifies a prefix sequence in the generation, the search
            can start from an arbitrary step.
        :probs lprobs:
            The next-step log probability of each vocabulary entry. *Shape:*
            :math:`(N,B,V)`, where :math:`N` is the batch size, :math:`B` is
            the number of beams, and :math:`V` is the size of the vocabulary.
        :probs scores:
            The cumulative beam score of each previous step in the search.
            *Shape:* :math:`(N,B,S_{prv})`, where :math:`N` is the batch size,
            :math:`B` is the number of beams, and :math:`S_{prev}` is the length
            of the generated sequence so far.

        :returns:
            - The top 2 x beam size scores. *Shape:* :math:`(N,2xB)`, where
              :math:`N` is the batch size and :math:`2xB` is twice the beam
              size.
            - The indices of the top scores within their beams. *Shape:* Same
              as the first return value.
            - The indices of the beams of the top scores within their batch.
              *Shape:* Same as the first return value.
        """


@final
class StandardBeamSearch(BeamSearch):
    """Represents a standard beam search algoritm."""

    @finaloverride
    def step(
        self, step_nr: int, is_start_step: bool, lprobs: Tensor, scores: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, beam_size, vocab_size = lprobs.size()

        if is_start_step:
            # At the initial step, all hypotheses are equally likely, so we use
            # only the first beam.
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # Make probabilities contain cumulative scores for each hypothesis.
            lprobs = lprobs + scores[:, :, step_nr - 1].unsqueeze(-1)

        # Take the best 2 x `beam_size` predictions. We'll choose the first
        # `beam_size` of these which don't predict EOS to continue with.
        # (N, 2 x B)
        top_scores, top_indices = torch.topk(
            lprobs.view(batch_size, -1), k=min(2 * beam_size, vocab_size)
        )

        # Return scores, beam-relative indices, and beam indices.
        return top_scores, top_indices % vocab_size, top_indices // vocab_size
