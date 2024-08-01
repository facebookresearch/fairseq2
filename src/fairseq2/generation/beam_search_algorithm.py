# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence, final

import torch
from torch import Tensor

from fairseq2.factory_registry import ConfigBoundFactoryRegistry


class BeamSearchAlgorithm(Protocol):
    """Represents a beam search algorithm."""

    def __call__(self, beam_size: int, lprobs: Tensor, step_scores: Tensor) -> BeamStep:
        """Take a single step.

        A subclass implementation is expected to return the best 2 x `beam_size`
        candidates. The sequence generator will choose the first `beam_size` of
        these which don't predict EOS to continue with.

        :param beam_size:
            The beam size.
        :param lprobs:
            The next-step log probability of each vocabulary entry. *Shape:*
            :math:`(N,V)`, where :math:`N` is the batch size and :math:`V` is
            the size of the vocabulary.
        :param step_scores:
            The cumulative score of each step in the beam. *Shape:* :math:`(N,S)`,
            where :math:`N` is the batch size and :math:`S` is the length of the
            beam.
        """


@final
@dataclass
class BeamStep:
    """Represents the output of a beam search algorithm."""

    seq_indices: Tensor
    """The beam sequence indices. *Shape:* :math:`(B)`, where :math:`B` is the
    beam size."""

    vocab_indices: Tensor
    """The vocabulary indices. *Shape:* Same as ``seq_indices``."""

    scores: Tensor
    """The scores. *Shape:* Same as ``seq_indices``."""

    def masked_select(self, mask: Tensor) -> BeamStep:
        """Reduce the beam to the sequences included in ``mask``."""
        seq_indices = self.seq_indices.masked_select(mask)

        vocab_indices = self.vocab_indices.masked_select(mask)

        scores = self.scores.masked_select(mask)

        return BeamStep(seq_indices, vocab_indices, scores)

    def first(self, count: int) -> BeamStep:
        """Slice the beam to the first ``count`` sequences."""
        seq_indices = self.seq_indices[:count]

        vocab_indices = self.vocab_indices[:count]

        scores = self.scores[:count]

        return BeamStep(seq_indices, vocab_indices, scores)

    @staticmethod
    def merge(steps: Sequence[BeamStep]) -> BeamStep:
        """Merge ``steps`` into a single beam."""
        seq_indices = torch.cat([s.seq_indices for s in steps])

        vocab_indices = torch.cat([s.vocab_indices for s in steps])

        scores = torch.cat([s.scores for s in steps])

        return BeamStep(seq_indices, vocab_indices, scores)


# TODO: Remove once Python 3.9 support is dropped.
# typing_extensions in Python<=3.9 has a bug that causes this line to fail with:
#    Parameters to generic types must be types. Got []
# See https://stackoverflow.com/questions/73974069/generic-paramspec-on-python-3-9.
if TYPE_CHECKING:
    beam_search_factories = ConfigBoundFactoryRegistry[[], BeamSearchAlgorithm]()
else:
    beam_search_factories = ConfigBoundFactoryRegistry()


@final
class StandardBeamSearchAlgorithm(BeamSearchAlgorithm):
    """Represents a standard beam search algoritm."""

    def __call__(self, beam_size: int, lprobs: Tensor, step_scores: Tensor) -> BeamStep:
        vocab_size = lprobs.size(1)

        # Make the probabilities contain cumulative scores for each hypothesis.
        # (N, V) + (N, 1) = (N, V)
        lprobs = lprobs + step_scores[:, -1].unsqueeze(-1)

        # (N, V) -> (N x V)
        lprobs = lprobs.view(-1)

        # (2 x B)
        top_scores, top_indices = torch.topk(lprobs, k=min(2 * beam_size, vocab_size))

        return BeamStep(top_indices // vocab_size, top_indices % vocab_size, top_scores)


@dataclass
class StandardBeamSearchConfig:
    """Holds the configuration of a :class:`StandardBeamSearchConfig`."""


beam_search_factories.register(
    "standard", lambda _: StandardBeamSearchAlgorithm(), StandardBeamSearchConfig
)
