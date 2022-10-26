# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from typing import Generic, NamedTuple, Optional, TypeVar

import torch
import torch.nn as nn
from overrides import overrides
from torch import Tensor

from .tokenizer import Tokenizer

SearchState = TypeVar("SearchState")


def log_prob(
    val: Tensor,
    temperature: float,
    *,
    pad: int = Tokenizer.PAD,
) -> Tensor:
    """
    Compute the log-probability of the inputs.

    Assumes [batch, value] shape.

    :param val: the values.
    :param temperature: (TODO) the temperature.
    :param pad: the id of the PAD token.
    :return: the new lprobs.
    """
    # TODO: temperature
    lprobs = nn.functional.log_softmax(val, dim=1)
    lprobs[lprobs != lprobs] = -torch.inf
    lprobs[:, pad] = -torch.inf
    return lprobs


def unk_penalty(
    lprobs: Tensor,
    penalty: float,
    *,
    unk: int = Tokenizer.UNK,
) -> Tensor:
    """
    Penalize the unknown token by `penalty`.

    Assumes [batch, value] shape.
    Makes a copy.

    :param lprobs: the probs.
    :param penalty: the penalty.
    :param unk: the unknown token offset.
    :return: the new lprobs.
    """
    lprobs = lprobs.clone()
    unk_penalty_(lprobs, penalty=penalty, unk=unk)
    return lprobs


def unk_penalty_(
    lprobs: Tensor,
    penalty: float,
    *,
    unk: int = Tokenizer.UNK,
) -> None:
    """
    Penalize the unknown token in-place.

    Assumes [batch, value] shape.

    :param lprobs: the probs.
    :param penalty: the penalty.
    :param unk: the unknown token offset.
    """
    lprobs[:, unk] -= penalty


def prevent_eos(
    lprobs: Tensor,
    *,
    eos: int = Tokenizer.EOS,
) -> Tensor:
    """
    Prevent the EOS token from being selected by forcing it to -inf.

    Assumes [batch, value] shape.
    Makes a copy.

    :param lprobs: the probs.
    :param eos: the EOS token.
    :return: the new probs.
    """
    lprobs = lprobs.clone()
    prevent_eos_(lprobs, eos=eos)
    return lprobs


def prevent_eos_(
    lprobs: Tensor,
    *,
    eos: int = Tokenizer.EOS,
) -> None:
    """
    Prevent the EOS token from being selected by forcing it to -inf in-place.

    Assumes [batch, value] shape.

    :param lprobs: the probs.
    :param eos: the EOS token.
    """
    lprobs[:, eos] = -torch.inf


def force_eos(
    lprobs: Tensor,
    *,
    eos: int = Tokenizer.EOS,
) -> Tensor:
    """
    Force all log probs except the EOS token to -inf.

    :param lprobs: the probs.
    :param eos: the EOS token.
    :return: the new probs.
    """
    lprobs = lprobs.clone()
    force_eos_(lprobs, eos=eos)
    return lprobs


def force_eos_(
    lprobs: Tensor,
    *,
    eos: int = Tokenizer.EOS,
) -> None:
    """
    Force all log probs except the EOS token to -inf.

    :param lprobs: the probs.
    :param eos: the EOS token.
    :return: the new probs.
    """
    lprobs[:, :eos] = -torch.inf
    lprobs[:, eos + 1 :] = -torch.inf


class SearchResult(NamedTuple):
    tokens: Tensor
    """(bsz x beam_size x len) generated tokens"""

    scores: Tensor
    """(bsz x beam_size) scores of each generation"""


class TopKResult(NamedTuple):
    values: Tensor
    indices: Tensor


class Search(Generic[SearchState]):
    tokenizer: Tokenizer

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def step(
        self,
        dec_out: Tensor,
        state: SearchState,
    ) -> SearchState:
        """Take a single search step.

        Args:
            step: the current search step, starting at 0
            dec_out: (bsz x input_beam_size x vocab_size)
                the model output
            state: the current search state (implementation dependent)
        Return: The new search state
        """
        ...

    def log_prob(self, dec_out: Tensor) -> Tensor:
        """Converts the decoder output and convert it to log probs of the next token.

        This is a good place to implement eg:
            - temperature sampling,
            - unk penalty,
            ...

        There are helper functions in this module to help do so.

        dec_out = batch size x vocab size
        """
        ...

    def chose_beams(self, lprobs_beam: Tensor) -> TopKResult:
        """
        lprobs_beam: (batch size x beam size x vocab_size) log prob of the full beam + next token
        """
        ...

    def prepare_state(
        self,
        src_tokens: Tensor,
        prefix_tokens: Optional[Tensor] = None,
    ) -> SearchState:
        """Prepares a search state for the search.

        Args:
            - src_tokens: input sequence.
            - prefix_tokens: forced generations for the target side.
            They need to be copied into the SearchState.
        """
        ...

    def finalize(
        self,
        state: SearchState,
        *,
        top: int = 0,
    ) -> SearchResult:
        """From the state, extract the generated sequences and their scores.

        Args:
            - state: state
            - top: restrict the result to the n-best generations per input sequence.
            Use 0 (default) to get all generations.
        """
        ...


@dataclasses.dataclass
class BeamSearchState:
    max_len: int
    tokens: Tensor
    scores: Tensor
    finished_mask: Tensor
    order: Tensor
    step: int = 0
    done: bool = False


class BeamSearch(Search[BeamSearchState]):
    def __init__(
        self,
        tokenizer: Tokenizer,
        beam_size: int = 1,
        unk_penalty: float = 1.0,
        min_len: int = 10,
        max_len: int = 256,
    ) -> None:
        super().__init__(tokenizer=tokenizer)
        assert beam_size == 1, "beam_size > 1 not implemented"
        self.beam_size = beam_size
        self.min_len = min_len
        self.max_len = max_len
        self.unk_penalty = unk_penalty
        self._step = 0

    @overrides
    def prepare_state(
        self,
        src_tokens: Tensor,
        prefix_tokens: Optional[Tensor] = None,
    ) -> BeamSearchState:
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size
        max_len = min(self.max_len, 2 * src_len + 10)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1).long()
        order = order.to(src_tokens.device)

        # initialize buffers
        # +2 for eos and pad
        scores = torch.zeros(bsz * beam_size, max_len + 2).to(src_tokens).float()
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.tokenizer.PAD)
        )
        tokens[:, 0] = self.tokenizer.BOS if prefix_tokens is None else prefix_tokens[0]

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        finished = torch.zeros(bsz, beam_size).to(src_tokens)
        return BeamSearchState(max_len, tokens, scores, finished, order, step=0)

    @overrides
    def log_prob(self, dec_out: Tensor) -> Tensor:
        lprobs = log_prob(dec_out, temperature=0.1, pad=self.tokenizer.PAD)
        unk_penalty_(lprobs, self.unk_penalty, unk=self.tokenizer.UNK)
        if self._step >= self.max_len:
            force_eos_(lprobs, eos=self.tokenizer.EOS)

        # TODO: prefix tokens
        # handle prefix tokens (possibly with different lengths)
        # if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:
        # lprobs, tokens, scores = self._prefix_tokens(step, lprobs, scores, tokens, prefix_tokens, beam_size)
        if self._step < self.min_len:
            # minimum length constraint (does not apply if using prefix_tokens)
            prevent_eos_(lprobs, eos=self.tokenizer.EOS)

        # if self.should_set_src_lengths:
        #     self.search.set_src_lengths(src_lengths)

        # if self.repeat_ngram_blocker is not None:
        #     lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

        return lprobs

    @torch.jit.export
    @overrides
    def step(
        self,
        dec_out: Tensor,
        state: BeamSearchState,
    ) -> BeamSearchState:
        # TODO allow passing attentions from the decoder
        beam_size = self.beam_size
        lprobs = self.log_prob(dec_out)
        n_candidate, vocab_size = lprobs.size()
        bsz = n_candidate // beam_size
        step = state.step
        state.step += 1
        state.done = state.step >= state.max_len

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            lprobs = lprobs + state.scores[:, step - 1].unsqueeze(-1)

        scores, next_tokens = self.chose_beams(lprobs.view(bsz, -1, vocab_size))
        # TODO fix for beam_size != 1, the beam id should be extracted from the next_tokens
        # Not sure if this logic belongs here or to chose_beam
        state.tokens[:, step + 1] = next_tokens.squeeze(-1)
        state.scores[:, step + 1] = scores.squeeze(-1)
        return state

    @overrides
    def chose_beams(self, lprobs_beam: Tensor) -> TopKResult:
        bsz, beam_size, vocab_size = lprobs_beam.shape
        return torch.topk(lprobs_beam.view(bsz, -1), k=beam_size)  # type: ignore

    @overrides
    def finalize(
        self,
        state: BeamSearchState,
        *,
        top: int = 0,
    ) -> SearchResult:
        beam_size = self.beam_size
        bsz = state.tokens.size(0) // beam_size
        top = top if top > 0 else beam_size
        return SearchResult(
            state.tokens.view(bsz, beam_size, -1)[:, :top, :],
            state.scores.view(bsz, beam_size, -1)[:, :top, :],
        )


# TODO
# class PrefixConstrainedBeamSearch(Search):
# class LexicallyConstrainedBeamSearch(Search):
# class LengthConstrainedBeamSearch(Search):
# class DiverseBeamSearch(Search):
# class Sampling(Search):
# class DiverseSiblingsSearch(Search):
