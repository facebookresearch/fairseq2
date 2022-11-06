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
    bos: int = Tokenizer.BOS,
) -> Tensor:
    """Compute the log-probability of the inputs.

    Assumes [batch, value] shape.

    :param val: the values.
    :param temperature: (TODO) the temperature.
    :param pad: the id of the PAD token.
    :param bos: the id of the BOS token.
    :return: the new lprobs.
    """
    # TODO: temperature
    lprobs = nn.functional.log_softmax(val, dim=1)
    lprobs[lprobs != lprobs] = -torch.inf
    lprobs[:, pad] = -torch.inf
    lprobs[:, bos] = -torch.inf
    return lprobs


def unk_penalty(
    lprobs: Tensor,
    penalty: float,
    *,
    unk: int = Tokenizer.UNK,
) -> Tensor:
    """Penalize the unknown token by `penalty`.

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
    """Penalize the unknown token in-place.

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
    """Prevent the EOS token from being selected by forcing it to -inf.

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
    """Prevent the EOS token from being selected by forcing it to -inf in-
    place.

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
    """Force all log probs except the EOS token to -inf.

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
    """Force all log probs except the EOS token to -inf.

    :param lprobs: the probs.
    :param eos: the EOS token.
    :return: the new probs.
    """
    lprobs[:, :eos] = -torch.inf
    lprobs[:, eos + 1 :] = -torch.inf


class SearchResult(NamedTuple):
    tokens: Tensor
    """(bsz x beam_size x len) generated tokens."""

    scores: Tensor
    """(bsz x beam_size) scores of each generation."""


class BeamChoice(NamedTuple):
    scores: Tensor
    tokens: Tensor
    beams: Tensor


class Search(Generic[SearchState]):
    tokenizer: Tokenizer

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def next_query(
        self,
        state: SearchState,
    ) -> Tensor:
        """Gives the next target query.

        :param state:
        :return: (batch * beam, vocab) inference query.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def log_prob(
        self,
        dec_out: Tensor,
        *,
        step: int,
        max_len: int,
    ) -> Tensor:
        """Converts the decoder output and convert it to log probs of the next
        token.

        This is a good place to implement eg:
            - temperature sampling,
            - unk penalty,
            ...

        There are helper functions in this module to help do so.

        dec_out = batch size x vocab size

        :param dec_out: the decoder output (batch, -1)
        :param step: the step.
        :param max_len: the minimum length.
        :return: the (batch, ps)
        """
        raise NotImplementedError

    def choose_beams(
        self,
        beam_size: int,
        lprobs_beam: Tensor,
    ) -> BeamChoice:
        """
        beam_size: the number of beams to select.
        lprobs_beam: (batch size, input beam size, vocab_size) log prob of the full beam + next token
        """
        raise NotImplementedError

    def prepare_state(
        self,
        src_tokens: Tensor,
        *,
        prefix_tokens: Optional[Tensor] = None,
    ) -> SearchState:
        """Prepares a search state for the search.

        Args:
            - src_tokens: input sequence.
            - prefix_tokens: forced generations for the target side.
            They need to be copied into the SearchState.
        """
        raise NotImplementedError

    def finalize(
        self,
        state: SearchState,
        *,
        top: int = 0,
    ) -> SearchResult:
        """From the state, extract the generated sequences and their scores.

        Args:
            - state: state
            - top: restrict the result to the n-best generations per input sequence,
              in descending score order.
              Use 0 (default) to get all generations unsorted.
        """
        raise NotImplementedError


# TODO: python3.10: kw_only=True
@dataclasses.dataclass
class BeamSearchState:
    max_len: int

    tokens: Tensor
    "(bsz * beam_size, buf)"

    scores: Tensor
    "(bsz * beam_size, buf)"

    finished_mask: Tensor
    "(bsz, beam_size)"

    order: Tensor
    "(bsz * beam_size)"

    step: int = 0
    done: bool = False

    @property
    def flat_size(self) -> int:
        return self.tokens.shape[0]

    @property
    def batch_size(self) -> int:
        return self.finished_mask.shape[0]

    @property
    def beam_size(self) -> int:
        return self.finished_mask.shape[1]

    def tokens_beam_view(self) -> Tensor:
        return self.tokens.view(self.batch_size, self.beam_size, -1)

    def scores_beam_view(self) -> Tensor:
        return self.scores.view(self.batch_size, self.beam_size, -1)


class BeamSearch(Search[BeamSearchState]):
    beam_size: int
    min_len: int
    max_len: int
    unk_penalty: float

    def __init__(
        self,
        tokenizer: Tokenizer,
        beam_size: int = 2,
        unk_penalty: float = 1.0,
        min_len: int = 10,
        max_len: int = 256,
    ) -> None:
        super().__init__(tokenizer=tokenizer)
        self.beam_size = beam_size
        self.min_len = min_len
        self.max_len = max_len
        self.unk_penalty = unk_penalty

    @overrides
    def prepare_state(
        self,
        src_tokens: Tensor,
        *,
        prefix_tokens: Optional[Tensor] = None,
    ) -> BeamSearchState:
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size
        max_len = min(self.max_len, 2 * src_len + 10)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        order = (
            torch.arange(
                bsz,
                dtype=torch.int64,
                device=src_tokens.device,
            )
            .view(-1, 1)
            .repeat(1, beam_size)
            .view(-1)
        )

        # initialize buffers
        # +2 for eos and pad
        scores = torch.zeros(
            size=(bsz * beam_size, max_len + 2),
            dtype=torch.float32,
            device=src_tokens.device,
        )

        tokens = torch.full(
            size=(bsz * beam_size, max_len + 2),
            fill_value=self.tokenizer.PAD,
            dtype=torch.long,
            device=src_tokens.device,
        )
        if prefix_tokens is not None:
            # TODO(crutcher): this assumes that there may be a different prefix
            # for every batch member; what is the actual guarantee?
            tokens[:, 0] = prefix_tokens.T[0].view(-1, 1).repeat(1, beam_size).view(-1)
        else:
            tokens[:, 0] = self.tokenizer.BOS

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        finished_mask = torch.zeros(
            size=(bsz, beam_size),
            dtype=torch.bool,
            device=src_tokens.device,
        )

        return BeamSearchState(
            max_len=max_len,
            tokens=tokens,
            scores=scores,
            finished_mask=finished_mask,
            order=order,
            step=0,
        )

    @overrides
    def log_prob(
        self,
        dec_out: Tensor,
        *,
        step: int,
        max_len: int,
    ) -> Tensor:
        lprobs = log_prob(
            dec_out,
            temperature=0.1,
            pad=self.tokenizer.PAD,
            bos=self.tokenizer.BOS,
        )

        unk_penalty_(lprobs, self.unk_penalty, unk=self.tokenizer.UNK)
        if step >= max_len:
            force_eos_(lprobs, eos=self.tokenizer.EOS)

        # TODO: prefix tokens
        # handle prefix tokens (possibly with different lengths)
        # if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:
        # lprobs, tokens, scores = self._prefix_tokens(step, lprobs, scores, tokens, prefix_tokens, beam_size)
        if step < self.min_len:
            # minimum length constraint (does not apply if using prefix_tokens)
            prevent_eos_(lprobs, eos=self.tokenizer.EOS)

        # if self.should_set_src_lengths:
        #     self.search.set_src_lengths(src_lengths)

        # if self.repeat_ngram_blocker is not None:
        #     lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

        return lprobs

    @overrides
    def next_query(
        self,
        state: BeamSearchState,
    ) -> Tensor:
        return state.tokens[:, : state.step + 1]

    @overrides
    def step(
        self,
        dec_out: Tensor,
        state: BeamSearchState,
    ) -> BeamSearchState:
        assert not state.done, "Stepping on a completed search: state.done == True"

        n_candidate, vocab_size = dec_out.size()
        input_beam_size = n_candidate // state.batch_size

        assert input_beam_size == state.beam_size, (
            f"input_beam_size {input_beam_size} must == "
            f"state beam_size {state.beam_size}"
        )

        assert dec_out.shape[1] == self.tokenizer.vocab_size(), (
            f"Input dec_out vocab size {dec_out.shape[1]}) != "
            f"tokenizer vocab size: {self.tokenizer.vocab_size()}"
        )

        # # TODO: should finished_map be persistent state?
        # state.finished_mask = (
        #     state.tokens_beam_view()[:, :, state.step] == self.tokenizer.EOS
        # )

        state.step += 1

        lprobs = self.log_prob(
            dec_out,
            step=state.step,
            max_len=state.max_len,
        )

        lprobs_beam = lprobs.view(state.batch_size, -1, vocab_size)

        # We assume now:
        #  * lprobs_beam is laid out now as (bsz, step.beam_size, vocab)

        # Adjust lprobs_beam_view by the previous state scores:
        lprobs_beam += torch.broadcast_to(
            state.scores_beam_view()[..., state.step - 1].unsqueeze(-1),
            lprobs_beam.shape,
        )

        if state.finished_mask.any():
            # for any (batch, beam) such that state.finished_mask[batch, beam] is true:
            # lprob_beam[ batch, beam, tok!=PAD ] = -inf
            # lprob_beam[ batch, beam, tok==PAD ] = scores_beam_view()[batch, beam, state.step-1]

            # nasty attempt:
            # mask = (
            #     torch.broadcast_to(
            #         state.finished_mask.unsqueeze(-1).unsqueeze(-1),
            #         [state.batch_size, state.beam_size, vocab_size],
            #     )
            #     .contiguous()
            #     .view(state.batch_size, input_beam_size, vocab_size)
            # )
            # TODO: work out appropriate mask expression here
            for batch_idx, batch_mask in enumerate(state.finished_mask):
                for beam_idx, beam_mask in enumerate(batch_mask):
                    if bool(beam_mask):
                        lprobs_beam[batch_idx, beam_idx, :] = -torch.inf
                        lprobs_beam[
                            batch_idx, beam_idx, self.tokenizer.PAD
                        ] = state.scores_beam_view()[
                            batch_idx, beam_idx, state.step - 1
                        ]

        # layout: (bsz, beam_size)
        next_scores, next_tokens, source_beams = self.choose_beams(
            beam_size=state.beam_size,
            lprobs_beam=lprobs_beam,
        )

        # index offset map to go from (bsz, beam) to (bsz * beam) indices.
        flat_offset = (
            (
                state.beam_size
                * torch.arange(state.beam_size, device=state.tokens.device)
            )
            .view(state.beam_size, 1)
            .repeat(1, state.batch_size)
            .view(-1)
        )

        # flat (bsz * beam) selection index.
        flat_beams = source_beams.view(-1) + flat_offset

        state.tokens = torch.index_select(state.tokens, dim=0, index=flat_beams)
        state.scores = torch.index_select(state.scores, dim=0, index=flat_beams)

        state.tokens[:, state.step] = next_tokens.view(-1)
        state.scores[:, state.step] = next_scores.view(-1)

        # TODO: should finished_map be persistent state?
        state.finished_mask = (
            state.tokens_beam_view()[:, :, state.step] == self.tokenizer.EOS
        )
        state.done = (state.step >= state.max_len) or bool(state.finished_mask.all())

        return state

    @overrides
    def choose_beams(
        self,
        beam_size: int,
        lprobs_beam: Tensor,
    ) -> BeamChoice:
        bsz, input_beam_size, vocab_size = lprobs_beam.shape

        assert input_beam_size == beam_size, (
            f"input_beam_size ({input_beam_size}) must == " f"beam_size ({beam_size})"
        )

        # we are interested in the top scoring (beam_size) tokens across all beams
        # in a give batch. by viewing the input as (bsz, input_beam_size * vocab)),
        # the topk(beam_size) gives us those scores and their indices in the combined space.
        multi_beam_view = lprobs_beam.view(bsz, -1)

        # by flattening the (bsz, beam) dimension, topk selects the highest prob
        scores, indices = torch.topk(multi_beam_view.contiguous(), k=beam_size)

        tokens = indices % vocab_size

        sel_beams = indices.div(vocab_size, rounding_mode="trunc")

        # layout (bsz, beam)
        return BeamChoice(scores, tokens, sel_beams)

    @overrides
    def finalize(
        self,
        state: BeamSearchState,
        *,
        top: int = 0,
    ) -> SearchResult:
        tokens = state.tokens_beam_view()
        scores = state.scores_beam_view()

        if top:
            step = state.step
            top_ind = torch.topk(
                scores[:, :, step],
                k=top,
                dim=1,
            ).indices

            scores = torch.stack(
                [
                    torch.stack([scores[n, k, :] for k in beam_ind])
                    for n, beam_ind in enumerate(top_ind)
                ]
            )
            tokens = torch.stack(
                [
                    torch.stack([tokens[n, k, :] for k in beam_ind])
                    for n, beam_ind in enumerate(top_ind)
                ]
            )

        return SearchResult(
            tokens=tokens,
            scores=scores,
        )


# TODO
# class PrefixConstrainedBeamSearch(Search):
# class LexicallyConstrainedBeamSearch(Search):
# class LengthConstrainedBeamSearch(Search):
# class DiverseBeamSearch(Search):
# class Sampling(Search):
# class DiverseSiblingsSearch(Search):
