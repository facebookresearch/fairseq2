# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Sequence, Union

import torch
import torch.nn as nn
from overrides import overrides
from torch import Tensor

from fairseq2.data import CString, StringLike
from fairseq2.data.text import Tokenizer, VocabularyInfo
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.nn.incremental_state import IncrementalStateBag


def token_penalty_(
    lprobs: Tensor,
    *,
    token: Union[int, slice],
    penalty: float,
) -> None:
    """Penalize the token in-place.

    Assumes [batch, value] shape.

    :param *:
    :param lprobs: the probs.
    :param token: the token to penalize.
    :param penalty: the penalty.
    """
    lprobs[:, token] -= penalty


def force_token_(
    lprobs: Tensor,
    *,
    token: int,
) -> None:
    """Force all log probs except the token to -inf.

    :param *:
    :param lprobs: the probs.
    :param token: the token.
    """
    token_penalty_(lprobs, token=slice(token), penalty=torch.inf)
    token_penalty_(lprobs, token=slice(token + 1, None), penalty=torch.inf)


def decoder_out_to_log_prob(
    decoder_out: Tensor,
    temperature: float,
    *,
    pad: int,
    bos: int,
) -> Tensor:
    """Compute the log-probability of the inputs.

    Assumes [batch, value] shape.

    :param decoder_out: the values.
    :param temperature: (TODO) the temperature.
    :param pad: the id of the PAD token.
    :param bos: the id of the BOS token.
    :return: the new lprobs.
    """
    # TODO: temperature
    lprobs = nn.functional.log_softmax(decoder_out, dim=1)
    lprobs[lprobs != lprobs] = -torch.inf
    token_penalty_(lprobs, token=pad, penalty=torch.inf)
    token_penalty_(lprobs, token=bos, penalty=torch.inf)
    return lprobs


class SearchResult(NamedTuple):
    tokens: Tensor
    """(bsz x beam_size x len) generated tokens."""

    scores: Tensor
    """(bsz x beam_size) scores of each generation."""


class BeamChoice(NamedTuple):
    # layout (bsz, beam)
    scores: Tensor
    tokens: Tensor
    beams: Tensor


@dataclass(frozen=True)
class SearchStrategy(ABC):
    """Abstract token generation search strategy."""

    beam_size: int
    vocab_info: VocabularyInfo

    @dataclass
    class SearchJob(ABC):
        """Abstract token generation search job instance."""

        strategy: "SearchStrategy"
        batch_size: int
        done: bool

        @property
        def beam_size(self) -> int:
            return self.strategy.beam_size

        @abstractmethod
        def next_query(self) -> Tensor:
            """Gives the next target query.

            :return: (batch * beam, vocab) inference query.
            """
            raise NotImplementedError

        @abstractmethod
        def update(
            self,
            decoder_out: Tensor,
        ) -> bool:
            """Process a (batch * beam, scores) inference Tensor.

            :param decoder_out: the (batch * beam, scores) decoder output.
            :return: is the job still active?
            """
            raise NotImplementedError

        @abstractmethod
        def finalize(
            self,
            *,
            top: int = 0,
        ) -> SearchResult:
            """Extract the generated sequences and their scores.

            :param top: restrict the result to the n-best generations per input sequence,
                  in descending score order. Use 0 (default) to get all generations unsorted.
            :return: a SearchResult
            """
            raise NotImplementedError

    @abstractmethod
    def new_search_job(
        self,
        src_tokens: Tensor,
        *,
        prefix_tokens: Optional[Tensor] = None,
    ) -> SearchJob:
        """Prepares a search state for the search.

        :param src_tokens: (batch, seq) input sequences.
        :param prefix_tokens: [Optional] Either (batch, seq) or (seq) prefixes.
        :return: a new SearchJob.
        """
        raise NotImplementedError

    @torch.inference_mode()
    def generate(
        self,
        model: EncoderDecoderModel,
        src_tokens: Tensor,
        src_token_lens: Optional[Tensor],
        prefix_tokens: Optional[Tensor] = None,
        top: int = 0,
    ) -> torch.Tensor:
        """
        Uses the model and a search strategy to find the more likely generations.
        """
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("forward_encoder"):
            encoder_out, encoder_padding_mask = model.encode(src_tokens, src_token_lens)

        encoder_out = _stretch_to_beams(encoder_out, self.beam_size)
        if encoder_padding_mask is not None:
            encoder_padding_mask = _stretch_to_beams(
                encoder_padding_mask, self.beam_size
            )

        # prepare the search state
        job = self.new_search_job(src_tokens, prefix_tokens=prefix_tokens)
        state_bag = IncrementalStateBag()

        while not job.done:
            with torch.autograd.profiler.record_function("forward_decoder"):
                query_tokens = job.next_query()

                padding_mask = query_tokens.ne(self.vocab_info.pad_idx)
                seq_lens = torch.count_nonzero(padding_mask, dim=-1)

                decoder_out = model.decode_and_project(
                    query_tokens, seq_lens, encoder_out, encoder_padding_mask, state_bag
                )
                logits = decoder_out.logits.squeeze(1)

                state_bag.increment_step()

            with torch.autograd.profiler.record_function("search_step"):
                # Select the last time step prediction
                job.update(logits)

        tokens = job.finalize(top=top).tokens
        return tokens.view(-1, tokens.shape[-1])

    def generate_str(
        self,
        model: EncoderDecoderModel,
        tokenizer: Tokenizer,
        sentences: Union[StringLike, Sequence[StringLike]],
        *,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        device: torch.device,
    ) -> List[StringLike]:
        if isinstance(sentences, (str, CString)):
            sentences = [sentences]

        task = "translation"

        src_encoder = tokenizer.create_encoder(
            task, lang=src_lang, mode="source", device=device
        )
        tgt_encoder = tokenizer.create_encoder(
            task, lang=tgt_lang, mode="target", device=device
        )

        tgt_decoder = tokenizer.create_decoder()

        src_indices = src_encoder(sentences)
        # Start with an empty sentence.
        tgt_indices = tgt_encoder([""] * len(sentences))

        padding_mask = src_indices.ne(self.vocab_info.pad_idx)
        src_indices_lens = torch.count_nonzero(padding_mask, dim=-1)

        tgt_indices = self.generate(
            model, src_indices, src_indices_lens, top=1, prefix_tokens=tgt_indices
        )

        return tgt_decoder(tgt_indices.squeeze(1))


@dataclass
class BeamSearchJob(SearchStrategy.SearchJob):
    strategy: "BeamSearchStrategy"
    batch_size: int
    max_len: int
    step: int

    tokens: Tensor
    "(bsz, beam_size, seq)"

    scores: Tensor
    "(bsz, beam_size, seq)"

    finished_mask: Tensor
    "(bsz, beam_size)"

    @property
    def flat_size(self) -> int:
        return self.batch_size * self.beam_size

    def __init__(
        self,
        *,
        strategy: "BeamSearchStrategy",
        src_tokens: Tensor,
        prefix_tokens: Optional[Tensor] = None,
    ):
        batch_size, src_len = src_tokens.size()[:2]

        super().__init__(
            strategy=strategy,
            batch_size=batch_size,
            done=False,
        )

        self.max_len = min(strategy.max_len, 2 * src_len + 10)

        state_size = (self.batch_size, strategy.beam_size, self.max_len + 2)

        # initialize buffers
        # +2 for eos and pad
        self.scores = torch.zeros(
            size=state_size,
            dtype=torch.float32,
            device=src_tokens.device,
        )

        self.tokens = torch.full(
            size=state_size,
            fill_value=strategy.vocab_info.pad_idx,
            dtype=torch.long,
            device=src_tokens.device,
        )

        self.step = 0
        if prefix_tokens is None:
            self.tokens[:, :, 0] = strategy.vocab_info.bos_idx
            self.n_prefix_tokens = 1
        elif prefix_tokens.ndim == 0:
            self.tokens[:, :, 0] = prefix_tokens
            self.n_prefix_tokens = 1
        elif prefix_tokens.ndim == 1:
            self.tokens[:, :, : prefix_tokens.size(-1)] = prefix_tokens
            self.n_prefix_tokens = prefix_tokens.size(-1)
        elif prefix_tokens.ndim == 2:
            tokens_pre = self.tokens[:, :, : prefix_tokens.size(-1)]
            tokens_pre[...] = torch.broadcast_to(  # type: ignore
                prefix_tokens.unsqueeze(1),
                tokens_pre.shape,
            )
            self.n_prefix_tokens = prefix_tokens.size(-1)
        else:
            raise ValueError(f"Invalid prefix_tokens.shape: {prefix_tokens.shape}")

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        self.finished_mask = torch.zeros(
            size=(self.batch_size, strategy.beam_size),
            dtype=torch.bool,
            device=src_tokens.device,
        )

    @overrides
    def next_query(self) -> Tensor:
        return self.tokens.view(self.flat_size, -1)[:, self.step : self.step + 1]

    def tokens_beam_view(self) -> Tensor:
        return self.tokens.view(self.batch_size, self.beam_size, -1)

    def scores_beam_view(self) -> Tensor:
        return self.scores.view(self.batch_size, self.beam_size, -1)

    @overrides
    def update(
        self,
        decoder_out: Tensor,
    ) -> bool:
        assert not self.done, "Stepping on a completed search: self.done == True"

        n_candidate, vocab_size = decoder_out.size()
        input_beam_size = n_candidate // self.batch_size

        assert input_beam_size == self.beam_size, (
            f"input_beam_size {input_beam_size} must == "
            f"state beam_size {self.beam_size}"
        )

        assert decoder_out.shape[1] == self.strategy.vocab_info.size, (
            f"Input decoder_out vocab size {decoder_out.shape[1]}) != "
            f"tokenizer vocab size: {self.strategy.vocab_info.size}"
        )

        self.step += 1

        lprobs_flat = self._log_prob(
            decoder_out=decoder_out,
            step=self.step,
            max_len=self.max_len,
        )

        lprobs_beam = lprobs_flat.view(self.batch_size, self.beam_size, vocab_size)

        # We assume now:
        #  * lprobs_beam is laid out now as (bsz, step.beam_size, vocab)

        # Adjust lprobs_beam_view by the previous state scores:
        lprobs_beam += self.scores[..., self.step - 1].unsqueeze(-1)

        if self.finished_mask.any():
            # for any finished beam, force PAD, but keep the beam score.
            PAD = self.strategy.vocab_info.pad_idx
            lprobs_beam.masked_fill_(self.finished_mask.unsqueeze(-1), -torch.inf)
            final_scores = self.scores[self.finished_mask][:, self.step - 1]
            lprobs_beam[:, :, PAD].masked_scatter_(self.finished_mask, final_scores)

        # layout: (bsz, beam_size)
        next_scores, next_tokens, source_beams = self._choose_beams(
            lprobs_beam=lprobs_beam
        )

        # Select the best prefix beams
        self.tokens = torch.stack(
            [
                torch.index_select(
                    self.tokens[batch],
                    dim=0,
                    index=beams,
                )
                for batch, beams in enumerate(source_beams)
            ],
        )
        self.scores = torch.stack(
            [
                torch.index_select(
                    self.scores[batch],
                    dim=0,
                    index=beams,
                )
                for batch, beams in enumerate(source_beams)
            ],
        )

        # update the state
        self.tokens[:, :, self.step] = next_tokens
        self.scores[:, :, self.step] = next_scores

        # Generations are marked as finished at the first EOS
        self.finished_mask += (
            self.tokens[:, :, self.step] == self.strategy.vocab_info.eos_idx
        )
        self.done = (self.step >= self.max_len) or bool(self.finished_mask.all())

        return not self.done

    def _log_prob(
        self,
        decoder_out: Tensor,
        *,
        step: int,
        max_len: int,
    ) -> Tensor:
        lprobs = decoder_out_to_log_prob(
            decoder_out,
            temperature=0.1,
            pad=self.strategy.vocab_info.pad_idx,
            bos=self.strategy.vocab_info.bos_idx,
        )

        token_penalty_(
            lprobs,
            token=self.strategy.vocab_info.unk_idx,
            penalty=self.strategy.unk_penalty,
        )
        if step >= max_len:
            force_token_(lprobs, token=self.strategy.vocab_info.eos_idx)

        if step < self.strategy.min_len:
            # minimum length constraint (does not apply if using prefix_tokens)
            token_penalty_(
                lprobs, token=self.strategy.vocab_info.eos_idx, penalty=torch.inf
            )

        # if self.should_set_src_lengths:
        #     self.search.set_src_lengths(src_lengths)

        # if self.repeat_ngram_blocker is not None:
        #     lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

        return lprobs

    def _choose_prefix_tokens(self, lprobs_beam: Tensor) -> BeamChoice:
        """Force all beams to chose symbol from 'prefix_tokens'."""
        # Alternatively this could be moved to _log_prob by penalizing all non-forced tokens.
        assert self.step < self.n_prefix_tokens
        # prefix_tokens has already been copied to self.tokens
        next_tokens = self.tokens[:, :, self.step]
        scores = torch.take_along_dim(lprobs_beam, next_tokens).reshape(
            next_tokens.shape
        )
        return BeamChoice(
            scores=scores,
            tokens=next_tokens,
            beams=torch.zeros_like(next_tokens),
        )

    def _choose_beams(self, lprobs_beam: Tensor) -> BeamChoice:
        bsz, input_beam_size, vocab_size = lprobs_beam.shape

        if self.step < self.n_prefix_tokens:
            return self._choose_prefix_tokens(lprobs_beam)
        elif self.step == self.n_prefix_tokens:
            # At this point all beams are the same.
            # We need to compute topk only on one beam, otherwise we will extract
            # the same token across all beams.
            # Note: we could theoritically modify next_query() to avoid computing
            # probs on identical beams, but there are some shape issues with "encoder_out".
            lprobs_beam = lprobs_beam[:, :1, :]

        # we are interested in the top scoring (beam_size) tokens across all beams
        # in a given batch. by viewing the input as (bsz, input_beam_size * vocab)),
        # the topk(beam_size) gives us those scores and their indices in the combined space.
        multi_beam_view = lprobs_beam.view(bsz, -1)

        # by flattening the (bsz, beam) dimension, topk selects the highest prob
        scores, indices = torch.topk(multi_beam_view.contiguous(), k=self.beam_size)

        tokens = indices % vocab_size

        sel_beams = indices.div(vocab_size, rounding_mode="trunc")

        # layout (bsz, beam)
        return BeamChoice(scores, tokens, sel_beams)

    @overrides
    def finalize(
        self,
        *,
        top: int = 0,
    ) -> SearchResult:
        tokens = self.tokens
        scores = self.scores

        if top:
            step = self.step
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

        return SearchResult(tokens=tokens, scores=scores)


@dataclass(frozen=True)
class BeamSearchStrategy(SearchStrategy):
    min_len: int
    max_len: int
    unk_penalty: float

    def __init__(
        self,
        *,
        beam_size: int = 2,
        vocab_info: VocabularyInfo,
        unk_penalty: float = 1.0,
        min_len: int = 10,
        max_len: int = 256,
    ) -> None:
        super().__init__(beam_size=beam_size, vocab_info=vocab_info)
        # because we're frozen.
        self.__dict__["min_len"] = min_len
        self.__dict__["max_len"] = max_len
        self.__dict__["unk_penalty"] = unk_penalty

    @overrides
    def new_search_job(
        self,
        src_tokens: Tensor,
        *,
        prefix_tokens: Optional[Tensor] = None,
    ) -> BeamSearchJob:
        return BeamSearchJob(
            strategy=self,
            src_tokens=src_tokens,
            prefix_tokens=prefix_tokens,
        )


# TODO
# class PrefixConstrainedBeamSearch(Search):
# class LexicallyConstrainedBeamSearch(Search):
# class LengthConstrainedBeamSearch(Search):
# class DiverseBeamSearch(Search):
# class Sampling(Search):
# class DiverseSiblingsSearch(Search):


def _stretch_to_beams(t: torch.Tensor, beam_size: int) -> torch.Tensor:
    """Stretch (batch, ...) to (batch * beam_size, ...)

    :param t: the tensor
    :param beam_size: the target beam size.
    :return: a new stretched tensor copy.
    """
    # duplicate over the beam space; this makes a copy
    return torch.broadcast_to(
        t.unsqueeze(1),
        (t.shape[0], beam_size, *t.shape[1:]),
    ).reshape(-1, *t.shape[1:])
