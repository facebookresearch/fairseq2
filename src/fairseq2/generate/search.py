# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Union

import torch
import torch.nn as nn
from overrides import overrides
from torch import Tensor

from fairseq2.generate.tokenizer import Tokenizer, TokenMeta
from fairseq2.models.transformer import Transformer
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


def dec_out_to_log_prob(
    dec_out: Tensor,
    temperature: float,
    *,
    pad: int,
    bos: int,
) -> Tensor:
    """Compute the log-probability of the inputs.

    Assumes [batch, value] shape.

    :param dec_out: the values.
    :param temperature: (TODO) the temperature.
    :param pad: the id of the PAD token.
    :param bos: the id of the BOS token.
    :return: the new lprobs.
    """
    # TODO: temperature
    lprobs = nn.functional.log_softmax(dec_out, dim=1)
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
    token_meta: TokenMeta

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
            dec_out: Tensor,
        ) -> bool:
            """Process a (batch * beam, scores) inference Tensor.

            :param dec_out: the (batch * beam, scores) decoder output.
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
        model: Transformer,
        src_tokens: Tensor,
        prefix_tokens: Optional[Tensor] = None,
        top: int = 0,
    ) -> torch.Tensor:
        """
        Uses the model and a search strategy to find the more likely generations.
        """
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("forward_encoder"):
            enc_out, enc_attn_mask = model.encode(src_tokens)

        enc_out = _stretch_to_beams(enc_out, self.beam_size)
        if enc_attn_mask is not None:
            enc_attn_mask = _stretch_to_beams(enc_attn_mask, self.beam_size)

        # prepare the search state
        job = self.new_search_job(src_tokens, prefix_tokens=prefix_tokens)
        state_bag = IncrementalStateBag()

        while not job.done:
            with torch.autograd.profiler.record_function("forward_decoder"):
                query_tokens = job.next_query()

                dec_out = model.decode_and_score(
                    query_tokens, enc_out, enc_attn_mask, state_bag
                )
                dec_out = dec_out.squeeze(1)

                state_bag.increment_step()

            with torch.autograd.profiler.record_function("search_step"):
                # Select the last time step prediction
                job.update(dec_out)

        tokens = job.finalize(top=top).tokens
        return tokens.view(-1, tokens.shape[-1])

    def generate_str(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        sentences: List[str],
        *,
        src_bos: str = "",
        tgt_bos: str = "",
        device: torch.device,
    ) -> List[str]:
        src_bos_tok = tokenizer.special_tokens[src_bos] if src_bos else -1
        src_tokens = tokenizer.encode_batch(sentences, bos=src_bos_tok).to(device)
        tgt_bos_tok = (
            torch.tensor(
                [tokenizer.EOS, tokenizer.special_tokens[tgt_bos]], dtype=torch.long
            ).to(device)
            if tgt_bos
            else None
        )
        tgt_tokens = self.generate(model, src_tokens, top=1, prefix_tokens=tgt_bos_tok)
        return tokenizer.decode_batch(tgt_tokens.squeeze(1))


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
            fill_value=strategy.token_meta.PAD,
            dtype=torch.long,
            device=src_tokens.device,
        )

        self.step = 0
        if prefix_tokens is None:
            self.tokens[:, :, 0] = strategy.token_meta.BOS
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
        dec_out: Tensor,
    ) -> bool:
        assert not self.done, "Stepping on a completed search: self.done == True"

        n_candidate, vocab_size = dec_out.size()
        input_beam_size = n_candidate // self.batch_size

        assert input_beam_size == self.beam_size, (
            f"input_beam_size {input_beam_size} must == "
            f"state beam_size {self.beam_size}"
        )

        assert dec_out.shape[1] == self.strategy.token_meta.vocab_size, (
            f"Input dec_out vocab size {dec_out.shape[1]}) != "
            f"tokenizer vocab size: {self.strategy.token_meta.vocab_size}"
        )

        self.step += 1

        lprobs_flat = self._log_prob(
            dec_out=dec_out,
            step=self.step,
            max_len=self.max_len,
        )

        lprobs_beam = lprobs_flat.view(self.batch_size, self.beam_size, vocab_size)

        # We assume now:
        #  * lprobs_beam is laid out now as (bsz, step.beam_size, vocab)

        # Adjust lprobs_beam_view by the previous state scores:
        lprobs_beam += self.scores[..., self.step - 1].unsqueeze(-1)

        if self.finished_mask.any():
            # for any (batch, beam) such that self.finished_mask[batch, beam] is true:
            # lprob_beam[ batch, beam, tok!=PAD ] = -inf
            # lprob_beam[ batch, beam, tok==PAD ] = scores_beam_view()[batch, beam, self.step-1]

            # TODO: work out appropriate mask expression here
            for batch_idx, batch_mask in enumerate(self.finished_mask):
                for beam_idx, beam_mask in enumerate(batch_mask):
                    if bool(beam_mask):
                        lprobs_beam[batch_idx, beam_idx, :] = -torch.inf
                        lprobs_beam[
                            batch_idx,
                            beam_idx,
                            self.strategy.token_meta.PAD,
                        ] = self.scores[batch_idx, beam_idx, self.step - 1]

        # layout: (bsz, beam_size)
        next_scores, next_tokens, source_beams = self._choose_beams(
            lprobs_beam=lprobs_beam,
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
            self.tokens[:, :, self.step] == self.strategy.token_meta.EOS
        )
        self.done = (self.step >= self.max_len) or bool(self.finished_mask.all())

        return not self.done

    def _log_prob(
        self,
        dec_out: Tensor,
        *,
        step: int,
        max_len: int,
    ) -> Tensor:
        lprobs = dec_out_to_log_prob(
            dec_out,
            temperature=0.1,
            pad=self.strategy.token_meta.PAD,
            bos=self.strategy.token_meta.BOS,
        )

        token_penalty_(
            lprobs,
            token=self.strategy.token_meta.UNK,
            penalty=self.strategy.unk_penalty,
        )
        if step >= max_len:
            force_token_(lprobs, token=self.strategy.token_meta.EOS)

        if step < self.strategy.min_len:
            # minimum length constraint (does not apply if using prefix_tokens)
            token_penalty_(
                lprobs, token=self.strategy.token_meta.EOS, penalty=torch.inf
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

    def _choose_beams(
        self,
        lprobs_beam: Tensor,
    ) -> BeamChoice:
        if self.step < self.n_prefix_tokens:
            return self._choose_prefix_tokens(lprobs_beam)

        bsz, input_beam_size, vocab_size = lprobs_beam.shape

        assert input_beam_size == self.beam_size, (
            f"input_beam_size ({input_beam_size}) must == "
            f"beam_size ({self.beam_size})"
        )

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
        token_meta: Union[Tokenizer, TokenMeta],
        unk_penalty: float = 1.0,
        min_len: int = 10,
        max_len: int = 256,
    ) -> None:
        if isinstance(token_meta, Tokenizer):
            token_meta = TokenMeta.from_tokenizer(token_meta)

        super().__init__(
            beam_size=beam_size,
            token_meta=token_meta,
        )
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
