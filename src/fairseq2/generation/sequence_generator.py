# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

import torch
from torch import Tensor
from torch.nn.functional import log_softmax

from fairseq2.data import Collater, SequenceData, VocabularyInfo
from fairseq2.generation.beam_search import BeamSearch, StandardBeamSearch
from fairseq2.models.encoder_decoder import Seq2SeqDecoder
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.typing import Device


@dataclass
class SequenceGeneratorOptions:
    """Holds the options to pass to a sequence generator."""

    beam_size: int = 5
    """The beam size."""

    min_seq_len: int = 1
    """The minimum length of generated sequences, not including the length of
    prefix and suffix sequences."""

    max_seq_len: int = 256
    """The maximum length of generated sequences, not including the length of
    prefix and suffix sequences. See also ``seq_len_a`` and ``seq_len_b``."""

    seq_len_a: float = 1
    """The `a` term of `ax + b` where `x` is the source sequence length. The
    generated sequences will have the maximum length ``min(max_seq_len, ax + b)``."""

    seq_len_b: float = 100
    """The `b` term of `ax + b` where `x` is the source sequence length. The
    generated sequences will have the maximum length ``min(max_seq_len, ax + b)``."""

    len_penalty: float = 1.0
    """The length penalty, where values less than 1.0 favor shorter, values
    greater than 1.0 favor longer sequences."""

    unk_penalty: float = 0.0
    """The unknown symbol penalty, where values less than 0 produce more UNKs,
    values greater than 0 produce fewer UNKs."""

    normalize_scores: bool = True
    """If ``True``, normalizes scores by the length of generated sequences."""

    search: Optional[BeamSearch] = None
    """The beam search implementation to use."""


class Seq2SeqGenerator:
    """Represents a sequence-to-sequence generator."""

    decoder: Seq2SeqDecoder
    opts: SequenceGeneratorOptions
    beam_size: int
    eos_idx: int
    pad_idx: Optional[int]
    unk_idx: Optional[int]
    vocabulary_info: VocabularyInfo
    search: BeamSearch
    collater: Collater

    def __init__(
        self,
        decoder: Seq2SeqDecoder,
        vocabulary_info: VocabularyInfo,
        opts: Optional[SequenceGeneratorOptions] = None,
    ) -> None:
        """
        :param decoder:
            The decoder to use.
        :param vocabulary_info:
            The vocabulary information for the generated sequences.
        :param opts:
            The sequence generation options.
        """
        self.decoder = decoder

        self.opts = opts or SequenceGeneratorOptions()

        if vocabulary_info.pad_idx is None:
            self.beam_size = min(self.opts.beam_size, vocabulary_info.size)
        else:
            # -1 since we never select PAD.
            self.beam_size = min(self.opts.beam_size, vocabulary_info.size - 1)

        if vocabulary_info.eos_idx is None:
            raise ValueError(
                "`vocabulary_info` must have `eos_idx` set for sequence generation."
            )

        self.eos_idx = vocabulary_info.eos_idx
        self.unk_idx = vocabulary_info.unk_idx
        self.pad_idx = vocabulary_info.pad_idx

        self.vocabulary_size = vocabulary_info.size

        self.search = self.opts.search or StandardBeamSearch()

        if vocabulary_info.pad_idx is None:
            self.collater = Collater()
        else:
            self.collater = Collater(self.pad_idx, pad_to_multiple=2)

    def __call__(
        self,
        prefix_tokens: Optional[Tensor],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[Tensor],
        source_seq_len: Optional[int] = None,
    ) -> "SequenceGeneratorOutput":
        opts = self.opts

        if source_seq_len is None:
            max_seq_len = opts.max_seq_len
        else:
            max_seq_len = min(
                opts.max_seq_len, int(opts.seq_len_a * source_seq_len + opts.seq_len_b)
            )

        if opts.min_seq_len > max_seq_len:
            raise ValueError(
                f"The maximum sequence length must be greater than or equal to `min_seq_len` ({opts.min_seq_len}), but is {max_seq_len} instead."
            )

        device = encoder_output.device

        beam_size = opts.beam_size

        batch_size = encoder_output.size(0)
        bsz = batch_size

        # Fan out `encoder_output` to `batch_size` x `beam_size`.
        # (N)
        fan_out_indices = torch.arange(batch_size, device=device)

        # (N) -> (N x B)
        fan_out_indices = fan_out_indices.repeat_interleave(beam_size)
        # (N) -> (N x B)
        encoder_output = encoder_output.index_select(dim=0, index=fan_out_indices)
        # (N) -> (N x B)
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.index_select(
                dim=0, index=fan_out_indices
            )

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_seq_len + 1).to(encoder_output).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_seq_len + 1).to(encoder_output).long()
        )  # +2 for eos and pad
        if self.pad_idx is not None:
            tokens.fill_(self.pad_idx)

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(encoder_output).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized: List[List[Hypothesis]] = [[] for i in range(bsz)]
        # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        if prefix_tokens is None:
            prefix_seq_len = 1

            tokens[:, 0] = self.eos_idx  # if bos_token is None else bos_token
        else:
            # We have to copy `prefix_seq` to `tokens` and ensure that `decoder`
            # is run once before the actual search to bootstrap its state.
            if prefix_tokens.dim() >= 2:
                raise ValueError(
                    f"`prefix_tokens` must be a scalar or a 1-dimensional tensor, but is {prefix_tokens.dim()}-dimensional instead."
                )

            if prefix_tokens.dim() == 0:
                prefix_seq_len = 1
            else:
                prefix_seq_len = prefix_tokens.size(0)

            if prefix_seq_len > max_seq_len:
                raise ValueError(
                    f"The length of `prefix_seq` must be less than or equal to the maximum sequence length ({max_seq_len}), but is {prefix_seq_len} instead."
                )

            tokens[:, :prefix_seq_len] = prefix_tokens

        # The state bag of `decoder`.
        state_bag = IncrementalStateBag()

        decoder_output, decoder_padding_mask = self.decoder.decode(
            tokens[:, :prefix_seq_len],
            None,
            encoder_output,
            encoder_padding_mask,
            state_bag,
        )

        state_bag.increment_step(prefix_seq_len)

        start_step = prefix_seq_len - 1

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens)
            .to(encoder_output.device)
        )
        cand_offsets = (
            torch.arange(0, cand_size).type_as(tokens).to(encoder_output.device)
        )

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        for step in range(start_step, max_seq_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                # Update beam indices in the decoder's state bag.
                state_bag.reorder(reorder_state)

                # And, update the encoder output too.
                encoder_output = encoder_output.index_select(dim=0, index=reorder_state)

                if encoder_padding_mask is not None:
                    encoder_padding_mask = encoder_padding_mask.index_select(
                        dim=0, index=reorder_state
                    )
            # Get log-probs for next sequence step.
            decoder_output, decoder_padding_mask = self.decoder.decode(
                tokens[:, step : step + 1],
                None,  # We never select PAD.
                encoder_output,
                encoder_padding_mask,
                state_bag,
            )

            state_bag.increment_step()

            model_output = self.decoder.project(decoder_output, decoder_padding_mask)

            # For numerical stability run in single precision.
            # (N x B, 1, V)
            lprobs = log_softmax(model_output.logits, dim=-1, dtype=torch.float32)

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            if self.pad_idx is not None:
                lprobs[:, :, self.pad_idx] = -math.inf  # never select pad

            if self.unk_idx is not None:
                lprobs[:, :, self.unk_idx] -= self.opts.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_seq_len:
                lprobs[:, :, : self.eos_idx] = -math.inf
                lprobs[:, :, self.eos_idx + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if step < self.opts.min_seq_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, :, self.eos_idx] = -math.inf

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                step == start_step,
                lprobs.view(bsz, -1, self.vocabulary_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos_idx) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    max_seq_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()
            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            # Till start step all token rows are identical.
            if step > start_step:
                tokens[:, : step + 1] = torch.index_select(
                    tokens[:, : step + 1], dim=0, index=active_bbsz_idx
                )

            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )

            # Till start step all score rows are identical.
            if step > start_step:
                scores[:, : step + 1] = torch.index_select(
                    scores[:, : step + 1], dim=0, index=active_bbsz_idx
                )

            scores.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem.score.item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return SequenceGeneratorOutput(
            batches=finalized, device=device, collater=self.collater
        )

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx: Tensor,
        eos_scores: Tensor,
        tokens: Tensor,
        scores: Tensor,
        finalized: List[List["Hypothesis"]],
        finished: List[bool],
        beam_size: int,
        max_len: int,
    ) -> List[int]:
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[:, : step + 1]

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.opts.normalize_scores:
            eos_scores /= (step + 1) ** self.opts.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

        unfin_idx = torch.div(bbsz_idx, beam_size, rounding_mode="trunc")
        sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent << 32) + unfin_idx
        unique_seen: List[int] = torch.unique(seen).tolist()

        sent_list: List[int] = sent.tolist()
        for i in range(bbsz_idx.size()[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                finalized[sent_list[i]].append(
                    Hypothesis(
                        seq=tokens_clone[i],
                        score=eos_scores[i],
                        step_scores=pos_scores[i],
                    )
                )

        newly_finished: List[int] = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)

            if not finished[unique_sent] and self.is_finished(
                step, unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size
            ):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ) -> bool:
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False


@dataclass
class SequenceGeneratorOutput:
    """Holds the output of a sequence generator."""

    batches: List[List["Hypothesis"]]
    """The list of hypothesis generated per batch per beam."""

    device: Device
    """The device on which generated sequences reside."""

    collater: Optional[Collater] = None
    """The collater to use in :meth:`collate`."""

    def collate(
        self, hypo_idx: int = 0, skip_batch: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Collate the generated sequences at index ``hypo_idx`` in each batch
        into a single tensor.

        :param hypo_idx:
            The index of hypothesis to extract from each batch.
        :param skip_batch:
            If ``True``, if a batch has no hypothesis at index `hypo_idx`, it
            will be skipped instead of raising an error.

        :returns:
          - The collated sequences. *Shape:* :math:`(N,S)`, where :math:`N` is
            the length of :attr:`batches` and :math:`S` is the sequence length.
          - An array where each element represents the length of the sequence at
            the same index in the first returned value. *Shape:* where :math:`N`
            is the batch size.
        """
        if self.collater is None:
            raise RuntimeError("The output has no associated `Collater` instance.")

        if not self.batches and not skip_batch:
            raise ValueError("The output must contain at least one batch.")

        seqs = []

        for batch_idx, batch in enumerate(self.batches):
            if hypo_idx >= len(batch):
                if not skip_batch:
                    raise ValueError(
                        f"Each batch must have at least {hypo_idx + 1} hypotheses, but batch {batch_idx} has only {len(batch)}."
                    )

                continue

            seqs.append(batch[hypo_idx].seq)

        if not seqs:
            # Return a zero-dimensional (not scalar!) tensor.
            return torch.empty((0,), device=self.device, dtype=torch.int), None

        output = cast(SequenceData, self.collater(seqs))

        return output["seqs"], output["seq_lens"] if output["is_ragged"] else None


@dataclass
class Hypothesis:
    """Represents a hypothesis produced by a sequence generator."""

    seq: Tensor
    """The generated sequence."""

    score: Tensor
    """The score of the hypothesis."""

    step_scores: Tensor
    """The score of each individual sequence step."""
