# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union, final

import torch
from torch import Tensor
from torch.nn.functional import softmax

from fairseq2.data import VocabularyInfo
from fairseq2.generation.generator import (
    Hypothesis,
    Seq2SeqGenerator,
    Seq2SeqGeneratorOutput,
    SequenceGenerator,
    SequenceGeneratorOutput,
    StepHook,
)
from fairseq2.generation.step_processor import StepProcessor
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.ops import repeat_interleave
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import finaloverride, override


@final
class SamplingSequenceGenerator(SequenceGenerator):
    """Represents a sequence generator based on sampling."""

    sampler: Sampler
    num_gens: int
    min_gen_len: int
    max_gen_len: int
    max_seq_len: int
    echo_prompt: bool
    compute_scores: bool
    normalize_scores: bool
    temperature: float
    unk_penalty: float
    len_penalty: float
    prefill_chunk_size: Optional[int]
    decode_capacity_increment: Optional[int]
    step_processors: List[StepProcessor]

    def __init__(
        self,
        model: DecoderModel,
        sampler: Sampler,
        *,
        num_gens: int = 1,
        min_gen_len: int = 1,
        max_gen_len: int = 128,
        max_seq_len: int = 1024,
        echo_prompt: bool = False,
        compute_scores: bool = False,
        normalize_scores: bool = True,
        temperature: float = 0.6,
        unk_penalty: float = 0.0,
        len_penalty: float = 1.0,
        prefill_chunk_size: Optional[int] = 512,
        decode_capacity_increment: Optional[int] = 16,
        step_processors: Optional[Sequence[StepProcessor]] = None,
    ) -> None:
        """
        :param model:
            The decoder model to use for generation.
        :param sampler:
            The sampling algorithm.
        :param num_gens:
            The number of sequences to generate per prompt.
        :param min_gen_len:
            The minimum allowed generation length.
        :param max_gen_len:
            The maximum allowed generation length.
        :param max_seq_len:
            The maximum allowed sequence length including prompt.
        :param echo_prompt:
            If ``True``, returns generated sequences with prompts appended.
        :param compute_scores:
            If ``True``, computes scores of generated sequences.
        :param normalize_scores:
            If ``True``, normalizes scores by lengths of generated sequences.
        :param temperature:
            The logit temperature, where values greater than 1.0 produce more
            uniform logits; values less than 1.0 produce sharper logits.
        :param unk_penalty:
            The UNK symbol penalty, where values less than 0 produce more UNKs;
            values greater than 0 produce fewer UNKs.
        :param len_penalty:
            The length penalty, where values less than 1.0 favor shorter
            sequences; values greater than 1.0 favor longer sequences.
        :param prefill_chunk_size:
            The prefill will be performed incrementally by chunks of this size.
            If ``None``, the entire prefill will be performed at once.
        :param decode_capacity_increment:
            The sequence length capacity of state tensors will be incremented by
            multiples of this value. If ``None``, state tensors will be
            preallocated with a capacity of ``max_seq_len``.
        :param step_processors:
            The processors to call at each generation step.
        """
        super().__init__(model)

        if min_gen_len < 1:
            raise ValueError(
                f"`min_gen_len` must be greater than or equal to 1, but is {min_gen_len} instead."
            )

        if max_gen_len < 1:
            raise ValueError(
                f"`max_gen_len` must be greater than or equal to 1, but is {max_gen_len} instead."
            )

        if min_gen_len > max_gen_len:
            raise ValueError(
                f"`min_gen_len` must be less than or equal to `max_gen_len` ({max_gen_len}), but is {min_gen_len} instead."
            )

        if prefill_chunk_size is not None and prefill_chunk_size < 1:
            raise ValueError(
                f"`prefill_chunk_size` must be greater than or equal to 1, but is {prefill_chunk_size} instead."
            )

        if decode_capacity_increment is not None and decode_capacity_increment < 1:
            raise ValueError(
                f"`decode_capacity_increment` must be greater than or equal to 1, but is {decode_capacity_increment} instead."
            )

        self.sampler = sampler
        self.num_gens = num_gens
        self.min_gen_len = min_gen_len
        self.max_gen_len = max_gen_len
        self.max_seq_len = max_seq_len
        self.echo_prompt = echo_prompt
        self.compute_scores = compute_scores
        self.normalize_scores = normalize_scores
        self.temperature = temperature
        self.unk_penalty = unk_penalty
        self.len_penalty = len_penalty
        self.prefill_chunk_size = prefill_chunk_size
        self.decode_capacity_increment = decode_capacity_increment

        if step_processors:
            self.step_processors = list(step_processors)
        else:
            self.step_processors = []

    @finaloverride
    @torch.inference_mode()
    def __call__(
        self, prompt_seqs: Tensor, prompt_padding_mask: Optional[PaddingMask]
    ) -> SequenceGeneratorOutput:
        op = _SamplingSequenceGeneratorOp(
            self.model,
            prompt_seqs,
            prompt_padding_mask,
            self.sampler,
            self.num_gens,
            self.min_gen_len,
            self.max_gen_len,
            self.max_seq_len,
            self.echo_prompt,
            self.compute_scores,
            self.normalize_scores,
            self.temperature,
            self.unk_penalty,
            self.len_penalty,
            self.prefill_chunk_size,
            self.decode_capacity_increment,
            self.step_processors,
            self._step_hooks,
        )

        hypotheses = op()

        return SequenceGeneratorOutput(hypotheses)


@final
class SamplingSeq2SeqGenerator(Seq2SeqGenerator):
    """Represents a sequence-to-sequence generator based on sampling."""

    sampler: Sampler
    num_gens: int
    min_gen_len: int
    max_gen_len: Tuple[int, int]
    max_seq_len: int
    echo_prompt: bool
    compute_scores: bool
    normalize_scores: bool
    temperature: float
    unk_penalty: float
    len_penalty: float
    prefill_chunk_size: Optional[int]
    decode_capacity_increment: Optional[int]
    step_processors: List[StepProcessor]

    def __init__(
        self,
        model: EncoderDecoderModel,
        sampler: Sampler,
        *,
        num_gens: int = 1,
        min_gen_len: int = 1,
        max_gen_len: Tuple[int, int] = (1, 128),
        max_seq_len: int = 1024,
        echo_prompt: bool = False,
        compute_scores: bool = False,
        normalize_scores: bool = True,
        temperature: float = 0.6,
        unk_penalty: float = 0.0,
        len_penalty: float = 1.0,
        prefill_chunk_size: Optional[int] = 512,
        decode_capacity_increment: Optional[int] = 16,
        step_processors: Optional[Sequence[StepProcessor]] = None,
    ) -> None:
        """
        :param model:
            The encoder-decoder model to use for generation.
        :param sampler:
            The sampling algorithm.
        :param num_gens:
            The number of sequences to generate per prompt.
        :param min_gen_len:
            The minimum allowed generation length.
        :param max_gen_len:
            The maximum allowed generation length as ``ax + b``, where ``x`` is
            the source sequence length.
        :param max_seq_len:
            The maximum allowed sequence length including prompt.
        :param echo_prompt:
            If ``True``, returns generated sequences with prompts appended.
        :param compute_scores:
            If ``True``, computes scores of generated sequences.
        :param normalize_scores:
            If ``True``, normalizes scores by lengths of generated sequences.
        :param temperature:
            The logit temperature, where values greater than 1.0 produce more
            uniform logits; values less than 1.0 produce sharper logits.
        :param unk_penalty:
            The UNK symbol penalty, where values less than 0 produce more UNKs;
            values greater than 0 produce fewer UNKs.
        :param len_penalty:
            The length penalty, where values less than 1.0 favor shorter
            sequences; values greater than 1.0 favor longer sequences.
        :param prefill_chunk_size:
            The prefill will be performed incrementally by chunks of this size.
            If ``None``, the entire prefill will be performed at once.
        :param decode_capacity_increment:
            The sequence length capacity of state tensors will be incremented by
            multiples of this value. If ``None``, state tensors will be
            preallocated with a capacity of ``max_seq_len``.
        :param step_processors:
            The processors to call at each generation step.
        """
        super().__init__(model)

        if min_gen_len < 1:
            raise ValueError(
                f"`min_gen_len` must be greater than or equal to 1, but is {min_gen_len} instead."
            )

        if prefill_chunk_size is not None and prefill_chunk_size < 1:
            raise ValueError(
                f"`prefill_chunk_size` must be greater than or equal to 1, but is {prefill_chunk_size} instead."
            )

        if decode_capacity_increment is not None and decode_capacity_increment < 1:
            raise ValueError(
                f"`decode_capacity_increment` must be greater than or equal to 1, but is {decode_capacity_increment} instead."
            )

        self.sampler = sampler
        self.num_gens = num_gens
        self.min_gen_len = min_gen_len
        self.max_gen_len = max_gen_len
        self.max_seq_len = max_seq_len
        self.echo_prompt = echo_prompt
        self.compute_scores = compute_scores
        self.normalize_scores = normalize_scores
        self.temperature = temperature
        self.unk_penalty = unk_penalty
        self.len_penalty = len_penalty
        self.prefill_chunk_size = prefill_chunk_size
        self.decode_capacity_increment = decode_capacity_increment

        if step_processors:
            self.step_processors = list(step_processors)
        else:
            self.step_processors = []

    @finaloverride
    @torch.inference_mode()
    def __call__(
        self,
        source_seqs: Tensor,
        source_padding_mask: Optional[PaddingMask],
        prompt_seqs: Tensor,
        prompt_padding_mask: Optional[PaddingMask],
    ) -> Seq2SeqGeneratorOutput:
        # (P, S)
        encoder_output, encoder_padding_mask = self.model.encode(
            source_seqs, source_padding_mask
        )

        if source_padding_mask is None:
            max_source_len = source_seqs.size(1)
        else:
            max_source_len = int(source_padding_mask.seq_lens.max())

        a_term, b_term = self.max_gen_len

        # In seq2seq generation, the maximum generation length is relative to
        # the source sequence length.
        max_gen_len = int(a_term * max_source_len + b_term)

        if max_gen_len < 1:
            raise ValueError(
                f"`max_gen_len` must be greater than or equal to 1, but is {max_gen_len} instead. Adjust your `max_gen_len` argument."
            )

        if self.min_gen_len > max_gen_len:
            raise ValueError(
                f"`min_gen_len` must be less than or equal to `max_gen_len` ({max_gen_len}), but is {self.min_gen_len} instead. Adjust your `max_gen_len` argument."
            )

        op = _SamplingSeq2SeqGeneratorOp(
            self.model,
            encoder_output,
            encoder_padding_mask,
            prompt_seqs,
            prompt_padding_mask,
            self.sampler,
            self.num_gens,
            self.min_gen_len,
            max_gen_len,
            self.max_seq_len,
            self.echo_prompt,
            self.compute_scores,
            self.normalize_scores,
            self.temperature,
            self.unk_penalty,
            self.len_penalty,
            self.prefill_chunk_size,
            self.decode_capacity_increment,
            self.step_processors,
            self._step_hooks,
        )

        hypotheses = op()

        return Seq2SeqGeneratorOutput(hypotheses, encoder_output, encoder_padding_mask)


class Sampler(ABC):
    """Represents a sampling algorithm."""

    @abstractmethod
    def __call__(self, probs: Tensor) -> Tensor:
        """
        :param probs:
            The next-step probability of each vocabulary entry. *Shape:*
            :math:`(N,V)`, where :math:`N` is the batch size and :math:`V` is
            the size of the vocabulary.
        """


@final
class TopPSampler(Sampler):
    """Selects the next step randomly from the smallest set of candidates for
    which the cumulative probability exceeds a specified value p.

    Also known as Nucleus Sampling as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1904.09751`.
    """

    p: float

    def __init__(self, p: float = 0.9) -> None:
        """
        :param p:
            The cumulative probability threshold.
        """
        self.p = p

    @finaloverride
    def __call__(self, probs: Tensor) -> Tensor:
        # Previous operations in the generation like step processors might have
        # modified the probabilities. Normalize the distribution.
        probs = probs / probs.sum(dim=-1, keepdim=True)

        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

        # (N, V)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = (cumsum_probs - sorted_probs) > self.p

        sorted_probs[mask] = 0.0

        # Normalize.
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        # (N, 1)
        indices = sorted_indices.gather(
            dim=-1, index=torch.multinomial(sorted_probs, num_samples=1)
        )

        # (N, 1) -> (N)
        return indices.squeeze(-1)  # type: ignore[no-any-return]


@final
class TopKSampler(Sampler):
    """Selects the next step randomly from the k mosty likely candidates."""

    k: int

    def __init__(self, k: int) -> None:
        """
        :param k:
            The number of candidates to select from.
        """
        self.k = k

    @finaloverride
    def __call__(self, probs: Tensor) -> Tensor:
        k = min(self.k, probs.size(1))

        if k == 1:
            # (N, 1)
            indices = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            # (N, V) -> (N, K)
            topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1, sorted=False)

            # Normalize.
            topk_probs /= topk_probs.sum(dim=-1, keepdim=True)

            # (N, 1)
            indices = topk_indices.gather(
                dim=-1, index=torch.multinomial(topk_probs, num_samples=1)
            )

        # (N, 1) -> (N)
        return indices.squeeze(-1)


class _SamplingSequenceGeneratorOpBase(ABC):
    sampler: Sampler
    eos_idx: int
    pad_idx: Optional[int]
    unk_idx: Optional[int]
    num_gens: int
    min_prompt_len: int
    max_prompt_len: int
    min_seq_len: int
    max_seq_len: int
    echo_prompt: bool
    compute_scores: bool
    normalize_scores: bool
    temperature: float
    unk_penalty: float
    len_penalty: float
    prefill_chunk_size: Optional[int]
    step_processors: List[StepProcessor]
    step_nr: int
    state_bag: IncrementalStateBag
    prompt_lens: Optional[Tensor]
    prompt_mask: Optional[Tensor]
    prompt_indices: Tensor
    seqs: Tensor
    step_scores: Optional[Tensor]
    output: List[List[Hypothesis]]
    step_hooks: Dict[int, StepHook]

    def __init__(
        self,
        prompt_seqs: Tensor,
        prompt_padding_mask: Optional[PaddingMask],
        sampler: Sampler,
        vocab_info: VocabularyInfo,
        num_gens: int,
        min_gen_len: int,
        max_gen_len: int,
        max_seq_len: int,
        echo_prompt: bool,
        compute_scores: bool,
        normalize_scores: bool,
        temperature: float,
        unk_penalty: float,
        len_penalty: float,
        prefill_chunk_size: Optional[int],
        decode_capacity_increment: Optional[int],
        step_processors: List[StepProcessor],
        step_hooks: Dict[int, StepHook],
    ) -> None:
        self.sampler = sampler

        assert vocab_info.eos_idx is not None

        self.eos_idx = vocab_info.eos_idx
        self.pad_idx = vocab_info.pad_idx
        self.unk_idx = vocab_info.unk_idx

        self.num_gens = num_gens

        min_prompt_idx: Union[int, Tensor]
        max_prompt_idx: Union[int, Tensor]

        if prompt_padding_mask is None:
            self.min_prompt_len, min_prompt_idx = prompt_seqs.size(1), 0
            self.max_prompt_len, max_prompt_idx = prompt_seqs.size(1), 0
        else:
            prompt_seq_lens = prompt_padding_mask.seq_lens

            min_prompt_len, min_prompt_idx = torch.min(prompt_seq_lens, dim=0)
            max_prompt_len, max_prompt_idx = torch.max(prompt_seq_lens, dim=0)

            self.min_prompt_len = int(min_prompt_len)
            self.max_prompt_len = int(max_prompt_len)

            if self.min_prompt_len == self.max_prompt_len:
                prompt_padding_mask = None

        if self.min_prompt_len < 1:
            raise ValueError(f"`prompt_seqs[{int(min_prompt_idx)}]` must not be empty.")

        if self.max_prompt_len >= max_seq_len:
            raise ValueError(
                f"The length of `prompt_seqs[{int(max_prompt_idx)}]` must be less than `max_seq_len` ({max_seq_len}), but is {self.max_prompt_len} instead."
            )

        self.min_seq_len = min(max_seq_len, self.max_prompt_len + min_gen_len)
        self.max_seq_len = min(max_seq_len, self.max_prompt_len + max_gen_len)

        self.echo_prompt = echo_prompt
        self.compute_scores = compute_scores
        self.normalize_scores = normalize_scores
        self.temperature = temperature
        self.unk_penalty = unk_penalty
        self.len_penalty = len_penalty
        self.prefill_chunk_size = prefill_chunk_size
        self.step_processors = step_processors
        self.step_hooks = step_hooks

        self.step_nr = 0

        self.state_bag = IncrementalStateBag(
            self.max_seq_len, capacity_increment=decode_capacity_increment
        )

        if prompt_padding_mask is None:
            self.prompt_lens = None
            self.prompt_mask = None
        else:
            # (P)
            self.prompt_lens = prompt_padding_mask.seq_lens

            # (P, S_prm)
            self.prompt_mask = prompt_padding_mask.materialize()

        device = prompt_seqs.device

        num_prompts = prompt_seqs.size(0)

        # Holds the prompt indices of the generated sequences.
        # (P)
        self.prompt_indices = torch.arange(num_prompts, device=device)

        # Holds the generated sequences.
        # (P, S)
        self.seqs = torch.empty(
            (num_prompts, self.max_seq_len), device=device, dtype=torch.int64
        )

        if self.compute_scores:
            # Holds the step scores of the generated sequences.
            # (P, S)
            self.step_scores = torch.zeros(
                (num_prompts, self.max_seq_len), device=device, dtype=torch.float32
            )
        else:
            self.step_scores = None

        # Bootstrap the sequences.
        self.seqs[:, : self.max_prompt_len] = prompt_seqs[:, : self.max_prompt_len]

        # Holds the sequences that have reached EOS.
        self.output = [[] for _ in range(num_prompts)]

    def __call__(self) -> List[List[Hypothesis]]:
        self._prepare_state()

        for self.step_nr in range(self.min_prompt_len, self.max_seq_len):
            if not self._step():
                break

        if self.compute_scores:
            # Sort the hypotheses by their scores before returning.
            for hypotheses in self.output:
                hypotheses.sort(key=lambda h: h.score, reverse=True)  # type: ignore[arg-type, return-value]

        return self.output

    def _prepare_state(self) -> None:
        # Fast-forward to the first step that needs to be generated.
        if self.min_prompt_len > 1:
            self._prefill()

        # Fan out the state to `num_prompts` x `num_gens`.
        if self.num_gens > 1:
            num_prompts = self.seqs.size(0)

            # (P)
            fan_out = torch.arange(num_prompts, device=self.seqs.device)

            # (P) -> (P x G)
            fan_out = repeat_interleave(fan_out, dim=0, repeat=self.num_gens)

            self._reorder_state(fan_out)

    def _prefill(self) -> None:
        chunk_begin = 0

        prefill_len = self.min_prompt_len

        while chunk_begin < prefill_len - 1:
            chunk_size = prefill_len - chunk_begin - 1

            # Decode by chunks of `prefill_chunk_size`.
            if self.prefill_chunk_size and chunk_size > self.prefill_chunk_size:
                chunk_size = self.prefill_chunk_size

            chunk_end = chunk_begin + chunk_size

            model_output = self._decode(self.seqs[:, chunk_begin:chunk_end])

            self.state_bag.increment_step_nr(chunk_size)

            logits = model_output.logits

            if self.temperature != 1.0:
                logits /= self.temperature

            # (P, S_prm - 1, V)
            probs = softmax(logits, dim=-1, dtype=torch.float32)

            if probs.isnan().any():
                raise RuntimeError(
                    "The model has produced one or more NaN probabilities during prefill. The sequence generator cannot continue."
                )

            if self.step_scores is not None:
                s = slice(chunk_begin + 1, chunk_end + 1)

                # Fetch the scores of the next prompt step.
                # (P, S_prm - 1, 1)
                prompt_scores = torch.gather(
                    probs, dim=-1, index=self.seqs[:, s].unsqueeze(-1)
                )

                # Bootstrap the step scores.
                # (P, S_prm - 1)
                self.step_scores[:, s] = prompt_scores.squeeze(-1)

            chunk_begin += chunk_size

        if self.step_hooks:
            seqs = self.seqs[:, :prefill_len]

            if self.step_scores is None:
                step_scores = None
            else:
                step_scores = self.step_scores[:, :prefill_len]

            for hook in self.step_hooks.values():
                hook(self.prompt_indices, seqs, step_scores, prefill=True)

    def _step(self) -> bool:
        # Generate the next step output.
        model_output = self._decode(self.seqs[:, self.step_nr - 1 : self.step_nr])

        self.state_bag.increment_step_nr()

        logits = model_output.logits

        if self.temperature != 1.0:
            logits /= self.temperature

        # (N, 1, V)
        probs = softmax(logits, dim=-1, dtype=torch.float32)

        # (N, 1, V) -> (N, V)
        probs.squeeze_(1)

        if probs.isnan().any():
            raise RuntimeError(
                f"The model has produced one or more NaN probabilities at step {self.step_nr}. The sequence generator cannot continue."
            )

        # If we are generating the last possible step, force it to be EOS
        # regardless of its score.
        if self.step_nr == self.max_seq_len - 1:
            batch_size = self.seqs.size(0)

            # (N)
            vocab_indices = self.seqs.new_full((batch_size,), self.eos_idx)
        else:
            # Process `probs` in-place if requested.
            for processor in self.step_processors:
                processor(self.seqs[:, : self.step_nr], probs)

            # Apply UNK penalty.
            if self.unk_idx is not None:
                probs[:, self.unk_idx] -= self.unk_penalty

            # Never allow PAD.
            if self.pad_idx is not None:
                probs[:, self.pad_idx] = 0

            # Do not allow EOS till we reach the minimum sequence length.
            if self.step_nr < self.min_seq_len - 1:
                probs[:, self.eos_idx] = 0

            # (N)
            vocab_indices = self.sampler(probs)

        # EOS mask of the current step.
        # (N)
        eos_mask = vocab_indices == self.eos_idx

        # Ignore the generated indices for the prompt sequences.
        if self.step_nr < self.max_prompt_len:
            assert self.prompt_mask is not None

            # (N)
            mask = self.prompt_mask[:, self.step_nr]

            # Override the generated indices.
            vocab_indices[mask] = self.seqs[mask, self.step_nr]

            # Ignore EOS in the prompt sequences.
            eos_mask[mask] = False
        else:
            self.prompt_mask = None  # Not needed anymore, release.

        # Record the current step.
        self.seqs[:, self.step_nr] = vocab_indices

        if self.step_scores is not None:
            # (N, 1)
            scores = torch.gather(probs, dim=-1, index=vocab_indices[:, None])

            # Record the scores of the current step.
            self.step_scores[:, self.step_nr] = scores.squeeze(1)

        if self.step_hooks:
            seqs = self.seqs[:, : self.step_nr + 1]

            if self.step_scores is None:
                step_scores = None
            else:
                step_scores = self.step_scores[:, : self.step_nr + 1]

            for hook in self.step_hooks.values():
                hook(self.prompt_indices, seqs, step_scores, prefill=False)

        # Retrieve the indices of the sequences that have reached EOS.
        # (F, 1)
        eos_seq_indices = eos_mask.nonzero()

        # If one or more sequences have reached EOS, move them to the output and
        # continue generating the remaining sequences.
        if len(eos_seq_indices) > 0:
            # Move the sequences that have reached EOS to the output.
            for seq_idx in eos_seq_indices:
                self._finish_sequence(int(seq_idx))

            # (N)
            active_seq_mask = ~eos_mask

            # (N - F, 1) -> (N - F)
            active_seq_indices = active_seq_mask.nonzero().squeeze(-1)

            # No sequence left, we can return.
            if len(active_seq_indices) == 0:
                return False

            # Otherwise, remove the sequences that have reached EOS from the
            # state and continue generating the remaining ones.
            self._reorder_state(active_seq_indices)

        return True

    @abstractmethod
    def _decode(self, seqs: Tensor) -> SequenceModelOutput:
        ...

    def _finish_sequence(self, seq_idx: int) -> None:
        if self.echo_prompt:
            start_step = 0
        else:
            if self.prompt_lens is None:
                start_step = self.max_prompt_len
            else:
                start_step = int(self.prompt_lens[seq_idx])

        seq_len = self.step_nr + 1

        # (S_out)
        seq = self.seqs[seq_idx, start_step:seq_len]

        # Do not keep `seqs` in memory.
        seq = seq.clone()

        if self.step_scores is not None:
            # (S)
            step_scores = self.step_scores[seq_idx, :seq_len]

            score = step_scores.sum()

            if self.normalize_scores:
                # Since the first step's score is always 0, do not include it in
                # the normalization.
                score /= (seq_len - 1) ** self.len_penalty

            # (S_out)
            step_scores = step_scores[start_step:]

            # Similar to `seqs`, do not keep `step_scores` in memory.
            step_scores = step_scores.clone()
        else:
            score = None

            step_scores = None

        prompt_idx = int(self.prompt_indices[seq_idx])

        self.output[prompt_idx].append(Hypothesis(seq, score, step_scores))

    def _reorder_state(self, new_order: Tensor) -> None:
        self.state_bag.reorder(new_order)

        # (N) -> (N - F)
        if self.prompt_lens is not None:
            self.prompt_lens = self.prompt_lens.index_select(dim=0, index=new_order)

        # (N, S_prm) -> (N - F, S_prm)
        if self.prompt_mask is not None:
            self.prompt_mask = self.prompt_mask.index_select(dim=0, index=new_order)

        # (N) -> (N - F)
        self.prompt_indices = self.prompt_indices.index_select(dim=0, index=new_order)

        # (N, S) -> (N - F, S)
        self.seqs = self.seqs.index_select(dim=0, index=new_order)

        # (N, S) -> (N - F, S)
        if self.step_scores is not None:
            self.step_scores = self.step_scores.index_select(dim=0, index=new_order)


class _SamplingSequenceGeneratorOp(_SamplingSequenceGeneratorOpBase):
    model: DecoderModel

    def __init__(
        self,
        model: DecoderModel,
        prompt_seqs: Tensor,
        prompt_padding_mask: Optional[PaddingMask],
        sampler: Sampler,
        num_gens: int,
        min_gen_len: int,
        max_gen_len: int,
        max_seq_len: int,
        echo_prompt: bool,
        compute_scores: bool,
        normalize_scores: bool,
        temperature: float,
        unk_penalty: float,
        len_penalty: float,
        prefill_chunk_size: Optional[int],
        decode_capacity_increment: Optional[int],
        step_processors: List[StepProcessor],
        step_hooks: Dict[int, StepHook],
    ) -> None:
        super().__init__(
            prompt_seqs,
            prompt_padding_mask,
            sampler,
            model.vocab_info,
            num_gens,
            min_gen_len,
            max_gen_len,
            max_seq_len,
            echo_prompt,
            compute_scores,
            normalize_scores,
            temperature,
            unk_penalty,
            len_penalty,
            prefill_chunk_size,
            decode_capacity_increment,
            step_processors,
            step_hooks,
        )

        self.model = model

    @override
    def _decode(self, seqs: Tensor) -> SequenceModelOutput:
        decoder_output, decoder_padding_mask = self.model.decode(
            seqs,
            None,  # We never use PAD in incremental decoding.
            state_bag=self.state_bag,
        )

        return self.model.project(decoder_output, decoder_padding_mask)


class _SamplingSeq2SeqGeneratorOp(_SamplingSequenceGeneratorOpBase):
    model: EncoderDecoderModel
    encoder_output: Tensor
    encoder_padding_mask: Optional[PaddingMask]

    def __init__(
        self,
        model: EncoderDecoderModel,
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
        prompt_seqs: Tensor,
        prompt_padding_mask: Optional[PaddingMask],
        sampler: Sampler,
        num_gens: int,
        min_gen_len: int,
        max_gen_len: int,
        max_seq_len: int,
        echo_prompt: bool,
        compute_scores: bool,
        normalize_scores: bool,
        temperature: float,
        unk_penalty: float,
        len_penalty: float,
        prefill_chunk_size: Optional[int],
        decode_capacity_increment: Optional[int],
        step_processors: List[StepProcessor],
        step_hooks: Dict[int, StepHook],
    ) -> None:
        super().__init__(
            prompt_seqs,
            prompt_padding_mask,
            sampler,
            model.target_vocab_info,
            num_gens,
            min_gen_len,
            max_gen_len,
            max_seq_len,
            echo_prompt,
            compute_scores,
            normalize_scores,
            temperature,
            unk_penalty,
            len_penalty,
            prefill_chunk_size,
            decode_capacity_increment,
            step_processors,
            step_hooks,
        )

        self.model = model
        self.encoder_output = encoder_output
        self.encoder_padding_mask = encoder_padding_mask

    @override
    def _decode(self, seqs: Tensor) -> SequenceModelOutput:
        decoder_output, decoder_padding_mask = self.model.decode(
            seqs,
            None,  # We never use PAD in incremental decoding.
            self.encoder_output,
            self.encoder_padding_mask,
            state_bag=self.state_bag,
        )

        return self.model.project(decoder_output, decoder_padding_mask)

    @override
    def _reorder_state(self, new_order: Tensor) -> None:
        super()._reorder_state(new_order)

        self.encoder_output = self.encoder_output.index_select(dim=0, index=new_order)

        if self.encoder_padding_mask is not None:
            encoder_seq_lens = self.encoder_padding_mask.seq_lens

            # (N) -> (N - F)
            encoder_seq_lens = encoder_seq_lens.index_select(dim=0, index=new_order)

            self.encoder_padding_mask = PaddingMask(
                encoder_seq_lens, batch_seq_len=self.encoder_output.size(1)
            )
