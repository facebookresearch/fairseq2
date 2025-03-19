# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from typing import final

import torch
from torch import Tensor
from torch.nn.functional import log_softmax
from torch.utils.hooks import RemovableHandle
from typing_extensions import override

from fairseq2.data import VocabularyInfo
from fairseq2.error import InternalError
from fairseq2.generation._beam_search._algo import (
    BeamSearchAlgorithm,
    BeamStep,
    StandardBeamSearchAlgorithm,
)
from fairseq2.generation._generator import (
    GenerationCounters,
    Hypothesis,
    Seq2SeqGenerator,
    Seq2SeqGeneratorOutput,
    SequenceGenerationError,
    SequenceGenerator,
    SequenceGeneratorOutput,
    StepHook,
)
from fairseq2.generation._step_processor import StepProcessor
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.nn import IncrementalStateBag
from fairseq2.nn.padding import PaddingMask
from fairseq2.utils.stopwatch import Stopwatch


@final
class BeamSearchSequenceGenerator(SequenceGenerator):
    """Represents a sequence generator based on beam search."""

    _model: DecoderModel
    _algorithm: BeamSearchAlgorithm
    _beam_size: int
    _min_gen_len: int
    _max_gen_len: int
    _max_seq_len: int
    _echo_prompt: bool
    _normalize_scores: bool
    _temperature: float
    _unk_penalty: float
    _len_penalty: float
    _prefill_chunk_size: int | None
    _decode_capacity_increment: int | None
    _step_processors: Sequence[StepProcessor]
    _step_hooks: dict[int, StepHook]

    def __init__(
        self,
        model: DecoderModel,
        *,
        algorithm: BeamSearchAlgorithm | None = None,
        beam_size: int = 5,
        min_gen_len: int = 1,
        max_gen_len: int = 128,
        max_seq_len: int | None = None,
        echo_prompt: bool = False,
        normalize_scores: bool = True,
        temperature: float = 1.0,
        unk_penalty: float = 0.0,
        len_penalty: float = 1.0,
        prefill_chunk_size: int | None = 512,
        decode_capacity_increment: int | None = 16,
        step_processors: Sequence[StepProcessor] | None = None,
    ) -> None:
        """
        :param model:
            The decoder model to use for generation.
        :param algorithm:
            The beam search algorithm.
        :param beam_size:
            The beam size.
        :param min_gen_len:
            The minimum generation length.
        :param max_gen_len:
            The maximum generation length.
        :param max_seq_len:
            The maximum sequence length including prompt.
        :param echo_prompt:
            If ``True``, returns generated sequences with prompts appended.
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
        if model.vocab_info.eos_idx is None:
            raise ValueError(
                "`model.vocab_info` must have `eos_idx` set for sequence generation."
            )

        model.eval()

        self._model = model

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

        if max_seq_len is None:
            effective_max_seq_len = model.max_seq_len
        else:
            effective_max_seq_len = min(max_seq_len, model.max_seq_len)

        self._algorithm = algorithm or StandardBeamSearchAlgorithm()
        self._beam_size = beam_size
        self._min_gen_len = min_gen_len
        self._max_gen_len = max_gen_len
        self._max_seq_len = effective_max_seq_len
        self._echo_prompt = echo_prompt
        self._normalize_scores = normalize_scores
        self._temperature = temperature
        self._unk_penalty = unk_penalty
        self._len_penalty = len_penalty
        self._prefill_chunk_size = prefill_chunk_size
        self._decode_capacity_increment = decode_capacity_increment

        if step_processors:
            self._step_processors = step_processors
        else:
            self._step_processors = []

        self._step_hooks = OrderedDict()

    @torch.inference_mode()
    @override
    def __call__(
        self, prompt_seqs: Tensor, prompt_padding_mask: PaddingMask | None
    ) -> SequenceGeneratorOutput:
        op = _BeamSearchSequenceGeneratorOp(
            self._model,
            prompt_seqs,
            prompt_padding_mask,
            self._algorithm,
            self._beam_size,
            self._min_gen_len,
            self._max_gen_len,
            self._max_seq_len,
            self._echo_prompt,
            self._normalize_scores,
            self._temperature,
            self._unk_penalty,
            self._len_penalty,
            self._prefill_chunk_size,
            self._decode_capacity_increment,
            self._step_processors,
            self._step_hooks,
        )

        hypotheses, counters = op()

        return SequenceGeneratorOutput(hypotheses, counters)

    @override
    def register_step_hook(self, hook: StepHook) -> RemovableHandle:
        handle = RemovableHandle(self._step_hooks)

        self._step_hooks[handle.id] = hook

        return handle

    @property
    @override
    def model(self) -> DecoderModel:
        return self._model


@final
class BeamSearchSeq2SeqGenerator(Seq2SeqGenerator):
    """Represents a sequence-to-sequence generator based on beam search."""

    _model: EncoderDecoderModel
    _algorithm: BeamSearchAlgorithm
    _beam_size: int
    _min_gen_len: int
    _max_gen_len: tuple[int, int]
    _max_seq_len: int
    _echo_prompt: bool
    _normalize_scores: bool
    _temperature: float
    _unk_penalty: float
    _len_penalty: float
    _prefill_chunk_size: int | None
    _decode_capacity_increment: int | None
    _step_processors: Sequence[StepProcessor]
    _step_hooks: dict[int, StepHook]

    def __init__(
        self,
        model: EncoderDecoderModel,
        *,
        algorithm: BeamSearchAlgorithm | None = None,
        beam_size: int = 5,
        min_gen_len: int = 1,
        max_gen_len: tuple[int, int] = (1, 128),
        max_seq_len: int | None = None,
        echo_prompt: bool = False,
        normalize_scores: bool = True,
        temperature: float = 1.0,
        unk_penalty: float = 0.0,
        len_penalty: float = 1.0,
        prefill_chunk_size: int | None = 512,
        decode_capacity_increment: int | None = 16,
        step_processors: Sequence[StepProcessor] | None = None,
    ) -> None:
        """
        :param model:
            The encoder-decoder model to use for generation.
        :param algorithm:
            The beam search algorithm.
        :param beam_size:
            The beam size.
        :param min_gen_len:
            The minimum generation length.
        :param max_gen_len:
            The maximum generation length as ``ax + b``, where ``x`` is the
            source sequence length.
        :param max_seq_len:
            The maximum sequence length including prompt.
        :param echo_prompt:
            If ``True``, returns generated sequences with prompts appended.
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
        if model.target_vocab_info.eos_idx is None:
            raise ValueError(
                "`model.vocab_info` must have `eos_idx` set for sequence generation."
            )

        model.eval()

        self._model = model

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

        if max_seq_len is None:
            effective_max_seq_len = model.max_target_seq_len
        else:
            effective_max_seq_len = min(max_seq_len, model.max_target_seq_len)

        self._algorithm = algorithm or StandardBeamSearchAlgorithm()
        self._beam_size = beam_size
        self._min_gen_len = min_gen_len
        self._max_gen_len = max_gen_len
        self._max_seq_len = effective_max_seq_len
        self._echo_prompt = echo_prompt
        self._normalize_scores = normalize_scores
        self._temperature = temperature
        self._unk_penalty = unk_penalty
        self._len_penalty = len_penalty
        self._prefill_chunk_size = prefill_chunk_size
        self._decode_capacity_increment = decode_capacity_increment

        if step_processors:
            self._step_processors = step_processors
        else:
            self._step_processors = []

        self._step_hooks = OrderedDict()

    @torch.inference_mode()
    @override
    def __call__(
        self,
        source_seqs: Tensor,
        source_padding_mask: PaddingMask | None,
        prompt_seqs: Tensor,
        prompt_padding_mask: PaddingMask | None,
    ) -> Seq2SeqGeneratorOutput:
        # (P, S)
        encoder_output, encoder_padding_mask = self.model.encode(
            source_seqs, source_padding_mask
        )

        if source_padding_mask is None:
            max_source_len = source_seqs.size(1)
        else:
            max_source_len = int(source_padding_mask.seq_lens.max())

        a_term, b_term = self._max_gen_len

        # In seq2seq generation, the maximum generation length is relative to
        # the source sequence length.
        max_gen_len = int(a_term * max_source_len + b_term)

        if max_gen_len < 1:
            raise ValueError(
                f"`max_gen_len` must be greater than or equal to 1, but is {max_gen_len} instead. Adjust your `max_gen_len` argument."
            )

        if self._min_gen_len > max_gen_len:
            raise ValueError(
                f"`min_gen_len` must be less than or equal to `max_gen_len` ({max_gen_len}), but is {self._min_gen_len} instead. Adjust your `max_gen_len` argument."
            )

        op = _BeamSearchSeq2SeqGeneratorOp(
            self._model,
            encoder_output,
            encoder_padding_mask,
            prompt_seqs,
            prompt_padding_mask,
            self._algorithm,
            self._beam_size,
            self._min_gen_len,
            max_gen_len,
            self._max_seq_len,
            self._echo_prompt,
            self._normalize_scores,
            self._temperature,
            self._unk_penalty,
            self._len_penalty,
            self._prefill_chunk_size,
            self._decode_capacity_increment,
            self._step_processors,
            self._step_hooks,
        )

        hypotheses, counters = op()

        return Seq2SeqGeneratorOutput(
            hypotheses, encoder_output, encoder_padding_mask, counters
        )

    @override
    def register_step_hook(self, hook: StepHook) -> RemovableHandle:
        handle = RemovableHandle(self._step_hooks)

        self._step_hooks[handle.id] = hook

        return handle

    @property
    @override
    def model(self) -> EncoderDecoderModel:
        return self._model


class _AbstractBeamSearchSequenceGeneratorOp(ABC):
    _algorithm: BeamSearchAlgorithm
    _eos_idx: int
    _pad_idx: int | None
    _unk_idx: int | None
    _beam_size: int
    _min_prompt_len: int
    _max_prompt_len: int
    _min_seq_len: int
    _max_seq_len: int
    _echo_prompt: bool
    _normalize_scores: bool
    _temperature: float
    _unk_penalty: float
    _len_penalty: float
    _prefill_chunk_size: int | None
    _step_processors: Sequence[StepProcessor]
    _step_hooks: dict[int, StepHook]
    _step_nr: int
    _state_bag: IncrementalStateBag
    _prompt_lens: Tensor | None
    _prompt_mask: Tensor | None
    _beam_sizes: list[int]
    _prompt_indices: Tensor
    _seqs: Tensor
    _step_scores: Tensor
    _output: list[list[Hypothesis]]
    _watch: Stopwatch
    _counters: GenerationCounters

    def __init__(
        self,
        prompt_seqs: Tensor,
        prompt_padding_mask: PaddingMask | None,
        algorithm: BeamSearchAlgorithm,
        vocab_info: VocabularyInfo,
        beam_size: int,
        min_gen_len: int,
        max_gen_len: int,
        max_seq_len: int,
        echo_prompt: bool,
        normalize_scores: bool,
        temperature: float,
        unk_penalty: float,
        len_penalty: float,
        prefill_chunk_size: int | None,
        decode_capacity_increment: int | None,
        step_processors: Sequence[StepProcessor],
        step_hooks: dict[int, StepHook],
    ) -> None:
        self._algorithm = algorithm

        if vocab_info.eos_idx is None:
            raise InternalError("`vocab_info.eos_idx` is `None`.")

        self._eos_idx = vocab_info.eos_idx
        self._pad_idx = vocab_info.pad_idx
        self._unk_idx = vocab_info.unk_idx

        self._beam_size = beam_size

        min_prompt_idx: int | Tensor
        max_prompt_idx: int | Tensor

        if prompt_padding_mask is None:
            self._min_prompt_len, min_prompt_idx = prompt_seqs.size(1), 0
            self._max_prompt_len, max_prompt_idx = prompt_seqs.size(1), 0
        else:
            prompt_seq_lens = prompt_padding_mask.seq_lens

            min_prompt_len, min_prompt_idx = torch.min(prompt_seq_lens, dim=0)
            max_prompt_len, max_prompt_idx = torch.max(prompt_seq_lens, dim=0)

            self._min_prompt_len = int(min_prompt_len)
            self._max_prompt_len = int(max_prompt_len)

            if self._min_prompt_len == self._max_prompt_len:
                prompt_padding_mask = None

        if self._min_prompt_len < 1:
            raise ValueError(f"`prompt_seqs[{int(min_prompt_idx)}]` must not be empty.")

        if self._max_prompt_len >= max_seq_len:
            raise ValueError(
                f"The length of `prompt_seqs[{int(max_prompt_idx)}]` must be less than `max_seq_len` ({max_seq_len}), but is {self._max_prompt_len} instead."
            )

        self._min_seq_len = min(max_seq_len, self._max_prompt_len + min_gen_len)
        self._max_seq_len = min(max_seq_len, self._max_prompt_len + max_gen_len)

        self._echo_prompt = echo_prompt
        self._normalize_scores = normalize_scores
        self._temperature = temperature
        self._unk_penalty = unk_penalty
        self._len_penalty = len_penalty
        self._prefill_chunk_size = prefill_chunk_size
        self._step_processors = step_processors
        self._step_hooks = step_hooks

        self._step_nr = 0

        self._state_bag = IncrementalStateBag(
            self._max_seq_len, capacity_increment=decode_capacity_increment
        )

        if prompt_padding_mask is None:
            self._prompt_lens = None
            self._prompt_mask = None
        else:
            # (P)
            self._prompt_lens = prompt_padding_mask.seq_lens

            # (P, S_prm)
            self._prompt_mask = prompt_padding_mask.materialize()

        device = prompt_seqs.device

        num_prompts = prompt_seqs.size(0)

        # Holds the sizes of the beams.
        self._beam_sizes = [1 for _ in range(num_prompts)]

        # Holds the prompt indices of the generated sequences.
        # (P)
        self._prompt_indices = torch.arange(num_prompts, device=device)

        # Holds the generated sequences.
        # (P, S)
        self._seqs = torch.empty(
            (num_prompts, self._max_seq_len), device=device, dtype=torch.int64
        )

        # Holds the step scores of the generated sequences.
        # (P, S)
        self._step_scores = torch.zeros(
            (num_prompts, self._max_seq_len), device=device, dtype=torch.float32
        )

        # Bootstrap the sequences.
        self._seqs[:, : self._max_prompt_len] = prompt_seqs[:, : self._max_prompt_len]

        # Holds the sequences that have reached EOS.
        self._output = [[] for _ in range(num_prompts)]

        self._watch = Stopwatch(device=self._seqs.device)

        self._counters = GenerationCounters()

    def __call__(self) -> tuple[list[list[Hypothesis]], GenerationCounters]:
        self._prepare_state()

        self._watch.start()

        for self._step_nr in range(self._min_prompt_len, self._max_seq_len):
            if not self._step():
                break

        self._watch.stop()

        self._counters.generation_time = self._watch.get_elapsed_time()

        self._watch.reset()

        self._counters.cache_size = self._state_bag.size_bytes()
        self._counters.cache_capacity = self._state_bag.capacity_bytes()

        # Sort the hypotheses by their scores before returning.
        for hypotheses in self._output:
            hypotheses.sort(key=lambda h: h.score, reverse=True)  # type: ignore[arg-type, return-value]

        return self._output, self._counters

    def _prepare_state(self) -> None:
        # Fast-forward to the first step that needs to be generated.
        if self._min_prompt_len > 1:
            self._prefill()

    def _prefill(self) -> None:
        chunk_begin = 0

        prefill_len = self._min_prompt_len

        while chunk_begin < prefill_len - 1:
            chunk_size = prefill_len - chunk_begin - 1

            # Decode by chunks of `prefill_chunk_size`.
            if self._prefill_chunk_size and chunk_size > self._prefill_chunk_size:
                chunk_size = self._prefill_chunk_size

            chunk_end = chunk_begin + chunk_size

            model_output = self._decode(self._seqs[:, chunk_begin:chunk_end])

            self._state_bag.increment_step_nr(chunk_size)

            logits = model_output.logits

            if self._temperature != 1.0:
                logits /= self._temperature

            # (P, S_prm - 1, V)
            lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

            if lprobs.isnan().any():
                raise SequenceGenerationError(
                    "The model has produced one or more NaN probabilities during prefill. The sequence generator cannot continue."
                )

            s = slice(chunk_begin + 1, chunk_end + 1)

            # Fetch the scores of the next prompt step.
            # (P, S_prm - 1, 1)
            prompt_scores = torch.gather(
                lprobs, dim=-1, index=self._seqs[:, s].unsqueeze(-1)
            )

            # (P, S_prm - 1, 1) -> (P, S_prm - 1)
            prompt_scores.squeeze_(-1).cumsum_(dim=-1)

            prompt_scores += self._step_scores[:, chunk_begin].unsqueeze(-1)

            # Bootstrap the step scores.
            # (P x B, S_prm - 1)
            self._step_scores[:, s] = prompt_scores

            chunk_begin += chunk_size

        if self._step_hooks:
            seqs = self._seqs[:, :prefill_len]

            step_scores = self._step_scores[:, :prefill_len]

            for hook in self._step_hooks.values():
                hook(self._prompt_indices, seqs, step_scores, prefill=True)

        self._counters.prefill_size += prefill_len * self._seqs.size(0)

    def _step(self) -> bool:
        # Generate the next step output.
        model_output = self._decode(self._seqs[:, self._step_nr - 1 : self._step_nr])

        self._state_bag.increment_step_nr()

        self._counters.num_generated_elements += self._seqs.size(0)

        logits = model_output.logits

        if self._temperature != 1.0:
            logits /= self._temperature

        # (N, 1, V)
        lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

        # (N, 1, V) -> (N, V)
        lprobs.squeeze_(1)

        if lprobs.isnan().any():
            raise SequenceGenerationError(
                f"The model has produced one or more NaN probabilities at step {self._step_nr}. The sequence generator cannot continue."
            )

        # If we are generating the last possible step, force it to be EOS
        # regardless of its score.
        if self._step_nr == self._max_seq_len - 1:
            lprobs[:, : self._eos_idx]       = -torch.inf  # fmt: skip
            lprobs[:,   self._eos_idx + 1 :] = -torch.inf  # fmt: skip
        else:
            # Process `lprobs` in-place if requested.
            for processor in self._step_processors:
                processor(self._seqs[:, : self._step_nr], lprobs, lprob=True)

            # Apply UNK penalty.
            if self._unk_idx is not None:
                lprobs[:, self._unk_idx] -= self._unk_penalty

            # Never allow PAD.
            if self._pad_idx is not None:
                lprobs[:, self._pad_idx] = -torch.inf

            # Do not allow EOS till we reach the minimum sequence length.
            if self._step_nr < self._min_seq_len - 1:
                lprobs[:, self._eos_idx] = -torch.inf

        batch_offset = 0

        new_beam_sizes: list[int] = []

        beam_next_step_list: list[BeamStep] = []

        # We split the batch by `beam_sizes` and treat each beam separately.
        for beam_idx, (beam_lprobs, beam_step_scores) in enumerate(
            zip(lprobs.split(self._beam_sizes), self._step_scores.split(self._beam_sizes))  # fmt: skip
        ):
            beam_next_step = self._search_beam(
                beam_idx, batch_offset, beam_lprobs, beam_step_scores
            )

            # Bump the beam batch offset to the next beam.
            batch_offset += self._beam_sizes[beam_idx]

            # Check if the beam is terminated.
            if beam_next_step is None:
                continue

            beam_size = len(beam_next_step.seq_indices)

            # We should have terminated the beam if there are no sequences.
            if beam_size == 0:
                raise InternalError("`beam_size` is zero.")

            new_beam_sizes.append(beam_size)

            beam_next_step_list.append(beam_next_step)

        # No beam left, we can return.
        if len(new_beam_sizes) == 0:
            return False

        self._beam_sizes = new_beam_sizes

        # (N_new)
        next_step = BeamStep.merge(beam_next_step_list)

        self._reorder_state(next_step.seq_indices)

        # Record the current step.
        self._seqs[:, self._step_nr] = next_step.vocab_indices

        # Record the scores of the current step.
        self._step_scores[:, self._step_nr] = next_step.scores

        if self._step_hooks:
            seqs = self._seqs[:, : self._step_nr + 1]

            step_scores = self._step_scores[:, : self._step_nr + 1]

            for hook in self._step_hooks.values():
                hook(self._prompt_indices, seqs, step_scores, prefill=False)

        return True

    def _search_beam(
        self, beam_idx: int, batch_offset: int, lprobs: Tensor, step_scores: Tensor
    ) -> BeamStep | None:
        # Ignore the generated indices for the prompt sequences.
        if self._step_nr < self._max_prompt_len:
            if self._prompt_mask is None:
                raise InternalError("`_prompt_mask` is `None`.")

            # Check if the current beam is in a prompt sequence.
            if self._prompt_mask[batch_offset, self._step_nr]:
                # The size of a beam in a prompt sequence must be always 1.
                if len(lprobs) != 1:
                    raise InternalError(f"The length of `lprobs` is {len(lprobs)}.")

                seq_idx = torch.tensor([batch_offset], device=lprobs.device)

                # We just extract the prompt step along with its score and treat
                # it as the next beam step. So we keep a beam of size 1 until we
                # reach the end of the prompt.
                vocab_idx = self._seqs[batch_offset, self._step_nr : self._step_nr + 1]

                score = step_scores[0, self._step_nr - 1] + lprobs[0, vocab_idx]

                return BeamStep(seq_idx, vocab_idx, score)
        else:
            self._prompt_mask = None  # Not needed anymore, release.

        # We use the same beam search method as in fairseq, where we take the
        # best 2 x `beam_size` candidates and choose the first `beam_size` of
        # these which don't predict EOS to continue with.
        # (2 x B)
        next_step = self._algorithm.step(
            self._beam_size, lprobs, step_scores[:, : self._step_nr]
        )

        # Translate the sequence indices from beam to batch.
        next_step.seq_indices += batch_offset

        # (2 x B)
        eos_mask = next_step.vocab_indices == self._eos_idx

        # Consider EOS only when it's among the top `beam_size` indices.
        # (F)
        eos_seq_indices = next_step.seq_indices[: self._beam_size].masked_select(
            eos_mask[: self._beam_size]
        )

        # If one or more sequences have reached EOS, move them to the output.
        if len(eos_seq_indices) > 0:
            # (F)
            eos_scores = next_step.scores[: self._beam_size].masked_select(
                eos_mask[: self._beam_size]
            )

            for seq_idx, score in zip(eos_seq_indices, eos_scores):
                # If `True`, it means we have found `beam_size` hypotheses for
                # this beam.
                if self._finish_sequence(int(seq_idx), score):
                    return None

            # Filter out the sequences that have reached EOS.
            seq_mask = ~eos_mask

            next_step = next_step.masked_select(seq_mask)

        # We can have at most `beam_size` sequences in the beam.
        return next_step.first(self._beam_size)

    @abstractmethod
    def _decode(self, seqs: Tensor) -> SequenceModelOutput: ...

    def _finish_sequence(self, seq_idx: int, score: Tensor) -> bool:
        self._seqs[seq_idx, self._step_nr] = self._eos_idx

        self._step_scores[seq_idx, self._step_nr] = score

        if self._echo_prompt:
            start_step = 0
        else:
            if self._prompt_lens is None:
                start_step = self._max_prompt_len
            else:
                start_step = int(self._prompt_lens[seq_idx])

        seq_len = self._step_nr + 1

        # (S_out)
        seq = self._seqs[seq_idx, start_step:seq_len]

        # Do not keep `seqs` in memory.
        seq = seq.clone()

        # (S_out)
        step_scores = self._step_scores[seq_idx, start_step:seq_len]

        # Similar to `seqs`, do not keep `step_scores` in memory.
        step_scores = step_scores.clone()

        # Convert from cumulative to per-step scores.
        step_scores[1:] = step_scores[1:] - step_scores[:-1]

        if self._normalize_scores:
            # Since the first step's score is always 0, do not include it in
            # the normalization.
            score /= (seq_len - 1) ** self._len_penalty

        prompt_idx = int(self._prompt_indices[seq_idx])

        hypotheses = self._output[prompt_idx]

        hypotheses.append(Hypothesis(seq, score, step_scores))

        # If we have `beam_size` hypotheses for the prompt, we can remove the
        # beam.
        return len(hypotheses) == self._beam_size

    def _reorder_state(self, new_order: Tensor) -> None:
        self._state_bag.reorder(new_order)

        # (N) -> (N - F)
        if self._prompt_lens is not None:
            self._prompt_lens = self._prompt_lens.index_select(dim=0, index=new_order)

        # (N, S_prm) -> (N - F, S_prm)
        if self._prompt_mask is not None:
            self._prompt_mask = self._prompt_mask.index_select(dim=0, index=new_order)

        # (N) -> (N - F)
        self._prompt_indices = self._prompt_indices.index_select(dim=0, index=new_order)

        # (N, S) -> (N - F, S)
        self._seqs = self._seqs.index_select(dim=0, index=new_order)

        # (N, S) -> (N - F, S)
        self._step_scores = self._step_scores.index_select(dim=0, index=new_order)


@final
class _BeamSearchSequenceGeneratorOp(_AbstractBeamSearchSequenceGeneratorOp):
    _model: DecoderModel

    def __init__(
        self,
        model: DecoderModel,
        prompt_seqs: Tensor,
        prompt_padding_mask: PaddingMask | None,
        algorithm: BeamSearchAlgorithm,
        beam_size: int,
        min_gen_len: int,
        max_gen_len: int,
        max_seq_len: int,
        echo_prompt: bool,
        normalize_scores: bool,
        temperature: float,
        unk_penalty: float,
        len_penalty: float,
        prefill_chunk_size: int | None,
        decode_capacity_increment: int | None,
        step_processors: Sequence[StepProcessor],
        step_hooks: dict[int, StepHook],
    ) -> None:
        super().__init__(
            prompt_seqs,
            prompt_padding_mask,
            algorithm,
            model.vocab_info,
            beam_size,
            min_gen_len,
            max_gen_len,
            max_seq_len,
            echo_prompt,
            normalize_scores,
            temperature,
            unk_penalty,
            len_penalty,
            prefill_chunk_size,
            decode_capacity_increment,
            step_processors,
            step_hooks,
        )

        self._model = model

    @override
    def _decode(self, seqs: Tensor) -> SequenceModelOutput:
        decoder_output, decoder_padding_mask = self._model.decode(
            seqs,
            None,  # We never use PAD in incremental decoding.
            state_bag=self._state_bag,
        )

        return self._model.project(decoder_output, decoder_padding_mask)


@final
class _BeamSearchSeq2SeqGeneratorOp(_AbstractBeamSearchSequenceGeneratorOp):
    _model: EncoderDecoderModel
    _encoder_output: Tensor
    _encoder_padding_mask: PaddingMask | None

    def __init__(
        self,
        model: EncoderDecoderModel,
        encoder_output: Tensor,
        encoder_padding_mask: PaddingMask | None,
        prompt_seqs: Tensor,
        prompt_padding_mask: PaddingMask | None,
        algorithm: BeamSearchAlgorithm,
        beam_size: int,
        min_gen_len: int,
        max_gen_len: int,
        max_seq_len: int,
        echo_prompt: bool,
        normalize_scores: bool,
        temperature: float,
        unk_penalty: float,
        len_penalty: float,
        prefill_chunk_size: int | None,
        decode_capacity_increment: int | None,
        step_processors: Sequence[StepProcessor],
        step_hooks: dict[int, StepHook],
    ) -> None:
        super().__init__(
            prompt_seqs,
            prompt_padding_mask,
            algorithm,
            model.target_vocab_info,
            beam_size,
            min_gen_len,
            max_gen_len,
            max_seq_len,
            echo_prompt,
            normalize_scores,
            temperature,
            unk_penalty,
            len_penalty,
            prefill_chunk_size,
            decode_capacity_increment,
            step_processors,
            step_hooks,
        )

        self._model = model
        self._encoder_output = encoder_output
        self._encoder_padding_mask = encoder_padding_mask

    @override
    def _decode(self, seqs: Tensor) -> SequenceModelOutput:
        decoder_output, decoder_padding_mask = self._model.decode(
            seqs,
            None,  # We never use PAD in incremental decoding.
            self._encoder_output,
            self._encoder_padding_mask,
            state_bag=self._state_bag,
        )

        return self._model.project(decoder_output, decoder_padding_mask)

    @override
    def _reorder_state(self, new_order: Tensor) -> None:
        super()._reorder_state(new_order)

        self._encoder_output = self._encoder_output.index_select(dim=0, index=new_order)

        if self._encoder_padding_mask is not None:
            encoder_seq_lens = self._encoder_padding_mask.seq_lens

            # (N) -> (N - F)
            encoder_seq_lens = encoder_seq_lens.index_select(dim=0, index=new_order)

            self._encoder_padding_mask = PaddingMask(
                encoder_seq_lens, batch_seq_len=self._encoder_output.size(1)
            )
