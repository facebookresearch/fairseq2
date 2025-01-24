# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol, final

from torch import Tensor
from torch.utils.hooks import RemovableHandle
from typing_extensions import override

from fairseq2.models.decoder import DecoderModel
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.nn.padding import PaddingMask


class SequenceGenerator(ABC):
    """Represents a sequence generator."""

    @abstractmethod
    def __call__(
        self, prompt_seqs: Tensor, prompt_padding_mask: PaddingMask | None
    ) -> SequenceGeneratorOutput:
        """
        :param prompt_seqs:
            The prompt sequences. *Shape:* :math:`(N,S)`, where :math:`N` is the
            batch size and :math:`S` is the sequence length.
        :param prompt_padding_mask:
            The padding mask of ``prompt_seqs``. *Shape:* Same as ``prompt_seqs``.
        """

    @abstractmethod
    def register_step_hook(self, hook: StepHook) -> RemovableHandle:
        """Register a step hook on the sequence generator.

        The hook will be called after every generation step.

        :param hook:
            The hook to register.

        :returns:
            A handle that can be used to remove the added hook by calling
            ``handle.remove()``.
        """

    @property
    @abstractmethod
    def model(self) -> DecoderModel:
        """The associated decoder model."""


@dataclass
class SequenceGeneratorOutput:
    """Holds the output of a sequence generator."""

    hypotheses: list[list[Hypothesis]]
    """The list of hypothesis generated per prompt, ordered by score."""

    counters: GenerationCounters
    """The performance counters of the call."""


@dataclass
class Hypothesis:
    """Represents a hypothesis produced by a sequence generator."""

    seq: Tensor
    """The generated sequence. *Shape:* :math:`(S)`, where :math:`S` is the
    sequence length."""

    score: Tensor | None
    """The score of the hypothesis. *Shape:* Scalar."""

    step_scores: Tensor | None
    """The score of each sequence step. *Shape:* Same as ``seq``."""


@dataclass
class GenerationCounters:
    """Holds the performance counters of a generator call."""

    prefill_size: int = 0
    """The number of elements processed during the prefill step."""

    num_generated_elements: int = 0
    """The number of generated elements."""

    generation_time: float = 0
    """The generation time excluding prefill."""

    cache_size: int = 0
    """The final size of the incremental cache in bytes."""

    cache_capacity: int = 0
    """The final reserved capacity of the incremental cache in bytes."""


class SequenceGenerationError(Exception):
    pass


class AbstractSequenceGenerator(SequenceGenerator):
    """Provides a skeletal implementation of :class:`SequenceGenerator`."""

    _model: DecoderModel
    _step_hooks: dict[int, StepHook]

    def __init__(self, model: DecoderModel) -> None:
        """
        :param model:
            The decoder model to use for generation.
        """
        if model.vocab_info.eos_idx is None:
            raise ValueError(
                "`model.vocab_info` must have `eos_idx` set for sequence generation."
            )

        model.eval()

        self._model = model

        self._step_hooks = OrderedDict()

    @final
    @override
    def register_step_hook(self, hook: StepHook) -> RemovableHandle:
        handle = RemovableHandle(self._step_hooks)

        self._step_hooks[handle.id] = hook

        return handle

    @final
    @property
    @override
    def model(self) -> DecoderModel:
        return self._model


class Seq2SeqGenerator(ABC):
    """Represents a sequence-to-sequence generator."""

    @abstractmethod
    def __call__(
        self,
        source_seqs: Tensor,
        source_padding_mask: PaddingMask | None,
        prompt_seqs: Tensor,
        prompt_padding_mask: PaddingMask | None,
    ) -> Seq2SeqGeneratorOutput:
        """
        :param source_seqs:
            The source sequences. *Shape:* :math:`(N,S,*)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`*` is
            any number of sequence-specific dimensions including none.
        :param source_padding_mask:
            The padding mask of ``source_seqs``. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math:`S` is the sequence length.
        :param prompt_seqs:
            The prompt sequences. *Shape:* :math:`(N,S_{prm})`, where :math:`N`
            is the batch size and :math:`S_{prm}` is the prompt sequence length.
        :param prompt_padding_mask:
            The padding mask of ``prompt_seqs``. *Shape:* Same as ``prompt_seqs``.
        """

    @abstractmethod
    def register_step_hook(self, hook: StepHook) -> RemovableHandle:
        """Register a step hook on the sequence-to-sequence generator.

        The hook will be called after every generation step.

        :param hook:
            The hook to register.

        :returns:
            A handle that can be used to remove the added hook by calling
            ``handle.remove()``.
        """

    @property
    @abstractmethod
    def model(self) -> EncoderDecoderModel:
        """The associated encoder-decoder model."""


@dataclass
class Seq2SeqGeneratorOutput:
    hypotheses: list[list[Hypothesis]]
    """The list of hypothesis generated per prompt, ordered by score."""

    encoder_output: Tensor
    """The encoder output used in encoder-decoder attention. *Shape:*
    :math:`(N,S_{enc},M)`, where :math:`N` is the batch size, :math:`S_{enc}` is
    the encoder output sequence length, and :math:`M` is the dimensionality of
    the model."""

    encoder_padding_mask: PaddingMask | None
    """The padding mask of :attr:`encoder_output`. *Shape:* :math:`(N,S_{enc})`,
    where :math:`N` is the batch size and :math:`S_{enc}` is the encoder output
    sequence length."""

    counters: GenerationCounters
    """The performance counters of the call."""


class AbstractSeq2SeqGenerator(Seq2SeqGenerator):
    """Provides a skeletal implementation of :class:`Seq2SeqGenerator`."""

    _model: EncoderDecoderModel
    _step_hooks: dict[int, StepHook]

    def __init__(self, model: EncoderDecoderModel) -> None:
        """
        :param model:
            The encoder-decoder model to use for generation.
        """
        if model.target_vocab_info.eos_idx is None:
            raise ValueError(
                "`model.vocab_info` must have `eos_idx` set for sequence generation."
            )

        model.eval()

        self._model = model

        self._step_hooks = OrderedDict()

    @final
    @override
    def register_step_hook(self, hook: StepHook) -> RemovableHandle:
        handle = RemovableHandle(self._step_hooks)

        self._step_hooks[handle.id] = hook

        return handle

    @final
    @property
    @override
    def model(self) -> EncoderDecoderModel:
        return self._model


class StepHook(Protocol):
    """Represents a hook to pass to :meth:`~SequenceGenerator.register_step_hook`
    or :meth:`~Seq2SeqGenerator.register_step_hook`."""

    def __call__(
        self,
        prompt_indices: Tensor,
        seqs: Tensor,
        step_scores: Tensor | None,
        prefill: bool,
    ) -> None:
        """
        :param prompt_indices:
            The indices of the input prompts corresponding to each sequence in
            ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is the batch size.
        :param seqs:
            The sequences that are in process of being generated. *Shape:*
            :math:`(N,S)`, where :math:`N` is the batch size and :math:`S` is
            the sequence length generated so far.
        :param step_scores:
            The score of each step in ``seqs``. *Shape:* Same as ``seqs``.
        :param prefill:
            If ``True``, the hook is called as part of prompt prefill.
        """
