# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

from torch import Tensor

from fairseq2.chatbots import Chatbot, ChatMessage
from fairseq2.data.tokenizers import TokenDecoder, Tokenizer
from fairseq2.dependency import DependencyResolver
from fairseq2.device import CPU
from fairseq2.error import InternalError
from fairseq2.gang import Gang, Gangs
from fairseq2.generation import SequenceGenerator
from fairseq2.logging import log
from fairseq2.recipe.config import (
    CommonSection,
    GangSection,
    GeneratorSection,
    ReferenceModelSection,
    SequenceGeneratorSection,
    TokenizerSection,
)
from fairseq2.recipe.rng import manual_seed
from fairseq2.recipe.torch import configure_torch
from fairseq2.recipe.utils.log import log_environment_info
from fairseq2.utils.rng import RngBag, SeedHolder

import os
from fairseq2 import setup_fairseq2
from fairseq2.utils.device import DeviceDetectionError, determine_default_device
from fairseq2.error import SetupError
from fairseq2.gang import ProcessGroupGang, create_parallel_gangs
from fairseq2.device import CPU
from fairseq2.utils.rng import RngBag
from fairseq2.logging import log


def run_program(view: ProgramView) -> None:
    env = os.environ

    device = determine_default_device(env)

    rng_bag = RngBag.from_device_defaults(CPU, device)

    rng_bag.manual_seed(args.seed)

    log.info("Creating the root gang.")

    root_gang = ProgressGroupGang.create_root_process_group(device, timeout=999)

    log.info("Root gang created.")

    if args.tp_size > root_gang.size or root_gang.size % args.tp_size != 0:
        raise ArgumentError(
            "--tensor-parallel-size", f"must be a factor of the number of processes in the gang ({root_gang.size})"
        )

    gangs = create_parallel_gangs(root_gang, tp_size=args.tp_size)

    if gangs.dp.size > 1:
        log.warning("Using redundant data parallelism which may reduce throughput.")

    tokenizer_hub = get_tokenizer_hub()

    try:
        tokenizer = tokenizer_hub.load_tokenizer(args.model)
    except UnknownTokenizerError:
        xx

    sampler = TopPSampler(args.top_p)

    seq_generator = SamplingSequenceGenerator(
        model,
        tokenizer.vocab_info,
        sampler,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
    )

    program = Program(view, chatbot, seq_generator, tokenizer, gangs.root)

    program.run()




@dataclass(kw_only=True)
class ProgramConfig:
    model: ReferenceModelSection
    tokenizer: TokenizerSection
    gang: GangSection
    generator: GeneratorSection
    seq_generator: SequenceGeneratorSection
    common: CommonSection


def run_program(resolver: DependencyResolver) -> None:
    configure_torch(resolver)

    manual_seed(resolver)

    log_environment_info(resolver)

    gangs = resolver.resolve(Gangs)

    chatbot = resolver.resolve(Chatbot)

    tokenizer = resolver.resolve(Tokenizer)

    seq_generator = resolver.resolve(SequenceGenerator)

    view = resolver.resolve(ProgramView)

    seed_holder = resolver.resolve(SeedHolder)

    if gangs.dp.size > 1:
        log.warning("Using redundant data parallelism which may reduce throughput.")

    seed = seed_holder.advance()

    program = Program(view, chatbot, seq_generator, tokenizer, gangs.root, seed)

    program.run()


@final
class Program:
    _view: ProgramView
    _chatbot: Chatbot
    _generator: SequenceGenerator
    _tokenizer: Tokenizer
    _gang: Gang
    _dialog: list[ChatMessage]

    def __init__(
        self,
        view: ProgramView,
        chatbot: Chatbot,
        generator: SequenceGenerator,
        tokenizer: Tokenizer,
        gang: Gang,
    ) -> None:
        self._view = view
        self._chatbot = chatbot
        self._generator = generator
        self._tokenizer = tokenizer
        self._gang = gang
        self._dialog = []

    def run(self) -> None:
        if self._gang.rank == 0:
            self._run_interactive()
        else:
            self._run_non_interactive()

        self._gang.close()

    def _run_interactive(self) -> None:
        if self._chatbot.supports_system_prompt:
            prompt = self._view.input_system_prompt()
            if prompt:
                self._set_system_prompt(prompt)

        while True:
            message = self._view.input_message()
            if message == "bye":
                break

            self._send_message(message)

            self._receive_reply()

        self._finish_chat()

    def _set_system_prompt(self, prompt: str) -> None:
        message = ChatMessage(role="system", content=prompt)

        self._dialog.append(message)

        self._gang.broadcast_objects([message])

    def _send_message(self, content: str) -> None:
        message = ChatMessage(role="user", content=content)

        self._dialog.append(message)

        self._gang.broadcast_objects([message])

    def _receive_reply(self) -> None:
        self._view.print_reply("")

        text_decoder = self._tokenizer.create_decoder(skip_special_tokens=True)

        hook = ChatMessagePrintHook(self._view, text_decoder)

        with self._generator.register_step_hook(hook):
            response, _ = self._chatbot.response(self._dialog)

        self._view.print_reply_piece("\n")

        self._dialog.append(response)

    def _finish_chat(self) -> None:
        message = ChatMessage(role="user", content="bye")

        self._gang.broadcast_objects([message])

        self._view.print_reply("Bye!\n")

    def _run_non_interactive(self) -> None:
        while True:
            buffer: list[object] = [None]

            self._gang.broadcast_objects(buffer)

            if not isinstance(buffer[0], ChatMessage):
                raise InternalError(
                    f"The received object is of type `{type(buffer[0])}`."
                )

            message = buffer[0]

            if message.content == "bye":
                break

            self._dialog.append(message)

            if message.role == "system":
                continue

            response, _ = self._chatbot.response(self._dialog)

            self._dialog.append(response)


class ProgramView(ABC):
    @abstractmethod
    def input_system_prompt(self) -> str: ...

    @abstractmethod
    def input_message(self) -> str: ...

    @abstractmethod
    def print_reply(self, message: str) -> None: ...

    @abstractmethod
    def print_reply_piece(self, piece: str) -> None: ...


class ChatMessagePrintHook:
    _view: ProgramView
    _text_decoder: TokenDecoder
    _first_print: bool
    _prev_text_len: int

    def __init__(self, view: ProgramView, text_decoder: TokenDecoder) -> None:
        self._view = view
        self._text_decoder = text_decoder
        self._first_print = True
        self._prev_text_len = 0

    def __call__(
        self,
        prompt_indices: Tensor,
        seqs: Tensor,
        step_scores: Tensor | None,
        prefill: bool,
    ) -> None:
        if len(prompt_indices) != 1:
            raise InternalError(
                f"The length of `prompt_indices` is {len(prompt_indices)}."
            )

        # Do not print anything during prompt prefill.
        if prefill:
            return

        text = self._text_decoder(seqs[0])

        text_len = len(text)

        # If this is our first print, determine the length of the prompt text.
        if self._prev_text_len == 0:
            prev_text = self._text_decoder(seqs[0][:-1])

            prev_text_len = len(prev_text)
        else:
            prev_text_len = self._prev_text_len

        # Cache the length of the text so that we don't have to decode it twice
        # in the next step.
        self._prev_text_len = text_len

        # No need to print if we decoded a control symbol (e.g. EOS).
        if text_len == prev_text_len:
            return

        text = text[prev_text_len - text_len :]

        # Some models output multiple whitespace characters after the prompt.
        if self._first_print:
            text = text.lstrip()
            if not text:
                return

            self._first_print = False

        self._view.print_reply_piece(text)
