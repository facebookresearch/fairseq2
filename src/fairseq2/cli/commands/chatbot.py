# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import cast, final

import torch
from rich.console import Console
from torch import Tensor
from typing_extensions import override

from fairseq2.chatbots import Chatbot, ChatbotHandler, ChatMessage, UnknownChatbotError
from fairseq2.cli import CliCommandError, CliCommandHandler
from fairseq2.cli.utils.argparse import parse_dtype
from fairseq2.cli.utils.cluster import set_torch_distributed_variables
from fairseq2.cli.utils.rich import get_console
from fairseq2.context import RuntimeContext
from fairseq2.data.text.tokenizers import TextTokenDecoder, TextTokenizer
from fairseq2.error import InternalError
from fairseq2.gang import Gang, GangError
from fairseq2.generation import (
    SamplingSequenceGenerator,
    SequenceGenerator,
    TopPSampler,
)
from fairseq2.logging import log
from fairseq2.models.decoder import DecoderModel
from fairseq2.recipes import RecipeError
from fairseq2.recipes.common import (
    load_text_tokenizer,
    setup_gangs,
    setup_reference_model,
)
from fairseq2.recipes.config import GangSection, ReferenceModelSection
from fairseq2.typing import CPU
from fairseq2.utils.rng import RngBag


@final
class RunChatbotHandler(CliCommandHandler):
    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "-m",
            "--model",
            dest="model_name",
            metavar="MODEL_NAME",
            default="llama3_1_8b_instruct",
            help="instruct model name (default: %(default)s)",
        )

        parser.add_argument(
            "--dtype",
            type=parse_dtype,
            default=torch.bfloat16,
            help="data type of the model (default: %(default)s)",
        )

        parser.add_argument(
            "--tensor-parallel-size",
            type=int,
            default=1,
            help="tensor parallelism size (default: %(default)s)",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=2,
            help="random number generator seed for sequence generation (default: %(default)s)",
        )

        parser.add_argument(
            "--top-p",
            type=float,
            default=0.8,
            help="probability threshold for top-p sampling (default: %(default)s)",
        )

        parser.add_argument(
            "--temperature",
            type=float,
            default=0.6,
            help="sampling temperature (default: %(default)s)",
        )

        parser.add_argument(
            "--max-gen-len",
            type=int,
            default=2048,
            help="maximum sequence generation length (default: %(default)s)",
        )

        parser.add_argument(
            "--cluster",
            default="auto",
            help="cluster on which the chatbot runs (default: %(default)s)",
        )

    @override
    @torch.inference_mode()
    def run(
        self, context: RuntimeContext, parser: ArgumentParser, args: Namespace
    ) -> int:
        console = get_console()

        view = CliChatbotView(args.model_name, console)

        args.gang = GangSection(
            tensor_parallel_size=args.tensor_parallel_size, timeout=999
        )

        args.model = ReferenceModelSection(name=args.model_name)

        set_torch_distributed_variables(context, args.cluster)

        torch.set_float32_matmul_precision("high")

        try:
            gangs = setup_gangs(context, args)
        except RecipeError as ex:
            raise CliCommandError(
                "The chatbot setup has failed. See the nested exception for details."
            ) from ex

        if gangs.dp.size > 1:
            log.warning("Using redundant data parallelism which may reduce throughput. It is recommended to use one device per model shard (i.e. a single device for a non-sharded model).")  # fmt: skip

        try:
            model = setup_reference_model(
                DecoderModel,
                context,
                args.model_name,
                gangs,
                args.dtype,
                mp=False,
                torch_compile=False,
            )
        except RecipeError as ex:
            raise CliCommandError(
                "The chatbot setup has failed. See the nested exception for details."
            ) from ex

        module = cast(DecoderModel, model.module)

        sampler = TopPSampler(p=args.top_p)

        generator = SamplingSequenceGenerator(
            module, sampler, temperature=args.temperature, max_gen_len=args.max_gen_len
        )

        try:
            tokenizer = load_text_tokenizer(context, args)
        except RecipeError as ex:
            raise CliCommandError(
                "The chatbot setup has failed. See the nested exception for details."
            ) from ex

        card = context.asset_store.retrieve_card(args.model_name)

        family = card.field("model_family").as_(str)

        chatbot_handlers = context.get_registry(ChatbotHandler)

        try:
            chatbot_handler = chatbot_handlers.get(family)
        except LookupError:
            raise UnknownChatbotError(args.model_name) from None

        chatbot = chatbot_handler.create(generator, tokenizer)

        program = ChatbotProgram(
            view, chatbot, generator, tokenizer, gangs.root, args.seed
        )

        try:
            program.run()
        except GangError as ex:
            raise CliCommandError(
                "A collective communication operation has failed. See the nested exception for details."
            ) from ex
        except KeyboardInterrupt:
            console.print()

            raise

        return 0


@final
class ChatbotProgram:
    _view: ChatbotView
    _chatbot: Chatbot
    _generator: SequenceGenerator
    _tokenizer: TextTokenizer
    _root_gang: Gang
    _seed: int
    _dialog: list[ChatMessage]

    def __init__(
        self,
        view: ChatbotView,
        chatbot: Chatbot,
        generator: SequenceGenerator,
        tokenizer: TextTokenizer,
        root_gang: Gang,
        seed: int,
    ) -> None:
        self._view = view
        self._chatbot = chatbot
        self._generator = generator
        self._tokenizer = tokenizer
        self._root_gang = root_gang
        self._seed = seed

        self._dialog = []

    def run(self) -> None:
        rng_bag = RngBag.from_device_defaults(CPU, self._root_gang.device)

        rng_bag.manual_seed(self._seed)

        try:
            if self._root_gang.rank == 0:
                self._run_interactive()
            else:
                self._run_non_interactive()
        finally:
            self._root_gang.close()

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

        self._root_gang.broadcast_objects([message])

    def _send_message(self, content: str) -> None:
        message = ChatMessage(role="user", content=content)

        self._dialog.append(message)

        self._root_gang.broadcast_objects([message])

    def _receive_reply(self) -> None:
        self._view.print_reply("")

        text_decoder = self._tokenizer.create_decoder()

        hook = _PrintHook(self._view, text_decoder)

        with self._generator.register_step_hook(hook):
            response, _ = self._chatbot.response(self._dialog)

        self._view.print_reply_piece("\n")

        self._dialog.append(response)

    def _finish_chat(self) -> None:
        message = ChatMessage(role="user", content="bye")

        self._root_gang.broadcast_objects([message])

        self._view.print_reply("Bye!\n")

    def _run_non_interactive(self) -> None:
        while True:
            buffer: list[object] = [None]

            self._root_gang.broadcast_objects(buffer)

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


class ChatbotView(ABC):
    @abstractmethod
    def input_system_prompt(self) -> str: ...

    @abstractmethod
    def input_message(self) -> str: ...

    @abstractmethod
    def print_reply(self, message: str) -> None: ...

    @abstractmethod
    def print_reply_piece(self, piece: str) -> None: ...


@final
class CliChatbotView(ChatbotView):
    _name: str
    _console: Console
    _has_first_message: bool

    def __init__(self, name: str, console: Console) -> None:
        self._name = name
        self._console = console
        self._has_first_message = False

    @override
    def input_system_prompt(self) -> str:
        prompt = self._console.input("\nSystem Prompt (press enter to skip): ")

        self._console.print()

        return prompt

    @override
    def input_message(self) -> str:
        if not self._has_first_message:
            self._console.print("\nYou can end the chat by typing 'bye'.")

            self._has_first_message = True

        return self._console.input("\n[green bold]You> ")

    @override
    def print_reply(self, message: str) -> None:
        self._console.print(f"\n[blue bold]{self._name}> ", end="")

        self.print_reply_piece(message)

    @override
    def print_reply_piece(self, piece: str) -> None:
        self._console.print(piece, highlight=False, end="")


class _PrintHook:
    _view: ChatbotView
    _text_decoder: TextTokenDecoder
    _first_print: bool
    _prev_text_len: int

    def __init__(self, view: ChatbotView, text_decoder: TextTokenDecoder) -> None:
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

        # Some models output several whitespace characters after the prompt.
        if self._first_print:
            text = text.lstrip()
            if not text:
                return

            self._first_print = False

        self._view.print_reply_piece(text)
