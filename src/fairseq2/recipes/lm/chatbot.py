# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from datetime import timedelta
from typing import final

import torch
from rich.console import Console
from torch import Tensor
from typing_extensions import override

from fairseq2.chatbots import Chatbot, ChatbotRegistry, ChatMessage, register_chatbots
from fairseq2.data.text import TextTokenDecoder, TextTokenizer, load_text_tokenizer
from fairseq2.error import InternalError
from fairseq2.gang import Gang, is_torchrun
from fairseq2.generation import (
    SamplingSequenceGenerator,
    SequenceGenerator,
    TopPSampler,
)
from fairseq2.logging import get_log_writer
from fairseq2.models import load_model
from fairseq2.models.decoder import DecoderModel
from fairseq2.recipes.cli import CliCommandHandler
from fairseq2.recipes.cluster import ClusterError, ClusterRegistry, register_clusters
from fairseq2.recipes.console import get_console
from fairseq2.recipes.utils.argparse import parse_dtype
from fairseq2.recipes.utils.setup import setup_gangs
from fairseq2.setup import setup_fairseq2
from fairseq2.typing import CPU
from fairseq2.utils.rng import RngBag

log = get_log_writer(__name__)


@final
class ChatbotCommandHandler(CliCommandHandler):
    """Runs a chatbot."""

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
            help="cluster on which the recipe runs (default: %(default)s)",
        )

    @override
    def run(self, args: Namespace) -> int:
        cluster_registry = ClusterRegistry(is_torchrun=is_torchrun())

        register_clusters(cluster_registry)

        # Set up cluster-specific environment variables.
        try:
            cluster_handler = cluster_registry.get(args.cluster)
        except LookupError:
            log.exception("Chatbot is not running on a '{}' cluster.", args.cluster)  # fmt: skip

            sys.exit(1)

        try:
            cluster_handler.set_torch_distributed_variables()
        except ClusterError:
            log.exception("'{}' cluster environment cannot be set.", args.cluster)  # fmt: skip

            sys.exit(1)

        setup_fairseq2()

        # Since this is an interactive program, do not timeout while waiting for
        # user's input.
        root_gang, gangs = setup_gangs(
            log, tp_size=args.tensor_parallel_size, timeout=timedelta(days=999)
        )

        if gangs["dp"].size > 1:
            log.warning("Using redundant data parallelism which may slow down response times. It is recommended to use one device per model shard (i.e. a single device for a non-sharded model).")  # fmt: skip

        # Load the tokenizer.
        log.info("Loading {} tokenizer.", args.model_name)

        tokenizer = load_text_tokenizer(args.model_name)

        log.info("Tokenizer loaded.")

        # Load the model.
        log.info("Loading {} model.", args.model_name)

        model = load_model(args.model_name, gangs=gangs, dtype=args.dtype)

        if not isinstance(model, DecoderModel):
            log.exception("The model must be of type `{}`, but is of type `{}` instead.", DecoderModel, type(model))  # fmt: skip

            sys.exit(1)

        log.info("Model loaded.")

        # Initialize the chatbot.
        sampler = TopPSampler(p=args.top_p)

        generator = SamplingSequenceGenerator(
            model, sampler, temperature=args.temperature, max_gen_len=args.max_gen_len  # type: ignore[arg-type]
        )

        if model.family is None:
            log.error("The model has no family name defined.")

            sys.exit(1)

        registry = ChatbotRegistry()

        register_chatbots(registry)

        try:
            handler = registry.get(model.family)
        except LookupError:
            log.exception("The chatbot cannot be created.")

            sys.exit(1)

        chatbot = handler.make(generator, tokenizer)

        rng_bag = RngBag.from_device_defaults(CPU, root_gang.device)

        # Set the seed for sequence generation.
        rng_bag.manual_seed(args.seed)

        self._do_run(
            args.model_name,
            chatbot,
            generator,
            tokenizer,
            root_gang,
        )

        return 0

    def _do_run(
        self,
        chatbot_name: str,
        chatbot: Chatbot,
        generator: SequenceGenerator,
        tokenizer: TextTokenizer,
        gang: Gang,
    ) -> None:
        dialog = []

        if gang.rank == 0:
            console = get_console()

            try:
                console.print()

                if chatbot.supports_system_prompt:
                    system_prompt = console.input(
                        "System Prompt (press enter to skip): "
                    )

                    if system_prompt:
                        message = ChatMessage(role="system", content=system_prompt)

                        dialog.append(message)

                        gang.broadcast_objects([message])

                    console.print()

                console.print("You can end the chat by typing 'bye'.\n")

                while (prompt := console.input("[green bold]You> ")) != "bye":
                    message = ChatMessage(role="user", content=prompt)

                    dialog.append(message)

                    gang.broadcast_objects([message])

                    console.print(f"\n[blue bold]{chatbot_name}> ", end="")

                    hook = PrintHook(console, tokenizer)

                    with generator.register_step_hook(hook):
                        response, _ = chatbot(dialog)

                    console.print("\n")

                    dialog.append(response)

                gang.broadcast_objects([ChatMessage(role="user", content="bye")])

                console.print(f"\n[blue bold]{chatbot_name}>[/blue bold] Bye!")
            except KeyboardInterrupt:
                console.print()

                raise
        else:
            while True:
                message_buffer: list[object] = [None]

                gang.broadcast_objects(message_buffer)

                assert isinstance(message_buffer[0], ChatMessage)

                message = message_buffer[0]

                if message.content == "bye":
                    break

                dialog.append(message)

                if message.role == "system":
                    continue

                response, _ = chatbot(dialog)

                dialog.append(response)


@final
class PrintHook:
    _console: Console
    _text_decoder: TextTokenDecoder
    _first_print: bool
    _prev_text_len: int

    def __init__(self, console: Console, tokenizer: TextTokenizer) -> None:
        self._console = console
        self._text_decoder = tokenizer.create_decoder()
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

        self._console.print(text, highlight=False, end="")
