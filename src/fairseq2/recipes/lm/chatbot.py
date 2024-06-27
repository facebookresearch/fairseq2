# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from argparse import ArgumentParser, Namespace
from datetime import timedelta
from typing import List, Optional, final

import torch

from fairseq2.console import get_console
from fairseq2.data.text import load_text_tokenizer
from fairseq2.gang import Gang
from fairseq2.generation import (
    Chatbot,
    ChatMessage,
    SamplingSequenceGenerator,
    TopPSampler,
)
from fairseq2.logging import get_log_writer
from fairseq2.models import create_chatbot, load_model
from fairseq2.models.decoder import DecoderModel
from fairseq2.recipes.cli import CliCommandHandler
from fairseq2.recipes.logging import setup_basic_logging
from fairseq2.recipes.utils.argparse import parse_dtype
from fairseq2.recipes.utils.environment import default_env_setters
from fairseq2.recipes.utils.setup import setup_gangs
from fairseq2.typing import CPU, override
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
            default="llama3_8b_instruct",
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
            default=512,
            help="maximum sequence generation length (default: %(default)s)",
        )

        clusters = list(default_env_setters.names())

        clusters.sort()

        parser.add_argument(
            "--cluster",
            choices=["auto"] + clusters,
            default="auto",
            help="cluster on which the chatbot runs (default: %(default)s)",
        )

    @override
    def __call__(self, args: Namespace) -> None:
        setup_basic_logging()

        # Set up cluster-specific environment variables.
        if args.cluster == "auto":
            env_setter = default_env_setters.get_for_inferred_cluster()
        else:
            try:
                env_setter = default_env_setters.get(args.cluster)
            except RuntimeError:
                log.exception("Chatbot is not running on a '{}' cluster.", args.cluster)  # fmt: skip

                sys.exit(1)

        try:
            env_setter.set_torch_distributed_env()
        except RuntimeError:
            log.exception("'{}' cluster environment cannot be set.", env_setter.cluster)  # fmt: skip

            sys.exit(1)

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
            log.error("The model must be a decoder model.")

            sys.exit(1)

        log.info("Model loaded.")

        # Initialize the chatbot.
        sampler = TopPSampler(p=args.top_p)

        generator = SamplingSequenceGenerator(
            model, sampler, temperature=args.temperature, max_gen_len=args.max_gen_len
        )

        chatbot = create_chatbot(generator, tokenizer)

        rng_bag = RngBag.from_device_defaults(CPU, root_gang.device)

        # Set the seed for sequence generation.
        rng_bag.manual_seed(args.seed)

        self._do_run(args.model_name, chatbot, root_gang)

    def _do_run(self, chatbot_name: str, chatbot: Chatbot, gang: Gang) -> None:
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

                    response, _ = chatbot(dialog, stdout=True)

                    console.print("\n")

                    dialog.append(response)

                gang.broadcast_objects([ChatMessage(role="user", content="bye")])

                console.print(f"\n[blue bold]{chatbot_name}>[/blue bold] Bye!")
            except KeyboardInterrupt:
                console.print()

                raise
        else:
            while True:
                message_buffer: List[Optional[ChatMessage]] = [None]

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
