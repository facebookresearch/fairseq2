# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import final

import torch
from rich.console import Console
from typing_extensions import override

from fairseq2.cluster import UnknownClusterError
from fairseq2.composition import register_library
from fairseq2.dependency import DependencyResolver, StandardDependencyContainer
from fairseq2.recipe.composition import register_cli_errors
from fairseq2.recipe.config import (
    SAMPLING_GENERATOR,
    TOP_P_SAMPLER,
    CommonSection,
    GangSection,
    GeneratorSection,
    ReferenceModelSection,
    SamplerSection,
    SamplingConfig,
    SequenceGeneratorSection,
    TokenizerSection,
    TopPSamplerConfig,
)
from fairseq2.recipe.model import Model
from fairseq2.recipe.utils.argparse import parse_dtype
from fairseq2.utils.cli import ArgumentError, handle_errors
from fairseq2.utils.rich import configure_rich_logging, get_console

from .chatbots import UnknownChatbotError
from .composition import register_program
from .program import ProgramConfig, ProgramView, run_program


def _main() -> None:
    args = _parse_args()

    configure_rich_logging()

    container = StandardDependencyContainer()

    with handle_errors(container):
        # Library
        register_library(container)

        # Program
        register_program(container)

        # Program Configuration
        def load_config(resolver: DependencyResolver) -> object:
            return load_config_from_args(resolver, args)

        container.register(object, load_config, key="config")

        # Program View
        container.register(ProgramView, create_program_view)

        # CLI Errors
        register_cli_errors(container)

        try:
            run_program(container)
        except UnknownChatbotError as ex:
            raise ArgumentError(
                "--model",
                f"'{ex.model_name}' model does not have a chatbot implementation.",
            ) from None
        except UnknownClusterError as ex:
            s = ", ".join(ex.supported_clusters)

            raise ArgumentError(
                "cluster", f"'{ex.cluster}' is not a known cluster. Must be one of: auto, none, {s}"  # fmt: skip
            ) from None
        except KeyboardInterrupt:
            get_console().print()

            raise


def _parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        metavar="MODEL_NAME",
        default="llama3_1_8b_instruct",
        help="instruct model name (default: %(default)s)",
    )

    parser.add_argument(
        "--dtype",
        type=parse_dtype,
        default=torch.float16,
        help="data type of the model (default: %(default)s)",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        dest="tp_size",
        help="tensor parallelism size (default: %(default)s)",
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
        "--seed",
        type=int,
        default=2,
        help="random number generator seed for sequence generation (default: %(default)s)",
    )

    parser.add_argument(
        "--cluster",
        default="auto",
        help="cluster on which the chatbot runs (default: %(default)s)",
    )

    return parser.parse_args()


def load_config_from_args(
    resolver: DependencyResolver, args: Namespace
) -> ProgramConfig:
    return ProgramConfig(
        model=ReferenceModelSection(name=args.model),
        tokenizer=TokenizerSection(name=args.model),
        gang=GangSection(tensor_parallel_size=args.tensor_parallel_size),
        generator=GeneratorSection(dtype=args.dtype),
        seq_generator=SequenceGeneratorSection(
            name=SAMPLING_GENERATOR,
            config=SamplingConfig(
                sampler=SamplerSection(
                    name=TOP_P_SAMPLER,
                    config=TopPSamplerConfig(p=args.top_p),
                ),
                max_gen_len=args.max_gen_len,
                temperature=args.temperature,
            ),
        ),
        common=CommonSection(seed=args.seed, cluster=args.cluster),
    )


def create_program_view(resolver: DependencyResolver) -> ProgramView:
    model = resolver.resolve(Model)

    console = get_console()

    return CliProgramView(model.name, console)


@final
class CliProgramView(ProgramView):
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
