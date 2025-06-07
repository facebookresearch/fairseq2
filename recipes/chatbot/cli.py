# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import timedelta
from typing import final

import torch
from rich.console import Console
from typing_extensions import override

from fairseq2.assets import AssetNotFoundError, get_asset_store
from fairseq2.cluster import (
    ClusterNotDetectedError,
    ClusterNotSupportedError,
    set_torch_distributed_env_variables,
)
from fairseq2.data.tokenizers import load_tokenizer
from fairseq2.device import CPU, get_default_device
from fairseq2.gang import (
    FakeGang,
    Gang,
    GangError,
    ProcessGroupGang,
    create_parallel_gangs,
    raise_operational_gang_error,
)
from fairseq2.generation.sampling import SamplingSequenceGenerator, TopPSampler
from fairseq2.logging import log
from fairseq2.models import load_model
from fairseq2.models.clm import CausalLM
from fairseq2.utils.argparse import parse_dtype
from fairseq2.utils.log import (
    log_environment_variables,
    log_software_info,
    log_system_info,
)
from fairseq2.utils.rich import configure_rich_logging, get_console
from fairseq2.utils.rng import RngBag
from fairseq2.world_info import get_world_info

from .chatbot import StandardChatbot
from .llama import create_llama_dialog_encoder
from .program import Program, ProgramView


def _main() -> None:
    args = _parse_args()

    configure_rich_logging()

    try:
        _run(args)
    #    except UnknownChatbotError as ex:
    #        raise ArgumentError(
    #            "--model",
    #            f"'{ex.model_name}' model does not have a chatbot implementation.",
    #        ) from None
    #    except UnknownClusterError as ex:
    #        s = ", ".join(ex.supported_clusters)
    #
    #        raise ArgumentError(
    #            "cluster", f"'{ex.cluster}' is not a known cluster. Must be one of: auto, none, {s}"  # fmt: skip
    #        ) from None
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


def _run(args: Namespace) -> None:
    try:
        set_torch_distributed_env_variables(args.cluster)
    except ClusterNotSupportedError as ex:
        s = ", ".join(ex.supported_clusters)

        raise ArgumentError("--cluster", f"not a supported cluster. must be one of {s}")
    except ClusterNotDetectedError as ex:
        raise ArgumentError("--cluster", f"dede {ex.cluster}")

    device = get_default_device()

    if device.type == "cuda":
        torch.cuda.set_device(device)

    log.info("Device of the process set to {}.", device)

    log_system_info(device)

    log_software_info()

    log_environment_variables()

    rng_bag = RngBag.from_device_defaults(CPU, device)

    rng_bag.manual_seed(args.seed)

    log.info("Random number generator seed set to {}.", args.seed)

    log.info("Creating gangs.")

    root_gang: Gang

    world_info = get_world_info()
    if world_info.size > 1:
        timeout = timedelta(minutes=999)

        try:
            root_gang = ProcessGroupGang.create_default_process_group(
                device, timeout=timeout
            )
        except GangError as ex:
            raise_operational_gang_error(ex)
    else:
        root_gang = FakeGang(device)

    if root_gang.size % args.tp_size != 0:
        raise ArgumentError(
            "--tensor-parallel-size", f"must be a factor of the number of processes in the gang ({root_gang.size})"  # fmt: skip
        )

    try:
        gangs = create_parallel_gangs(root_gang, tp_size=args.tp_size)
    except GangError as ex:
        raise_operational_gang_error(ex)

    log.info("Gangs created.")

    if gangs.dp.size > 1:
        log.warning("Using redundant data parallelism which may reduce throughput.")

    try:
        card = get_asset_store().retrieve_card(args.model)
    except AssetNotFoundError:
        raise ArgumentError("--model", "not a known model") from None

    log.info("Loading {} model.", card.name)

    model = load_model(card, gangs=gangs, dtype=args.dtype, progress=True)

    log.info("Model loaded.")

    if not isinstance(model, CausalLM):
        raise ArgumentError("--model", "must be a causal language model")

    log.info("Loading {} tokenizer.", card.name)

    tokenizer = load_tokenizer(card, progress=True)

    log.info("Tokenizer loaded.")

    sampler = TopPSampler(args.top_p)

    seq_generator = SamplingSequenceGenerator(
        model,
        tokenizer.vocab_info,
        sampler,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
    )

    model_family = card.field("model_family").as_(str)
    if model_family != "llama":
        raise ValueError("dede")

    dialog_encoder = create_llama_dialog_encoder(tokenizer, device)

    text_decoder = tokenizer.create_decoder(skip_special_tokens=True)

    chatbot = StandardChatbot(seq_generator, dialog_encoder, text_decoder)

    console = get_console()

    view = CliProgramView(args.model, console)

    program = Program(view, chatbot, seq_generator, tokenizer, gangs.root)

    program.run()


class ArgumentError(Exception):
    param_name: str

    def __init__(self, param_name: str, message: str) -> None:
        super().__init__(f"argument: {param_name}: {message}")

        self.param_name = param_name


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
