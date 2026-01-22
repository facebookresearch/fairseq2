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
from torch.cuda import OutOfMemoryError
from typing_extensions import override

from fairseq2.assets import (
    AssetCardError,
    AssetDownloadError,
    AssetMetadataError,
    AssetNotFoundError,
    get_asset_store,
)
from fairseq2.cluster import (
    ClusterNotDetectedError,
    ClusterNotKnownError,
    set_torch_distributed_env_variables,
)
from fairseq2.composition import ExtensionError
from fairseq2.data.tokenizers import load_tokenizer
from fairseq2.device import CPU, get_default_device
from fairseq2.error import OperationalError
from fairseq2.gang import (
    FakeGang,
    Gang,
    GangError,
    ProcessGroupGang,
    create_parallel_gangs,
    raise_operational_gang_error,
)
from fairseq2.generation.sampling import SamplingSequenceGenerator, TopPSampler
from fairseq2.logging import configure_logging, log
from fairseq2.model_checkpoint import CorruptModelCheckpointError
from fairseq2.models import ModelGatedError, ModelNotKnownError, load_model
from fairseq2.models.clm import CausalLM
from fairseq2.recipe.error import (
    GangTopologyError,
    ModelTypeNotValidError,
    raise_model_type_not_valid_error,
)
from fairseq2.utils.argparse import parse_dtype
from fairseq2.utils.rich import get_console
from fairseq2.utils.rng import RngBag
from fairseq2.world_info import get_world_info

from .chatbot import StandardChatbot
from .llama import create_llama_dialog_encoder
from .mistral import MistralDialogEncoder
from .program import Program, ProgramView


@torch.inference_mode()
def _main() -> None:
    args = _parse_args()

    configure_logging()

    try:
        _run(args)
    except KeyboardInterrupt:
        get_console().print()

        raise
    except ClusterNotDetectedError as ex:
        log.error("{} cluster not detected.", ex.cluster)  # fmt: skip

        sys.exit(2)
    except ClusterNotKnownError as ex:
        log.error("{} is not a known cluster.", ex.cluster)  # fmt: skip

        sys.exit(2)
    except GangTopologyError as ex:
        log.error("--tensor-parallel-size must be a factor of the number of processes in the root gang ({}), but is {} instead.", ex.world_size, ex.tp_size)  # fmt: skip

        sys.exit(2)
    except ModelNotKnownError as ex:
        log.error("{} is not a known model.", ex.name)  # fmt: skip

        sys.exit(2)
    except ModelTypeNotValidError:
        log.error("Model must be a text-only causal language model.")  # fmt: skip

        sys.exit(2)
    except ModelGatedError as ex:
        log.error("{} is a gated model.", ex.name)  # fmt: skip

        sys.exit(2)
    except ChatbotNotSupportedError as ex:
        log.error("{} model does not have a chatbot.", ex.model_name)  # fmt: skip

        sys.exit(2)
    except AssetMetadataError as ex:
        log.exception("Asset metadata in {} is erroneous. See logged stack trace for details.", ex.source)  # fmt: skip

        sys.exit(1)
    except AssetCardError as ex:
        log.exception("{} asset card is erroneous. See logged stack trace for details.", ex.name)  # fmt: skip

        sys.exit(1)
    except AssetDownloadError as ex:
        log.exception("Failed to download {}.", ex.uri)

        sys.exit(1)
    except CorruptModelCheckpointError as ex:
        log.exception("Model checkpoint at {} is erroneous. See logged stack trace for details.", ex.path)  # fmt: skip

        sys.exit(1)
    except OperationalError:
        log.exception("Command failed due to an operational error. See logged stack trace for details.")  # fmt: skip

        sys.exit(1)
    except ExtensionError as ex:
        log.exception("{} extension failed to initialize. See logged stack trace for details.", ex.entry_point)  # fmt: skip

        sys.exit(1)
    except OutOfMemoryError:
        log.exception("CUDA out of memory.")  # fmt: skip

        sys.exit(1)
    except Exception:
        log.exception("Command failed due to an unexpected error. See logged stack trace for details and file a bug report to the corresponding author.")  # fmt: skip

        sys.exit(1)


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
    set_torch_distributed_env_variables(args.cluster)

    device = get_default_device()

    if device.type == "cuda":
        torch.cuda.set_device(device)

    log.info("Device of the process set to {}.", device)

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
        raise GangTopologyError(root_gang.size, args.tp_size)

    try:
        gangs = create_parallel_gangs(root_gang, tp_size=args.tp_size)
    except GangError as ex:
        raise_operational_gang_error(ex)

    log.info("Gangs created.")

    if gangs.dp.size > 1:
        log.warning("Using redundant data parallelism which may reduce throughput.")

    asset_store = get_asset_store()

    try:
        card = asset_store.retrieve_card(args.model)
    except AssetNotFoundError:
        raise ModelNotKnownError(args.model) from None

    log.info("Loading {} model.", card.name)

    model = load_model(card, gangs=gangs, dtype=args.dtype, progress=True)

    log.info("Model loaded.")

    model.requires_grad_(False)

    model.eval()

    log.info("{}", model)

    if not isinstance(model, CausalLM):
        raise raise_model_type_not_valid_error(model, CausalLM)

    log.info("Loading {} tokenizer.", card.name)

    tokenizer = load_tokenizer(card, gangs=gangs, progress=True)

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
    match model_family:
        case "llama":
            dialog_encoder = create_llama_dialog_encoder(tokenizer, device)
        case "mistral":
            dialog_encoder = MistralDialogEncoder(tokenizer, device)
        case _:
            raise ChatbotNotSupportedError(args.model)

    text_decoder = tokenizer.create_decoder(skip_special_tokens=True)

    chatbot = StandardChatbot(seq_generator, dialog_encoder, text_decoder)

    console = get_console()

    view = CliProgramView(args.model, console)

    program = Program(view, chatbot, seq_generator, tokenizer, gangs.root)

    program.run()


class ChatbotNotSupportedError(Exception):
    def __init__(self, model_name: str) -> None:
        super().__init__(f"{model_name} model does not have a chatbot.")

        self.model_name = model_name


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
