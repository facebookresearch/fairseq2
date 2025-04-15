# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from itertools import count
from pathlib import Path
from typing import final
from safetensors.torch import save_file

from typing_extensions import override

from fairseq2.assets import (
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
)
from fairseq2.cli import CliArgumentError, CliCommandHandler, CliCommandError
from fairseq2.cli.utils.rich import get_error_console
from fairseq2.context import RuntimeContext
from fairseq2.error import InternalError
from fairseq2.logging import log
from fairseq2.models import ModelConfigLoadError, ModelHandler
from fairseq2.models.qwen import (
    QWEN_MODEL_FAMILY,
    QwenConfig,
    convert_qwen_fs2_to_hf_checkpoint,
)

try:
    from transformers.models.qwen2 import Qwen2ForCausalLM
except ImportError:
    raise RuntimeError(
        "transformers library is required to convert Qwen model to HF checkpoint. Install it via `pip install transformers`."
    )

from fairseq2.utils.file import (
    TensorLoadError,
    TorchTensorLoader,
)


@final
class ConvertQwenCheckpointHandler(CliCommandHandler):
    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "model",
            type=str,
            help="model for which to generate params.json",
        )

        parser.add_argument(
            "input_dir",
            type=Path,
            help="checkpoint directory",
        )

        parser.add_argument(
            "output_dir",
            type=Path,
            help="output directory to store reference checkpoint",
        )

    @override
    def run(
        self, context: RuntimeContext, parser: ArgumentParser, args: Namespace
    ) -> int:
        file_system = context.file_system

        input_dir: Path = args.input_dir

        def read_error() -> CliCommandError:
            return CliCommandError(
                f"The '{input_dir}' directory cannot be read. See the nested exception for details."
            )

        try:
            input_exists = file_system.is_dir(input_dir)
        except OSError as ex:
            raise read_error() from ex

        if not input_exists:
            raise CliArgumentError("input_dir", "must be a directory")

        # Determine input checkpoint files.
        input_file = input_dir.joinpath("model.pt")

        try:
            input_file_exists = file_system.exists(input_file)
        except OSError as ex:
            raise read_error() from ex

        input_files = []

        if input_file_exists:
            input_files.append(input_file)
        else:
            raise NotImplementedError("TP>1 Qwen models not supported yet")
            for shard_idx in count():
                input_file = args.input_dir.joinpath(f"model.{shard_idx}.pt")

                try:
                    input_file_exists = file_system.exists(input_file)
                except OSError as ex:
                    raise read_error() from ex

                if not input_file_exists:
                    break

                input_files.append(input_file)

        if not input_files:
            raise CliArgumentError(
                "input_dir", "must contain a model checkpoint file (i.e. model.pt)"
            )

        output_dir: Path = args.output_dir

        def write_error() -> ProgramError:
            return ProgramError(
                f"The '{output_dir}' directory cannot be created. See the nested exception for details."
            )

        try:
            output_exists = file_system.exists(output_dir)
        except OSError as ex:
            raise write_error() from ex

        if output_exists:
            log.error("argument output_dir: already exists")

            return 2

        try:
            file_system.make_directory(output_dir)
        except OSError as ex:
            raise write_error() from ex

        # Load the model configuration.
        try:
            card = context.asset_store.retrieve_card(args.model)
        except AssetCardNotFoundError:
            raise CliArgumentError(
                "model", f"'{args.model}' is not a known LLaMA model. Use `fairseq2 assets list` to see the available models."  # fmt: skip
            ) from None
        except AssetCardError as ex:
            raise ProgramError(
                f"The '{args.model}' asset card cannot be read. See the nested exception for details."
            ) from ex

        try:
            family = card.field("model_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise CliArgumentError(
                "model", f"'{args.model}' is not a known LLaMA model. Use `fairseq2 assets list` to see the available models."  # fmt: skip
            ) from None
        except AssetCardError as ex:
            raise ProgramError(
                f"The '{args.model}' asset card cannot be read. See the nested exception for details."
            ) from ex

        if family != QWEN_MODEL_FAMILY:
            raise CliArgumentError(
                "model", f"'{args.model}' is not a model of QWEN2.5 family."
            )

        model_handlers = context.get_registry(ModelHandler)

        try:
            model_handler = model_handlers.get(QWEN_MODEL_FAMILY)
        except LookupError:
            raise InternalError(
                "The LLaMA model handler cannot be found. Please file a bug report."
            ) from None

        try:
            model_config = model_handler.load_config(card)
        except ModelConfigLoadError as ex:
            raise ProgramError(
                f"The configuration of '{args.model}' cannot be loaded. See the nested exception for details."
            ) from ex

        if not isinstance(model_config, QwenConfig):
            raise InternalError(
                "The model configuration type is not valid. Please file a bug report."
            )

        # Begin conversion.
        console = get_error_console()

        tensor_loader = TorchTensorLoader(file_system)

        with console.status("[bold green]Converting...") as status:
            for input_file in input_files:

                def file_write_error() -> ProgramError:
                    return ProgramError(
                        f"The '{input_file}' checkpoint file cannot be converted. See the nested exception for details."
                    )

                status.update(f"[bold green]Loading {input_file.name}...")

                try:
                    checkpoint = tensor_loader.load(input_file)
                except TensorLoadError as ex:
                    raise file_write_error() from ex

                status.update(f"[bold green]Converting {input_file.name} ...")  # fmt: skip

                hf_config = model_config.to_hf_config()

                try:
                    ref_state_dict = convert_qwen_fs2_to_hf_checkpoint(
                        checkpoint[checkpoint["model_key"]], model_config
                    )
                except (TypeError, KeyError) as ex:
                    raise file_write_error() from ex

                model = Qwen2ForCausalLM(hf_config)

                model.load_state_dict(ref_state_dict)

                model.save_pretrained(args.output_dir)
                hf_config.save_pretrained(args.output_dir)

                log.info("{} converted!", input_file.name)

        if model_config is None:
            return 0

        return 0
