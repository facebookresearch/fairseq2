# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from itertools import count
from pathlib import Path
from typing import final

from typing_extensions import override

from fairseq2.assets import (
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
)
from fairseq2.cli import CliArgumentError, CliCommandError, CliCommandHandler
from fairseq2.cli.utils.rich import get_error_console
from fairseq2.context import RuntimeContext
from fairseq2.error import InternalError
from fairseq2.logging import log
from fairseq2.models import ModelConfigLoadError, ModelHandler
from fairseq2.models.llama import LLAMA_MODEL_FAMILY, LLaMAConfig
from fairseq2.models.llama.integ import convert_to_reference_llama_checkpoint
from fairseq2.utils.file import (
    FileMode,
    StandardTensorLoader,
    TensorDumpError,
    TensorLoadError,
    TorchTensorDumper,
)


@final
class ConvertLLaMACheckpointHandler(CliCommandHandler):
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

        def write_error() -> CliCommandError:
            return CliCommandError(
                f"The '{output_dir}' directory cannot be created. See the nested exception for details."
            )

        try:
            output_exists = file_system.exists(output_dir)
        except OSError as ex:
            raise write_error() from ex

        if output_exists:
            log.error("argument output_dir: already exists")

            return 2

        output_files = []

        # Determine output checkpoint filenames.
        for shard_idx in range(len(input_files)):
            output_file = output_dir.joinpath(f"consolidated.{shard_idx:02d}.pth")

            output_files.append(output_file)

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
            raise CliCommandError(
                f"The '{args.model}' asset card cannot be read. See the nested exception for details."
            ) from ex

        try:
            family = card.field("model_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise CliArgumentError(
                "model", f"'{args.model}' is not a known LLaMA model. Use `fairseq2 assets list` to see the available models."  # fmt: skip
            ) from None
        except AssetCardError as ex:
            raise CliCommandError(
                f"The '{args.model}' asset card cannot be read. See the nested exception for details."
            ) from ex

        if family != LLAMA_MODEL_FAMILY:
            raise CliArgumentError(
                "model", f"'{args.model}' is not a model of LLaMA family."
            )

        model_handlers = context.get_registry(ModelHandler)

        try:
            model_handler = model_handlers.get(LLAMA_MODEL_FAMILY)
        except LookupError:
            raise InternalError(
                "The LLaMA model handler cannot be found. Please file a bug report."
            ) from None

        try:
            model_config = model_handler.load_config(card)
        except ModelConfigLoadError as ex:
            raise CliCommandError(
                f"The configuration of the '{args.model}' model cannot be loaded. See the nested exception for details."
            ) from ex

        if not isinstance(model_config, LLaMAConfig):
            raise InternalError(
                "The model configuration type is not valid. Please file a bug report."
            )

        # Begin conversion.
        console = get_error_console()

        tensor_loader = StandardTensorLoader(file_system)
        tensor_dumper = TorchTensorDumper(file_system)

        with console.status("[bold green]Converting...") as status:
            for input_file, output_file in zip(input_files, output_files):

                def file_write_error() -> CliCommandError:
                    return CliCommandError(
                        f"The '{input_file}' checkpoint file cannot be converted. See the nested exception for details."
                    )

                status.update(f"[bold green]Loading {input_file.name}...")

                try:
                    checkpoint = tensor_loader.load(input_file)
                except TensorLoadError as ex:
                    raise file_write_error() from ex

                status.update(f"[bold green]Converting {input_file.name} to {output_file.name}...")  # fmt: skip

                try:
                    ref_state_dict = convert_to_reference_llama_checkpoint(checkpoint)
                except (TypeError, KeyError) as ex:
                    raise file_write_error() from ex

                try:
                    tensor_dumper.dump(ref_state_dict, output_file)
                except TensorDumpError as ex:
                    raise file_write_error() from ex

                log.info("{} converted!", input_file.name)

        if model_config is None:
            return 0

        # Generate params.json to use with Hugging Face transformers.
        params = {
            "dim": model_config.model_dim,
            "n_layers": model_config.num_layers,
            "n_heads": model_config.num_attn_heads,
            "multiple_of": model_config.ffn_inner_dim_to_multiple,
            "rope_theta": model_config.rope_theta,
            "norm_eps": 1e-5,
        }

        if model_config.num_attn_heads != model_config.num_key_value_heads:
            params["n_kv_heads"] = model_config.num_key_value_heads

        if model_config.ffn_inner_dim_multiplier != 1.0:
            params["ffn_dim_multiplier"] = model_config.ffn_inner_dim_multiplier

        params_file = args.output_dir.joinpath("params.json")

        def param_write_error() -> CliCommandError:
            return CliCommandError(
                "params.json file cannot be created. See the nested exception for details."
            )

        try:
            fp = file_system.open_text(params_file, mode=FileMode.WRITE)
        except OSError as ex:
            raise param_write_error() from ex

        try:
            json.dump({"model": params}, fp)
        except OSError as ex:
            raise param_write_error() from ex
        finally:
            fp.close()

        log.info("params.json generated for {}!", args.model)

        return 0
