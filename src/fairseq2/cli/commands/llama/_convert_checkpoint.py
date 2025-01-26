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

from fairseq2.assets import AssetCardNotFoundError
from fairseq2.cli import CliCommandHandler
from fairseq2.context import RuntimeContext
from fairseq2.logging import log
from fairseq2.models import ModelHandler, UnknownModelArchitectureError
from fairseq2.models.llama import LLAMA_MODEL_FAMILY, LLaMAConfig
from fairseq2.models.llama.integ import convert_to_reference_llama_checkpoint
from fairseq2.recipes.utils.rich import get_error_console
from fairseq2.utils.file import (
    FileMode,
    TensorDumpError,
    TensorLoadError,
    TorchTensorDumper,
    TorchTensorLoader,
)


@final
class ConvertLLaMACheckpointHandler(CliCommandHandler):
    """Converts fairseq2 LLaMA checkpoints to reference checkpoints."""

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

        try:
            if file_system.exists(args.input_dir):
                input_exists = file_system.is_dir(args.input_dir)
            else:
                input_exists = False
        except OSError:
            log.exception("Input directory cannot be read. See the logged stack trace for details.")  # fmt: skip

            return 1

        if not input_exists:
            parser.error("input must be a directory")

            return 1

        try:
            output_exists = file_system.exists(args.output_dir)
        except OSError:
            log.exception("Output directory cannot be read. See the logged stack trace for details.")  # fmt: skip

            return 1

        if output_exists:
            parser.error("output directory already exists")

            return 1

        # Load the model configuration.
        try:
            card = context.asset_store.retrieve_card(args.model)
        except AssetCardNotFoundError:
            parser.error("unknown LLaMA model. Use `fairseq2 assets list` to see the available models.")  # fmt: skip

            return 1

        model_handlers = context.get_registry(ModelHandler)

        try:
            model_handler = model_handlers.get(LLAMA_MODEL_FAMILY)
        except LookupError:
            log.error("LLaMA model handler cannot be found. Please file a bug report.")  # fmt: skip

            return 1

        try:
            model_config = model_handler.load_config(card)
        except UnknownModelArchitectureError:
            log.error("Model has an unknown architecture. Please file a bug report to the model author.")  # fmt: skip

            return 1

        if not isinstance(model_config, LLaMAConfig):
            log.error("Model configuration has an invalid type. Please file a bug report.")  # fmt: skip

            return 1

        # Determine input checkpoint files.
        input_file = args.input_dir.joinpath("model.pt")

        try:
            input_file_exists = file_system.exists(input_file)
        except OSError:
            log.exception("Input directory cannot be read. See the logged stack trace for details.")  # fmt: skip

            return 1

        input_files = []

        if input_file_exists:
            input_files.append(input_file)
        else:
            for shard_idx in count():
                input_file = args.input_dir.joinpath(f"model.{shard_idx}.pt")

                try:
                    input_file_exists = file_system.exists(input_file)
                except OSError:
                    log.exception("Input directory cannot be read. See the logged stack trace for details.")  # fmt: skip

                    return 1

                if not input_file_exists:
                    break

                input_files.append(input_file)

        if not input_files:
            parser.error("input directory must contain a model checkpoint file (i.e. model.pt)")  # fmt: skip

            return 1

        output_files = []

        # Determine output checkpoint filenames.
        for shard_idx in range(len(input_files)):
            output_file = args.output_dir.joinpath(f"consolidated.{shard_idx:02d}.pth")

            output_files.append(output_file)

        try:
            file_system.make_directory(args.output_dir)
        except OSError:
            log.exception("Output directory cannot be created. See the logged stack trace for details.")  # fmt: skip

            return 1

        # Begin conversion.
        console = get_error_console()

        tensor_loader = TorchTensorLoader(context.file_system)
        tensor_dumper = TorchTensorDumper(context.file_system)

        with console.status("[bold green]Converting...") as status:
            for input_file, output_file in zip(input_files, output_files):
                status.update(f"[bold green]Loading {input_file.name}...")

                try:
                    checkpoint = tensor_loader.load(input_file)
                except TensorLoadError:
                    log.exception("{} checkpoint file cannot be loaded. See the logged stack trace for details.", input_file.name)  # fmt: skip

                    return 1

                status.update(f"[bold green]Converting {input_file.name} to {output_file.name}...")  # fmt: skip

                try:
                    ref_state_dict = convert_to_reference_llama_checkpoint(checkpoint)
                except (TypeError, KeyError):
                    log.exception("{} checkpoint file cannot be converted. See the logged stack trace for details.", input_file.name)  # fmt: skip

                    return 1

                try:
                    tensor_dumper.dump(ref_state_dict, output_file)
                except TensorDumpError:
                    log.exception("{} checkpoint file cannot be saved. See the logged stack trace for details.", output_file.name)  # fmt: skip

                    return 1

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

        try:
            fp = file_system.open_text(params_file, mode=FileMode.WRITE)
        except OSError:
            log.exception("params.json file cannot be created. See the logged stack trace for details.")  # fmt: skip

            return 1

        try:
            json.dump({"model": params}, fp)
        except (OSError, RuntimeError):
            log.exception(
                "params.json file cannot be created. See the logged stack trace for details."
            )

            return 1
        finally:
            fp.close()

        log.info("params.json generated for {}.", args.model)

        return 0
