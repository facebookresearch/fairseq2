# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import sys
import warnings
from argparse import ArgumentParser, Namespace
from itertools import count
from pathlib import Path
from typing import final
from warnings import catch_warnings

from typing_extensions import override

from fairseq2.assets import default_asset_store
from fairseq2.logging import get_log_writer
from fairseq2.models.llama import get_llama_model_hub
from fairseq2.models.llama.integ import convert_to_reference_checkpoint
from fairseq2.recipes.cli import CliCommandHandler
from fairseq2.recipes.utils.rich import get_error_console
from fairseq2.utils.file import dump_torch_tensors, load_torch_tensors

log = get_log_writer(__name__)


@final
class ConvertCheckpointCommandHandler(CliCommandHandler):
    """Converts fairseq2 LLaMA checkpoints to reference checkpoints."""

    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            metavar="ARCH_NAME",
            help="model name to fetch architecture to generate params.json",
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
    def run(self, parser: ArgumentParser, args: Namespace) -> int:
        if not args.input_dir.exists() or not args.input_dir.is_dir():
            log.error("`input_dir` must be a directory.")

            sys.exit(1)

        if args.output_dir.exists():
            log.error("`output_dir` must not exist.")

            sys.exit(1)

        arch = (
            default_asset_store.retrieve_card(args.model).field("model_arch").as_(str)
        )

        if arch:
            model_config = get_llama_model_hub().load_config(args.model)
        else:
            model_config = None

        input_files = []

        # Determine input checkpoint files.
        input_file = args.input_dir.joinpath("model.pt")
        if input_file.exists():
            input_files.append(input_file)
        else:
            for shard_idx in count():
                input_file = args.input_dir.joinpath(f"model.{shard_idx}.pt")
                if not input_file.exists():
                    break

                input_files.append(input_file)

        if not input_files:
            log.error("`input_dir` must contain a model checkpoint file (i.e. model.pt)")  # fmt: skip

            sys.exit(1)

        output_files = []

        # Determine output checkpoint filenames.
        for shard_idx in range(len(input_files)):
            output_file = args.output_dir.joinpath(f"consolidated.{shard_idx:02d}.pth")

            output_files.append(output_file)

        args.output_dir.mkdir(parents=True)

        # Begin conversion.
        with get_error_console().status("[bold green]Converting...") as status:
            for input_file, output_file in zip(input_files, output_files):
                status.update(f"[bold green]Loading {input_file.name}...")

                try:
                    with catch_warnings():
                        warnings.simplefilter("ignore")

                        checkpoint = load_torch_tensors(input_file)
                except RuntimeError:
                    log.exception(
                        "Checkpoint file {} cannot be loaded.", input_file.name
                    )

                    sys.exit(1)

                if all(key not in checkpoint for key in ["model_key", "model"]):
                    log.error("Checkpoint file {} does not contain a 'model_key' nor 'model' entry.", input_file.name)  # fmt: skip

                    sys.exit(1)

                status.update(
                    f"[bold green]Converting {input_file.name} to {output_file.name}..."
                )

                ref_state_dict = convert_to_reference_checkpoint(checkpoint)

                try:
                    dump_torch_tensors(ref_state_dict, output_file)
                except RuntimeError:
                    log.exception("Checkpoint file {} cannot be saved.", output_file.name)  # fmt: skip

                    sys.exit(1)

                log.info("{} converted!", input_file.name)

        # Generate a basic params.json, mainly to use with HG transformers.
        if model_config is not None:
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

            try:
                with args.output_dir.joinpath("params.json").open("w") as fp:
                    json.dump({"model", params}, fp)
            except RuntimeError:
                log.exception("params.json cannot be created.")

                sys.exit(1)

            log.info("params.json generated for {}.", args.model)

        return 0
