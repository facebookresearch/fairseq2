# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys
import warnings
from argparse import ArgumentParser, Namespace
from itertools import count
from pathlib import Path
from typing import final
from warnings import catch_warnings

import torch

from fairseq2.logging import get_log_writer
from fairseq2.models.llama import load_llama_config
from fairseq2.models.llama.integ import convert_to_reference_checkpoint
from fairseq2.recipes.cli import CliCommandHandler
from fairseq2.recipes.logging import console, setup_basic_logging
from fairseq2.typing import override

log = get_log_writer(__name__)


@final
class ConvertCheckpointCommand(CliCommandHandler):
    """Converts fairseq2 LLaMA checkpoints to reference checkpoints."""

    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--arch",
            metavar="ARCH_NAME",
            help="architecture name to generate params.json",
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
    def __call__(self, args: Namespace) -> None:
        setup_basic_logging()

        if not args.input_dir.exists() or not args.input_dir.is_dir():
            log.error("`input_dir` must be a directory.")

            sys.exit(1)

        if args.output_dir.exists():
            log.error("`output_dir` must not exist.")

            sys.exit(1)

        if args.arch:
            model_config = load_llama_config(args.arch)
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
        with console.status("[bold green]Converting...") as status:
            for input_file, output_file in zip(input_files, output_files):
                status.update(f"[bold green]Loading {input_file.name}...")

                try:
                    with catch_warnings():
                        warnings.simplefilter("ignore")

                        checkpoint = torch.load(input_file, weights_only=True)
                except RuntimeError:
                    log.exception(
                        "Checkpoint file {} cannot be loaded", input_file.name
                    )

                    sys.exit(1)

                if "model" not in checkpoint:
                    log.error("Checkpoint file {} does not contain a 'model' entry.", input_file.name)  # fmt: skip

                    sys.exit(1)

                status.update(f"[bold green]Converting {input_file.name}...")

                ref_state_dict = convert_to_reference_checkpoint(checkpoint)

                try:
                    torch.save(ref_state_dict, output_file)
                except RuntimeError:
                    log.exception(
                        "Checkpoint file {} cannot be saved", output_file.name
                    )

                    sys.exit(1)

                log.info("{} converted!", input_file.name)

        # Generate a basic params.json, mainly to use with HG transformers.
        if model_config is not None:
            params = {
                "model": {
                    "dim": model_config.model_dim,
                    "n_layers": model_config.num_layers,
                    "n_heads": model_config.num_attn_heads,
                    "ffn_dim_multiplier": model_config.ffn_inner_dim_scale,
                    "multiple_of": model_config.ffn_inner_dim_to_multiple,
                    "norm_eps": 1e-5,
                },
            }

            if model_config.num_attn_heads != model_config.num_key_value_heads:
                params["model"]["n_kv_heads"] = model_config.num_key_value_heads

            try:
                with args.output_dir.joinpath("params.json").open("w") as fp:
                    json.dump(params, fp)
            except RuntimeError:
                log.exception("params.json cannot be created.")

                sys.exit(1)

            log.info("params.json generated for {}.", args.arch)
