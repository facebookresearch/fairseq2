# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from fairseq2.assets import AssetNotFoundError
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import load_text_tokenizer
from fairseq2.datasets.batching import LengthBatching
from fairseq2.datasets.instruction import (
    GenericInstructionDataset,
    load_generic_instruction_dataset,
)
from fairseq2.logging import get_log_writer
from fairseq2.models import load_model
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.transformer import enable_memory_efficient_torch_sdpa
from fairseq2.recipes.evaluator import Evaluator
from fairseq2.recipes.lm.instruction_finetune import (
    InstructionFinetuneCriterion,
    InstructionValidUnit,
)
from fairseq2.recipes.utils.asset import (
    AssetReference,
    asset_as_path,
    retrieve_asset_card,
)
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import setup_gangs, to_data_parallel
from fairseq2.typing import CPU, META, DataType
from fairseq2.utils.profiler import Stopwatch
from fairseq2.utils.rng import manual_seed

log = get_log_writer(__name__)


@dataclass(kw_only=True)
class NLLEvalConfig:
    """Holds configuration of the perplexity evaluator recipe"""

    # Data
    dataset: AssetReference = "foo"
    """The name, path or path to the asset card of the dataset to evaluate on."""

    model: AssetReference = "llama3_1_8b"
    """The name or path to the asset card of the wav2vec 2.0 model to evaluate."""

    checkpoint_dir: Path | None = None
    """The checkpoint directory containing models saved by :class:`FileCheckpointManager`."""

    dtype: DataType = torch.bfloat16
    """The data type of the model."""

    data_parallelism: Literal["ddp", "fsdp"] = "fsdp"
    """The data parallelism API to use."""

    fsdp_wrap_granularity: Literal["layer", "stack", "model"] = "layer"
    """The granularity at which to wrap the model."""

    fsdp_reshard_after_forward: bool = True
    """If ``True``, reshards the parameters only after the backward pass."""

    tensor_parallel_size: int = 1
    """The size of tensor parallelism."""

    mixed_precision: Literal["none", "static", "dynamic"] = "static"
    """
    If 'none', the whole training will be run in `dtype`. If 'static', forward
    and backward passes will be run in `dtype`, but the optimizer step will be
    run in full precision. If 'dynamic', forward and backward passes will be run
    with `torch.amp` in `dtype`, but the optimizer step will be run in full
    precision.
    """

    max_num_tokens: int = 8192 * 2
    """The maximum number of tokens per batch."""

    max_seq_len: int = 8192
    """The maximum sequence length."""

    valid_split: str = "default"
    """The name of the valid data split."""

    example_shuffle_window: int = 10_000
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 1000
    """The size of the sliding window for shuffling batches."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    seed: int = 2
    """The random number generator seed to use."""


nll_eval_presets = ConfigRegistry[NLLEvalConfig]()

nll_eval_preset = nll_eval_presets.decorator


@nll_eval_preset("llama3_1_base_eval")
def _llama3_1_base_eval() -> NLLEvalConfig:
    return NLLEvalConfig()


@torch.inference_mode()
def load_nll_evaluator(
    config: NLLEvalConfig, output_dir: Path
) -> Evaluator[SequenceBatch]:
    wall_watch = Stopwatch(start=True)

    root_gang, gangs = setup_gangs(log, tp_size=config.tensor_parallel_size)

    dp_gang = gangs["dp"]  # data
    tp_gang = gangs["tp"]  # tensor

    # Load the tokenizer.
    model_card = retrieve_asset_card(config.model)

    log.info("Loading {} tokenizer.", model_card.name)

    tokenizer = load_text_tokenizer(model_card)

    log.info("Tokenizer loaded.")

    # Load the dataset.
    try:
        dataset_card = retrieve_asset_card(config.dataset)
    except AssetNotFoundError:
        dataset_card = None

    if dataset_card is not None:
        log.info("Loading {} preference optimization dataset.", dataset_card.name)

        dataset = load_generic_instruction_dataset(dataset_card)

        log.info("Dataset loaded.")
    else:
        dataset_path = asset_as_path(config.dataset)

        dataset = GenericInstructionDataset.from_path(dataset_path)

    seed = config.seed

    # Load the model
    manual_seed(seed, CPU, root_gang.device)

    seed += 1

    init_device = META

    dtype = config.dtype if config.mixed_precision == "none" else torch.float32

    gangs = {"dp": dp_gang, "tp": tp_gang}

    model_card = retrieve_asset_card(config.model)

    log.info("Loading {} model on data parallel rank 0 (per shard).", model_card.name)  # fmt: skip

    if dp_gang.rank == 0:
        init_device = root_gang.device
    model = load_model(
        model_card,
        gangs=gangs,
        device=init_device,
        dtype=dtype,
    )

    root_gang.barrier()

    log.info("Model loaded on rank 0.")

    if not isinstance(model, DecoderModel):
        raise ValueError(
            f"The model must be of type `{DecoderModel}`, but is of type `{type(model)}` instead."
        )

    mp_dtype = config.dtype if config.mixed_precision == "static" else None

    dp_model = to_data_parallel(
        model,
        dp_gang,
        config.data_parallelism,
        log,
        fsdp_broadcast_state=True,  # loading checkpoints not supported
        fsdp_reshard_after_forward=config.fsdp_reshard_after_forward,
        fsdp_mixed_precision_dtype=mp_dtype,
        fsdp_fp32_reduce=True,
        fsdp_wrap_granularity=config.fsdp_wrap_granularity,
    )

    enable_memory_efficient_torch_sdpa(dp_model, False)

    log_model(dp_model, log, rank=root_gang.rank)

    # loading the eval unit
    criterion = InstructionFinetuneCriterion(dp_model)
    unit = InstructionValidUnit(criterion, dp_gang)

    data_reader = dataset.create_reader(
        config.valid_split,
        tokenizer,
        dp_gang,
        config.max_seq_len,
        batching=LengthBatching(config.max_num_tokens),
        example_shuffle_window=config.example_shuffle_window,
        batch_shuffle_window=config.batch_shuffle_window,
        sync_mode="until_last",
        num_accumulate=1,
        num_prefetch=config.num_prefetch,
        seed=seed,
    )

    # TODO: Fix once we support static mixed precision on one device.
    if config.mixed_precision == "static":
        amp = root_gang.size == 1 or config.data_parallelism != "fsdp"
    else:
        amp = config.mixed_precision == "dynamic"

    # Initialize the evaluator.
    return Evaluator[SequenceBatch](
        units=[unit],
        data_readers=[data_reader],
        root_gang=root_gang,
        dtype=config.dtype,
        amp=amp,
        tb_dir=output_dir.joinpath("tb"),
        metrics_dir=output_dir.joinpath("metrics"),
        seed=seed,
        wall_watch=wall_watch,
    )
