# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from types import NoneType
from typing import Any

from torch.optim import Optimizer

from fairseq2.assets import AssetMetadataProvider
from fairseq2.checkpoint import CheckpointManager
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.dependency import DependencyContainer
from fairseq2.gang import Gangs
from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.metrics import (
    format_as_byte_size,
    format_as_float,
    format_as_int,
    format_as_percentage,
    format_as_seconds,
)
from fairseq2.metrics.recorders import MetricDescriptor
from fairseq2.optim.lr_schedulers import LRScheduler
from fairseq2.recipe.assets import (
    load_checkpoint_asset_provider,
    load_extra_asset_provider,
)
from fairseq2.recipe.base_model import load_base_model
from fairseq2.recipe.beam_search import (
    create_beam_search_seq2seq_generator,
    create_beam_search_seq_generator,
    create_standard_beam_search_algorithm,
)
from fairseq2.recipe.checkpoint import create_checkpoint_manager
from fairseq2.recipe.component import register_component
from fairseq2.recipe.config import (
    ADAMW_OPTIMIZER,
    BEAM_SEARCH_GENERATOR,
    COSINE_ANNEALING_LR,
    JSONL_METRIC_RECORDER,
    LOG_METRIC_RECORDER,
    MYLE_LR,
    NOAM_LR,
    POLYNOMIAL_DECAY_LR,
    SAMPLING_GENERATOR,
    SEQ2SEQ_BEAM_SEARCH_GENERATOR,
    SEQ2SEQ_SAMPLING_GENERATOR,
    STANDARD_BEAM_SEARCH_ALGO,
    TENSORBOARD_RECORDER,
    TOP_K_SAMPLER,
    TOP_P_SAMPLER,
    TORCH_PROFILER,
    TRI_STAGE_LR,
    WANDB_RECORDER,
    AdamWConfig,
    BeamSearchConfig,
    CosineAnnealingLRConfig,
    JsonlMetricRecorderConfig,
    LogMetricRecorderConfig,
    MyleLRConfig,
    NoamLRConfig,
    PolynomialDecayLRConfig,
    SamplingConfig,
    TensorBoardRecorderConfig,
    TopKSamplerConfig,
    TopPSamplerConfig,
    TorchProfilerConfig,
    TriStageLRConfig,
    WandbRecorderConfig,
)
from fairseq2.recipe.data_parallel_model import setup_data_parallel_model
from fairseq2.recipe.dataset import load_dataset
from fairseq2.recipe.gang import setup_gangs
from fairseq2.recipe.lr_schedulers import (
    create_cosine_annealing_lr,
    create_lr_scheduler,
    create_myle_lr,
    create_noam_lr,
    create_polynomial_decay_lr,
    create_tri_stage_lr,
)
from fairseq2.recipe.metric_recorders import (
    create_jsonl_metric_recorder,
    create_log_metric_recorder,
    create_tensorboard_recorder,
    create_wandb_recorder,
)
from fairseq2.recipe.model import Model
from fairseq2.recipe.optim import create_adamw, create_optimizer
from fairseq2.recipe.profilers import create_torch_profiler
from fairseq2.recipe.reference_model import (
    load_base_eval_model,
    load_base_generator_model,
    setup_eval_model,
    setup_generator_model,
)
from fairseq2.recipe.rng import create_seed_holder
from fairseq2.recipe.sampling import (
    create_sampling_seq2seq_generator,
    create_sampling_seq_generator,
    create_top_k_sampler,
    create_top_p_sampler,
)
from fairseq2.recipe.seq_generator import (
    create_seq2seq_generator,
    create_seq_generator,
)
from fairseq2.recipe.tokenizer import (
    load_source_tokenizer,
    load_target_tokenizer,
    load_tokenizer,
)
from fairseq2.utils.rng import SeedHolder
from fairseq2.utils.stopwatch import Stopwatch


def register_recipe_common(container: DependencyContainer) -> None:
    # Wall Watch
    wall_watch = Stopwatch()

    wall_watch.start()

    container.register_instance(Stopwatch, wall_watch)

    # SeedHolder
    container.register(SeedHolder, create_seed_holder)

    # Extra Assets
    container.register(AssetMetadataProvider, load_extra_asset_provider)

    container.register(AssetMetadataProvider, load_checkpoint_asset_provider)

    # Gangs
    container.register(Gangs, setup_gangs)

    # CheckpointManager
    container.register(CheckpointManager, create_checkpoint_manager)

    # Base Models
    container.register(Model, load_base_model, key="base")

    container.register(Model, load_base_eval_model, key="base_eval")

    container.register(Model, load_base_generator_model, key="base_generator")

    # Models
    container.register(Model, setup_data_parallel_model)

    container.register(Model, setup_eval_model, key="eval")

    container.register(Model, setup_generator_model, key="generator")

    # Dataset
    container.register(object, load_dataset, key="dataset")

    # Tokenizer
    container.register(Tokenizer, load_tokenizer)

    container.register(Tokenizer, load_source_tokenizer, key="source")
    container.register(Tokenizer, load_target_tokenizer, key="target")

    # Optimizer
    container.register(Optimizer, create_optimizer)

    # LRScheduler
    container.register(LRScheduler, create_lr_scheduler)

    # SequenceGenerator
    container.register(SequenceGenerator, create_seq_generator)

    # Seq2SeqGenerator
    container.register(Seq2SeqGenerator, create_seq2seq_generator)

    # Metrics
    register_metric_descriptors(container)

    # Components
    register_optimizers(container)
    register_lr_schedulers(container)
    register_seq_generators(container)
    register_samplers(container)
    register_seq2seq_generators(container)
    register_beam_search_algorithms(container)
    register_metric_recorders(container)
    register_profilers(container)


def register_optimizers(container: DependencyContainer) -> None:
    register_component(
        container,
        ADAMW_OPTIMIZER,
        AdamWConfig,
        factory=create_adamw,
    )


def register_lr_schedulers(container: DependencyContainer) -> None:
    register_component(
        container,
        COSINE_ANNEALING_LR,
        CosineAnnealingLRConfig,
        factory=create_cosine_annealing_lr,
    )

    register_component(
        container,
        MYLE_LR,
        MyleLRConfig,
        factory=create_myle_lr,
    )

    register_component(
        container,
        NOAM_LR,
        NoamLRConfig,
        factory=create_noam_lr,
    )

    register_component(
        container,
        POLYNOMIAL_DECAY_LR,
        PolynomialDecayLRConfig,
        factory=create_polynomial_decay_lr,
    )

    register_component(
        container,
        TRI_STAGE_LR,
        TriStageLRConfig,
        factory=create_tri_stage_lr,
    )


def register_seq_generators(container: DependencyContainer) -> None:
    register_component(
        container,
        SAMPLING_GENERATOR,
        SamplingConfig,
        factory=create_sampling_seq_generator,
    )

    register_component(
        container,
        BEAM_SEARCH_GENERATOR,
        BeamSearchConfig,
        factory=create_beam_search_seq_generator,
    )


def register_samplers(container: DependencyContainer) -> None:
    register_component(
        container,
        TOP_P_SAMPLER,
        TopPSamplerConfig,
        factory=create_top_p_sampler,
    )

    register_component(
        container,
        TOP_K_SAMPLER,
        TopKSamplerConfig,
        factory=create_top_k_sampler,
    )


def register_seq2seq_generators(container: DependencyContainer) -> None:
    register_component(
        container,
        SEQ2SEQ_SAMPLING_GENERATOR,
        SamplingConfig,
        factory=create_sampling_seq2seq_generator,
    )

    register_component(
        container,
        SEQ2SEQ_BEAM_SEARCH_GENERATOR,
        BeamSearchConfig,
        factory=create_beam_search_seq2seq_generator,
    )


def register_beam_search_algorithms(container: DependencyContainer) -> None:
    register_component(
        container,
        STANDARD_BEAM_SEARCH_ALGO,
        NoneType,
        factory=create_standard_beam_search_algorithm,
    )


def register_metric_recorders(container: DependencyContainer) -> None:
    register_component(
        container,
        JSONL_METRIC_RECORDER,
        JsonlMetricRecorderConfig,
        factory=create_jsonl_metric_recorder,
    )

    register_component(
        container,
        LOG_METRIC_RECORDER,
        LogMetricRecorderConfig,
        factory=create_log_metric_recorder,
    )

    register_component(
        container,
        TENSORBOARD_RECORDER,
        TensorBoardRecorderConfig,
        factory=create_tensorboard_recorder,
    )

    register_component(
        container,
        WANDB_RECORDER,
        WandbRecorderConfig,
        factory=create_wandb_recorder,
    )


def register_profilers(container: DependencyContainer) -> None:
    register_component(
        container,
        TORCH_PROFILER,
        TorchProfilerConfig,
        factory=create_torch_profiler,
    )


def register_metric_descriptors(container: DependencyContainer) -> None:
    def register(name: str, *args: Any, **kwargs: Any) -> None:
        container.register_instance(
            MetricDescriptor, MetricDescriptor(name, *args, **kwargs), key=name
        )

    # fmt: off
    register("loss",             "Loss",                   90, format_as_float)
    register("contrastive_loss", "Contrastive Loss",      100, format_as_float)
    register("ctc_loss",         "CTC Loss",              100, format_as_float)
    register("diversity_loss",   "Diversity Loss",        100, format_as_float)
    register("nll_loss",         "NLL Loss",              100, format_as_float)
    register("features_penalty", "Features Penalty",      110, format_as_float)
    register("accuracy",         "Accuracy",              200, format_as_float, higher_better=True)
    register("bleu",             "BLEU",                  200, format_as_float, higher_better=True)
    register("chrf",             "chrF++",                200, format_as_float, higher_better=True)
    register("uer",              "Unit Error Rate (UER)", 200, format_as_float)
    register("wer",              "Word Error Rate (WER)", 200, format_as_float)
    register("code_perplexity",  "Code Perplexity",       210, format_as_float)
    register("prob_perplexity",  "Prob Perplexity",       210, format_as_float)
    register("temperature",      "Temperature",           220, format_as_float)
    register("grad_norm",        "Gradient Norm",         300, format_as_float)
    register("data_epoch",       "Data Epoch",            490, format_as_int)
    register("data_time",        "Data Time",             500, format_as_seconds)
    register("compute_time",     "Compute Time",          501, format_as_seconds)
    register("lapse_time",       "Lapse Time",            502, format_as_seconds)
    register("total_time",       "Total Time",            505, format_as_seconds)
    register("wall_time",        "Wall Time",             510, format_as_seconds)
    register("lr",               "Learning Rate",         700, format_as_float)
    register("loss_scale",       "Loss Scale",            710, format_as_float)

    # Batch Metrics
    register("batch_size",                "Batch Size",                      800, format_as_int)
    register("elements_per_batch",        "Elements per Batch",              800, format_as_int)
    register("elements_per_second",       "Elements per Second",             810, format_as_int)
    register("num_examples",              "Number of Examples",              820, format_as_int)
    register("num_elements",              "Number of Elements",              830, format_as_int)
    register("num_source_elements",       "Number of Source Elements",       830, format_as_int)
    register("num_target_elements",       "Number of Target Elements",       830, format_as_int)
    register("padding",                   "Padding",                         835, format_as_int)
    register("padding_ratio",             "Padding Ratio (%)",               835, format_as_percentage)
    register("total_num_examples",        "Total Number of Examples",        840, format_as_int)
    register("total_num_elements",        "Total Number of Elements",        850, format_as_int)
    register("total_num_source_elements", "Total Number of Source Elements", 850, format_as_int)
    register("total_num_target_elements", "Total Number of Target Elements", 850, format_as_int)

    # Sequence Generator Metrics
    register("generator_prefill_size",        "Generator/Prefill Size",        900, format_as_int)
    register("generator_num_elements",        "Generator/Number of Elements",  901, format_as_int)
    register("generator_elements_per_second", "Generator/Elements per Second", 902, format_as_int)
    register("generator_cache_size",          "Generator/Cache Size",          903, format_as_byte_size)
    register("generator_cache_capacity",      "Generator/Cache Capacity",      904, format_as_byte_size)

    # Preference Optimization
    register("cpo_loss",         "CPO Loss",                             0, format_as_float)
    register("dpo_loss",         "DPO Loss",                             0, format_as_float)
    register("orpo_loss",        "ORPO Loss",                            0, format_as_float)
    register("simpo_loss",       "SimPO Loss",                           0, format_as_float)
    register("chosen_logps",     "Chosen Sequence Log Probabilities",   50, format_as_float)
    register("rejected_logps",   "Rejected Sequence Log Probabilities", 50, format_as_float)
    register("chosen_lengths",   "Chosen Sequence Length",              70, format_as_float)
    register("rejected_lengths", "Rejected Sequence Length",            70, format_as_float)

    # Memory
    register("peak_active_mem_bytes",   "Peak Active Device Memory",       920, format_as_byte_size)
    register("peak_active_mem_ratio",   "Peak Active Device Memory (%)",   920, format_as_percentage)
    register("peak_reserved_mem_bytes", "Peak Reserved Device Memory",     925, format_as_byte_size)
    register("peak_reserved_mem_ratio", "Peak Reserved Device Memory (%)", 925, format_as_percentage)
    # fmt: on
