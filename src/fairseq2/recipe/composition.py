# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from types import NoneType
from typing import Any

import torch
from torch.optim import Optimizer

from fairseq2.assets import AssetMetadataProvider
from fairseq2.checkpoint import CheckpointManager
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.device import Device
from fairseq2.evaluator import Evaluator
from fairseq2.gang import Gangs
from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.generation.beam_search import BeamSearchAlgorithm
from fairseq2.generation.sampling import Sampler
from fairseq2.generator import Generator
from fairseq2.metrics import (
    format_as_byte_size,
    format_as_float,
    format_as_int,
    format_as_percentage,
    format_as_seconds,
)
from fairseq2.metrics.recorders import MetricDescriptor
from fairseq2.model.context import ModelContext
from fairseq2.optim.lr_schedulers import LRScheduler
from fairseq2.recipe.assets import (
    _maybe_load_checkpoint_assets,
    _maybe_load_extra_assets,
)
from fairseq2.recipe.base import (
    EvalRecipe,
    GenerationRecipe,
    RecipeContext,
    TrainRecipe,
)
from fairseq2.recipe.beam_search import (
    _create_beam_search_seq2seq_generator,
    _create_beam_search_seq_generator,
    _create_standard_beam_search_algorithm,
)
from fairseq2.recipe.checkpoint import _create_checkpoint_manager
from fairseq2.recipe.cluster import WorldInfo, _create_world_info
from fairseq2.recipe.component import (
    ComponentManager,
    _create_component_manager,
    register_component,
)
from fairseq2.recipe.config import (
    ADAMW_OPTIMIZER,
    BEAM_SEARCH_GENERATOR,
    COSINE_ANNEALING_LR,
    MYLE_LR,
    NOAM_LR,
    POLYNOMIAL_DECAY_LR,
    SAMPLING_GENERATOR,
    STANDARD_BEAM_SEARCH_ALGO,
    TOP_K_SAMPLER,
    TOP_P_SAMPLER,
    TRI_STAGE_LR,
    AdamWConfig,
    BeamSearchConfig,
    CosineAnnealingLRConfig,
    MyleLRConfig,
    NoamLRConfig,
    PolynomialDecayLRConfig,
    SamplingConfig,
    TopKSamplerConfig,
    TopPSamplerConfig,
    TriStageLRConfig,
)
from fairseq2.recipe.dataset import _open_dataset
from fairseq2.recipe.device import _create_device
from fairseq2.recipe.eval_model import (
    _load_eval_model,
    _load_generator_model,
    _prepare_eval_model,
)
from fairseq2.recipe.gang import _create_gangs
from fairseq2.recipe.lr_schedulers import (
    _create_cosine_annealing_lr,
    _create_lr_scheduler,
    _create_myle_lr,
    _create_noam_lr,
    _create_polynomial_decay_lr,
    _create_tri_stage_lr,
)
from fairseq2.recipe.optim import _create_adamw, _create_optimizer
from fairseq2.recipe.progress import _create_progress_reporter
from fairseq2.recipe.rng import _create_seed_holder
from fairseq2.recipe.sampling import (
    _create_sampling_seq2seq_generator,
    _create_sampling_seq_generator,
    _create_top_k_sampler,
    _create_top_p_sampler,
)
from fairseq2.recipe.seq_generator import (
    _create_seq2seq_generator,
    _create_seq_generator,
)
from fairseq2.recipe.threading import _create_thread_pool
from fairseq2.recipe.tokenizer import (
    _load_source_tokenizer,
    _load_target_tokenizer,
    _load_tokenizer,
)
from fairseq2.recipe.train_model import (
    _create_or_load_model,
    _prepare_model,
)
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.task import Task
from fairseq2.trainer import Trainer
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.rng import SeedHolder
from fairseq2.utils.stopwatch import Stopwatch
from fairseq2.utils.threading import ThreadPool


def _register_train_recipe(container: DependencyContainer, recipe: TrainRecipe) -> None:
    register_common_recipe_objects(container)

    # Static Graph (for DDP)
    def has_static_graph(resolver: DependencyResolver) -> bool:
        context = RecipeContext(resolver)

        return recipe.has_static_autograd_graph(context)

    container.register(bool, has_static_graph, key="static_graph")

    # CheckpointManager
    container.register(CheckpointManager, _create_checkpoint_manager)

    # Model
    def create_model(resolver: DependencyResolver) -> ModelContext:
        context = RecipeContext(resolver)

        model_context = _create_or_load_model(resolver)

        model_context = recipe.prepare_model(context, model_context)

        return _prepare_model(resolver, model_context)

    container.register(ModelContext, create_model)

    # Optimizer
    container.register(Optimizer, _create_optimizer)

    # LRScheduler
    container.register(LRScheduler, _create_lr_scheduler)

    # Trainer
    def create_trainer(resolver: DependencyResolver) -> Trainer:
        context = RecipeContext(resolver)

        return recipe.create_trainer(context)

    container.register(Task, create_trainer)

    # Components
    _register_optimizers(container)

    _register_lr_schedulers(container)

    # Recipe Objects
    recipe.register(container)


def _register_eval_recipe(container: DependencyContainer, recipe: EvalRecipe) -> None:
    register_common_recipe_objects(container)

    # Model
    def load_model(resolver: DependencyResolver) -> ModelContext:
        context = RecipeContext(resolver)

        model_context = _load_eval_model(resolver)

        model_context = recipe.prepare_model(context, model_context)

        return _prepare_eval_model(resolver, model_context)

    container.register(ModelContext, load_model)

    # Evaluator
    @torch.inference_mode()
    def create_evaluator(resolver: DependencyResolver) -> Evaluator:
        context = RecipeContext(resolver)

        return recipe.create_evaluator(context)

    container.register(Task, create_evaluator)

    # Recipe Objects
    recipe.register(container)


def _register_generation_recipe(
    container: DependencyContainer, recipe: GenerationRecipe
) -> None:
    register_common_recipe_objects(container)

    # Model
    def load_model(resolver: DependencyResolver) -> ModelContext:
        context = RecipeContext(resolver)

        model_context = _load_generator_model(resolver)

        model_context = recipe.prepare_model(context, model_context)

        return _prepare_eval_model(resolver, model_context)

    container.register(ModelContext, load_model)

    # Generator
    @torch.inference_mode()
    def create_generator(resolver: DependencyResolver) -> Generator:
        context = RecipeContext(resolver)

        return recipe.create_generator(context)

    container.register(Task, create_generator)

    # Recipe Objects
    recipe.register(container)


def register_common_recipe_objects(container: DependencyContainer) -> None:
    # Wall Watch
    wall_watch = Stopwatch()

    wall_watch.start()

    container.register_instance(Stopwatch, wall_watch)

    # SeedHolder
    container.register(SeedHolder, _create_seed_holder)

    # WorldInfo
    container.register(WorldInfo, _create_world_info)

    # ProgressReporter
    container.register(ProgressReporter, _create_progress_reporter)

    # ThreadPool
    container.register(ThreadPool, _create_thread_pool)

    # ComponentManager
    container.register(ComponentManager, _create_component_manager)

    # Device
    container.register(Device, _create_device)

    # Extra Assets
    container.register(AssetMetadataProvider, _maybe_load_extra_assets)

    container.register(AssetMetadataProvider, _maybe_load_checkpoint_assets)

    # Gangs
    container.register(Gangs, _create_gangs)

    # Dataset
    container.register(object, _open_dataset, key="dataset")

    # Tokenizer
    container.register(Tokenizer, _load_tokenizer)

    container.register(Tokenizer, _load_source_tokenizer, key="source")
    container.register(Tokenizer, _load_target_tokenizer, key="target")

    # SequenceGenerator
    container.register(SequenceGenerator, _create_seq_generator)

    # Seq2SeqGenerator
    container.register(Seq2SeqGenerator, _create_seq2seq_generator)

    # Metrics
    _register_metric_descriptors(container)

    # Components
    _register_seq_generators(container)

    _register_samplers(container)

    _register_seq2seq_generators(container)

    _register_beam_search_algorithms(container)


def _register_optimizers(container: DependencyContainer) -> None:
    register_component(
        container,
        Optimizer,
        ADAMW_OPTIMIZER,
        config_kls=AdamWConfig,
        factory=_create_adamw,
    )


def _register_lr_schedulers(container: DependencyContainer) -> None:
    register_component(
        container,
        LRScheduler,
        COSINE_ANNEALING_LR,
        config_kls=CosineAnnealingLRConfig,
        factory=_create_cosine_annealing_lr,
    )

    register_component(
        container,
        LRScheduler,
        MYLE_LR,
        config_kls=MyleLRConfig,
        factory=_create_myle_lr,
    )

    register_component(
        container,
        LRScheduler,
        NOAM_LR,
        config_kls=NoamLRConfig,
        factory=_create_noam_lr,
    )

    register_component(
        container,
        LRScheduler,
        POLYNOMIAL_DECAY_LR,
        config_kls=PolynomialDecayLRConfig,
        factory=_create_polynomial_decay_lr,
    )

    register_component(
        container,
        LRScheduler,
        TRI_STAGE_LR,
        config_kls=TriStageLRConfig,
        factory=_create_tri_stage_lr,
    )


def _register_seq_generators(container: DependencyContainer) -> None:
    register_component(
        container,
        SequenceGenerator,
        SAMPLING_GENERATOR,
        config_kls=SamplingConfig,
        factory=_create_sampling_seq_generator,
    )

    register_component(
        container,
        SequenceGenerator,
        BEAM_SEARCH_GENERATOR,
        config_kls=BeamSearchConfig,
        factory=_create_beam_search_seq_generator,
    )


def _register_samplers(container: DependencyContainer) -> None:
    register_component(
        container,
        Sampler,
        TOP_P_SAMPLER,
        config_kls=TopPSamplerConfig,
        factory=_create_top_p_sampler,
    )

    register_component(
        container,
        Sampler,
        TOP_K_SAMPLER,
        config_kls=TopKSamplerConfig,
        factory=_create_top_k_sampler,
    )


def _register_seq2seq_generators(container: DependencyContainer) -> None:
    register_component(
        container,
        Seq2SeqGenerator,
        SAMPLING_GENERATOR,
        config_kls=SamplingConfig,
        factory=_create_sampling_seq2seq_generator,
    )

    register_component(
        container,
        Seq2SeqGenerator,
        BEAM_SEARCH_GENERATOR,
        config_kls=BeamSearchConfig,
        factory=_create_beam_search_seq2seq_generator,
    )


def _register_beam_search_algorithms(container: DependencyContainer) -> None:
    register_component(
        container,
        BeamSearchAlgorithm,
        STANDARD_BEAM_SEARCH_ALGO,
        config_kls=NoneType,
        factory=_create_standard_beam_search_algorithm,
    )


def _register_metric_descriptors(container: DependencyContainer) -> None:
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
