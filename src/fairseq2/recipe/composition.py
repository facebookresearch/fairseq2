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
from fairseq2.evaluator import Evaluator
from fairseq2.gang import Gangs
from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.generation.beam_search import (
    BeamSearchAlgorithm,
    StandardBeamSearchAlgorithm,
)
from fairseq2.generation.sampling import Sampler, TopKSampler, TopPSampler
from fairseq2.generator import Generator
from fairseq2.metrics import (
    format_as_byte_size,
    format_as_float,
    format_as_int,
    format_as_percentage,
    format_as_seconds,
)
from fairseq2.metrics.recorders import (
    CompositeMetricRecorder,
    MetricDescriptor,
    MetricRecorder,
    TensorBoardRecorder,
    WandbClient,
    WandbRecorder,
)
from fairseq2.model import Model
from fairseq2.nn.data_parallel import to_ddp, to_fsdp
from fairseq2.optim.lr_schedulers import LRScheduler
from fairseq2.profilers import Profiler
from fairseq2.recipe.asset_config import AssetConfigOverrider
from fairseq2.recipe.base import (
    EvalRecipe,
    GenerationRecipe,
    Recipe,
    RecipeContext,
    TrainRecipe,
)
from fairseq2.recipe.component import ComponentManager, register_component
from fairseq2.recipe.config import (
    ADAFACTOR_OPTIMIZER,
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
    AdafactorConfig,
    AdamWConfig,
    BeamSearchConfig,
    CosineAnnealingLRConfig,
    EvaluatorSection,
    GeneratorSection,
    MyleLRConfig,
    NoamLRConfig,
    PolynomialDecayLRConfig,
    SamplingConfig,
    TopKSamplerConfig,
    TopPSamplerConfig,
    TriStageLRConfig,
    get_config_section,
)
from fairseq2.recipe.config_preparer import RecipeConfigStructurer
from fairseq2.recipe.data_parallel import DPModelWrapper
from fairseq2.recipe.eval_model import (
    EvalModelBootstrapper,
    EvalModelFactory,
    EvalModelPreparer,
    load_eval_model,
)
from fairseq2.recipe.gang import _create_gangs
from fairseq2.recipe.logging import DistributedLogConfigurer
from fairseq2.recipe.metric_recorders import (
    WandbIdGenerator,
    WandbRunFactory,
    WandbRunIdManager,
    create_wandb_run,
    generate_wandb_id,
)
from fairseq2.recipe.model import (
    ModelBootstrapper,
    ModelMetadataSaver,
    ModelPreparer,
)
from fairseq2.recipe.recipe_preparer import (
    OutputDirectoryCreator,
    RecipeConfigDumper,
    RecipePreparer,
)
from fairseq2.recipe.sampling import SamplingFactory
from fairseq2.recipe.seed import SeedHolder
from fairseq2.recipe.sweep_tag import SweepTagGenerator
from fairseq2.recipe.task import TaskRunner
from fairseq2.recipe.tokenizer import TokenizerFactory, load_tokenizer
from fairseq2.recipe.torch import TorchConfigurer
from fairseq2.recipe.wire import (
    _AdafactorFactory,
    _AdamWFactory,
    _BeamSearchFactory,
    _CompositeMetricRecorder,
    _CompositeProfiler,
    _CosineAnnealingLRFactory,
    _DatasetFactory,
    _DDPModelWrapper,
    _DelegatingDPModelWrapper,
    _DelegatingEvalModelPreparer,
    _DelegatingModelPreparer,
    _DeviceStatTrackerFactory,
    _EvalModelFactory,
    _FSDPModelWrapper,
    _JsonlMetricRecorder,
    _LogMetricRecorder,
    _LRSchedulerFactory,
    _MaybeExtraAssetMetadataSource,
    _MaybeExtraModelMetadataSource,
    _MaybeTensorBoardRecorderFactory,
    _MaybeTorchProfilerFactory,
    _MaybeWandbRecorderFactory,
    _MetricRecorderFactory,
    _ModelFactory,
    _MyleLRFactory,
    _NoamLRFactory,
    _OptimizerFactory,
    _PolynomialDecayLRFactory,
    _RecipeEvalModelPreparer,
    _RecipeModelPreparer,
    _RecipePreparer,
    _SamplingFactory,
    _SeedHolder,
    _Seq2SeqGeneratorFactory,
    _SequenceGeneratorFactory,
    _StandardAssetConfigOverrider,
    _StandardCheckpointManager,
    _StandardComponentManager,
    _StandardDistributedLogConfigurer,
    _StandardEvalModelBootstrapper,
    _StandardEvalModelPreparer,
    _StandardModelBootstrapper,
    _StandardModelMetadataSaver,
    _StandardModelPreparer,
    _StandardOutputDirectoryCreator,
    _StandardRecipeConfigDumper,
    _StandardRecipeConfigStructurer,
    _StandardSweepTagGenerator,
    _StandardTorchConfigurer,
    _StandardWandbRunIdManager,
    _TaskRunner,
    _TensorBoardRecorder,
    _TokenizerFactory,
    _TriStageLRFactory,
    _WandbClientFactory,
    _WandbRecorder,
    _WorldPreparer,
)
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.task import Task
from fairseq2.trainer import Trainer
from fairseq2.utils.device_stat import DeviceStatTracker
from fairseq2.utils.stopwatch import Stopwatch
from fairseq2.world_info import WorldInfo


def _register_train_recipe(container: DependencyContainer, recipe: TrainRecipe) -> None:
    _register_common_recipe_objects(container)

    # Recipe
    container.register_instance(Recipe, recipe)

    container.register_instance(TrainRecipe, recipe)

    container.register_instance(type, recipe.config_kls, key="config_kls")

    # Model
    def create_model(resolver: DependencyResolver) -> Model:
        return _ModelFactory(resolver).create()

    container.register(Model, create_model)

    # DDP
    def create_ddp_model_wrapper(resolver: DependencyResolver) -> DPModelWrapper:
        recipe = resolver.resolve(TrainRecipe)

        context = RecipeContext(resolver)

        static_graph = recipe.has_static_autograd_graph(context)

        return _DDPModelWrapper(resolver, to_ddp, static_graph)

    container.register(DPModelWrapper, create_ddp_model_wrapper, key="ddp")

    # FSDP
    def create_fsdp_model_wrapper(resolver: DependencyResolver) -> DPModelWrapper:
        return _FSDPModelWrapper(resolver, to_fsdp)

    container.register(DPModelWrapper, create_fsdp_model_wrapper, key="fsdp")

    # Optimizer
    def create_optimizer(resolver: DependencyResolver) -> Optimizer:
        return _OptimizerFactory(resolver).create()

    container.register(Optimizer, create_optimizer)

    # LRScheduler
    def create_lr_scheduler(resolver: DependencyResolver) -> LRScheduler:
        return _LRSchedulerFactory(resolver).create()

    container.register(LRScheduler, create_lr_scheduler)

    # Wire
    container.register(ModelBootstrapper, _StandardModelBootstrapper)
    container.register(ModelMetadataSaver, _StandardModelMetadataSaver)
    container.register(ModelPreparer, _DelegatingModelPreparer)
    container.register(CheckpointManager, _StandardCheckpointManager)
    container.register(DPModelWrapper, _DelegatingDPModelWrapper)

    container.register(ModelPreparer, _RecipeModelPreparer, key="alt")
    container.register(ModelPreparer, _StandardModelPreparer, key="alt")

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
    _register_common_recipe_objects(container)

    # Recipe
    container.register_instance(Recipe, recipe)

    container.register_instance(EvalRecipe, recipe)

    container.register_instance(type, recipe.config_kls, key="config_kls")

    # Model
    def create_model(resolver: DependencyResolver) -> Model:
        section = get_config_section(resolver, "evaluator", EvaluatorSection)

        return load_eval_model(resolver, "model", section.dtype, section.amp)

    container.register(Model, create_model)

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
    _register_common_recipe_objects(container)

    # Recipe
    container.register_instance(Recipe, recipe)

    container.register_instance(GenerationRecipe, recipe)

    container.register_instance(type, recipe.config_kls, key="config_kls")

    # Model
    def create_model(resolver: DependencyResolver) -> Model:
        section = get_config_section(resolver, "generator", GeneratorSection)

        return load_eval_model(resolver, "model", section.dtype, section.amp)

    container.register(Model, create_model)

    # Generator
    @torch.inference_mode()
    def create_generator(resolver: DependencyResolver) -> Generator:
        context = RecipeContext(resolver)

        return recipe.create_generator(context)

    container.register(Task, create_generator)

    # Recipe Objects
    recipe.register(container)


def _register_common_recipe_objects(container: DependencyContainer) -> None:
    # Wall Watch
    wall_watch = Stopwatch()

    wall_watch.start()

    container.register_instance(Stopwatch, wall_watch)

    # WorldInfo
    def create_world_info(resolver: DependencyResolver) -> WorldInfo:
        return _WorldPreparer(resolver).prepare()

    container.register(WorldInfo, create_world_info)

    # DeviceStatTracker
    def create_device_stat_tracker(resolver: DependencyResolver) -> DeviceStatTracker:
        return _DeviceStatTrackerFactory(resolver).create()

    container.register(DeviceStatTracker, create_device_stat_tracker)

    # MetricRecorder
    def create_metric_recorder(resolver: DependencyResolver) -> MetricRecorder:
        return _MetricRecorderFactory(resolver).create()

    container.register(MetricRecorder, create_metric_recorder)

    # TensorBoardRecorder
    def create_tb_recorder(resolver: DependencyResolver) -> MetricRecorder | None:
        return _MaybeTensorBoardRecorderFactory(resolver).maybe_create()

    container.register(MetricRecorder, create_tb_recorder, key="alt")

    # WandbRecorder
    def create_wandb_recorder(resolver: DependencyResolver) -> MetricRecorder | None:
        return _MaybeWandbRecorderFactory(resolver).maybe_create()

    container.register(WandbRecorder, create_wandb_recorder, key="alt")

    # WandbClient
    def create_wandb_client(resolver: DependencyResolver) -> WandbClient:
        return _WandbClientFactory(resolver).create()

    container.register(WandbClient, create_wandb_client)

    # WandbRunFactory
    container.register_instance(WandbRunFactory, create_wandb_run)

    # WandbIdGenerator
    container.register_instance(WandbIdGenerator, generate_wandb_id)

    # TorchProfiler
    def create_torch_profiler(resolver: DependencyResolver) -> Profiler | None:
        return _MaybeTorchProfilerFactory(resolver).maybe_create()

    container.register(Profiler, create_torch_profiler, key="alt")

    # Asset Metadata
    def create_extra_asset_metadata_provider(
        resolver: DependencyResolver,
    ) -> AssetMetadataProvider | None:
        return _MaybeExtraAssetMetadataSource(resolver).maybe_load()

    container.register(AssetMetadataProvider, create_extra_asset_metadata_provider)

    # Model Metadata
    def create_extra_model_metadata_provider(
        resolver: DependencyResolver,
    ) -> AssetMetadataProvider | None:
        return _MaybeExtraModelMetadataSource(resolver).maybe_load()

    container.register(AssetMetadataProvider, create_extra_model_metadata_provider)

    # Dataset
    def create_dataset(resolver: DependencyResolver) -> object:
        return _DatasetFactory(resolver).create()

    container.register(object, create_dataset, key="dataset")

    # Tokenizer
    def create_tokenizer(resolver: DependencyResolver) -> Tokenizer:
        return load_tokenizer(resolver, section_name="tokenizer")

    container.register(Tokenizer, create_tokenizer)

    # Source Sequence Tokenizer
    def create_source_tokenizer(resolver: DependencyResolver) -> Tokenizer:
        return load_tokenizer(resolver, section_name="source_tokenizer")

    container.register(Tokenizer, create_source_tokenizer, key="source")

    # SequenceGenerator
    def create_seq_generator(resolver: DependencyResolver) -> SequenceGenerator:
        return _SequenceGeneratorFactory(resolver).create()

    container.register(SequenceGenerator, create_seq_generator)

    # Seq2SeqGenerator
    def create_seq2seq_generator(resolver: DependencyResolver) -> Seq2SeqGenerator:
        return _Seq2SeqGeneratorFactory(resolver).create()

    container.register(Seq2SeqGenerator, create_seq2seq_generator)

    # Wire
    container.register(AssetConfigOverrider, _StandardAssetConfigOverrider)
    container.register(ComponentManager, _StandardComponentManager)
    container.register(CompositeMetricRecorder, _CompositeMetricRecorder)
    container.register(RecipeConfigDumper, _StandardRecipeConfigDumper)
    container.register(DistributedLogConfigurer, _StandardDistributedLogConfigurer)
    container.register(MetricRecorder, _JsonlMetricRecorder, key="alt")
    container.register(MetricRecorder, _LogMetricRecorder, key="alt")
    container.register(Profiler, _CompositeProfiler)
    container.register(SeedHolder, _SeedHolder)
    container.register(RecipeConfigStructurer, _StandardRecipeConfigStructurer)
    container.register(OutputDirectoryCreator, _StandardOutputDirectoryCreator)
    container.register(RecipePreparer, _RecipePreparer)
    container.register(SweepTagGenerator, _StandardSweepTagGenerator)
    container.register(TaskRunner, _TaskRunner)
    container.register(TensorBoardRecorder, _TensorBoardRecorder)
    container.register(TokenizerFactory, _TokenizerFactory)
    container.register(EvalModelFactory, _EvalModelFactory)
    container.register(EvalModelBootstrapper, _StandardEvalModelBootstrapper)
    container.register(EvalModelPreparer, _DelegatingEvalModelPreparer)
    container.register(TorchConfigurer, _StandardTorchConfigurer)
    container.register(WandbRecorder, _WandbRecorder)
    container.register(WandbRunIdManager, _StandardWandbRunIdManager)

    container.register(EvalModelPreparer, _RecipeEvalModelPreparer, key="alt")
    container.register(EvalModelPreparer, _StandardEvalModelPreparer, key="alt")

    # Gangs
    container.register(Gangs, _create_gangs)

    # Metrics
    _register_metric_descriptors(container)

    # Components
    _register_sampling_generators(container)

    _register_samplers(container)

    _register_beam_search_generators(container)

    _register_beam_search_algorithms(container)


def _register_optimizers(container: DependencyContainer) -> None:
    # AdamW
    def create_adamw(resolver: DependencyResolver, config: AdamWConfig) -> Optimizer:
        return _AdamWFactory(resolver).create(config)

    register_component(
        container,
        Optimizer,
        ADAMW_OPTIMIZER,
        config_kls=AdamWConfig,
        factory=create_adamw,
    )

    # Adafactor
    def create_adafactor(
        resolver: DependencyResolver, config: AdafactorConfig
    ) -> Optimizer:
        return _AdafactorFactory(resolver).create(config)

    register_component(
        container,
        Optimizer,
        ADAFACTOR_OPTIMIZER,
        config_kls=AdafactorConfig,
        factory=create_adafactor,
    )


def _register_lr_schedulers(container: DependencyContainer) -> None:
    # CosineAnnealingLR
    def create_cosine_annealing_lr(
        resolver: DependencyResolver, config: CosineAnnealingLRConfig
    ) -> LRScheduler:
        return _CosineAnnealingLRFactory(resolver).create(config)

    register_component(
        container,
        LRScheduler,
        COSINE_ANNEALING_LR,
        config_kls=CosineAnnealingLRConfig,
        factory=create_cosine_annealing_lr,
    )

    # MyleLR
    def create_myle_lr(
        resolver: DependencyResolver, config: MyleLRConfig
    ) -> LRScheduler:
        return _MyleLRFactory(resolver).create(config)

    register_component(
        container,
        LRScheduler,
        MYLE_LR,
        config_kls=MyleLRConfig,
        factory=create_myle_lr,
    )

    # NoamLR
    def create_noam_lr(
        resolver: DependencyResolver, config: NoamLRConfig
    ) -> LRScheduler:
        return _NoamLRFactory(resolver).create(config)

    register_component(
        container,
        LRScheduler,
        NOAM_LR,
        config_kls=NoamLRConfig,
        factory=create_noam_lr,
    )

    # PolynomialDecayLR
    def create_polynomial_decay_lr(
        resolver: DependencyResolver, config: PolynomialDecayLRConfig
    ) -> LRScheduler:
        return _PolynomialDecayLRFactory(resolver).create(config)

    register_component(
        container,
        LRScheduler,
        POLYNOMIAL_DECAY_LR,
        config_kls=PolynomialDecayLRConfig,
        factory=create_polynomial_decay_lr,
    )

    # TriStageLR
    def create_tri_stage_lr(
        resolver: DependencyResolver, config: TriStageLRConfig
    ) -> LRScheduler:
        return _TriStageLRFactory(resolver).create(config)

    register_component(
        container,
        LRScheduler,
        TRI_STAGE_LR,
        config_kls=TriStageLRConfig,
        factory=create_tri_stage_lr,
    )


def _register_sampling_generators(container: DependencyContainer) -> None:
    container.register(SamplingFactory, _SamplingFactory)

    # Sequence
    def create_seq_generator(
        resolver: DependencyResolver, config: SamplingConfig
    ) -> SequenceGenerator:
        return _SamplingFactory(resolver).create_seq_generator(config)

    register_component(
        container,
        SequenceGenerator,
        SAMPLING_GENERATOR,
        config_kls=SamplingConfig,
        factory=create_seq_generator,
    )

    # Seq2Seq
    def create_seq2seq_generator(
        resolver: DependencyResolver, config: SamplingConfig
    ) -> Seq2SeqGenerator:
        return _SamplingFactory(resolver).create_seq2seq_generator(config)

    register_component(
        container,
        Seq2SeqGenerator,
        SAMPLING_GENERATOR,
        config_kls=SamplingConfig,
        factory=create_seq2seq_generator,
    )


def _register_samplers(container: DependencyContainer) -> None:
    # Top-P
    def create_top_p_sampler(
        resolver: DependencyResolver, config: TopPSamplerConfig
    ) -> Sampler:
        return TopPSampler(p=config.p)

    register_component(
        container,
        Sampler,
        TOP_P_SAMPLER,
        config_kls=TopPSamplerConfig,
        factory=create_top_p_sampler,
    )

    # Top-K
    def create_top_k_sampler(
        resolver: DependencyResolver, config: TopKSamplerConfig
    ) -> Sampler:
        return TopKSampler(k=config.k)

    register_component(
        container,
        Sampler,
        TOP_K_SAMPLER,
        config_kls=TopKSamplerConfig,
        factory=create_top_k_sampler,
    )


def _register_beam_search_generators(container: DependencyContainer) -> None:
    # Sequence
    def create_seq_generator(
        resolver: DependencyResolver, config: BeamSearchConfig
    ) -> SequenceGenerator:
        return _BeamSearchFactory(resolver).create_seq_generator(config)

    register_component(
        container,
        SequenceGenerator,
        BEAM_SEARCH_GENERATOR,
        config_kls=BeamSearchConfig,
        factory=create_seq_generator,
    )

    # Seq2Seq
    def create_seq2seq_generator(
        resolver: DependencyResolver, config: BeamSearchConfig
    ) -> Seq2SeqGenerator:
        return _BeamSearchFactory(resolver).create_seq2seq_generator(config)

    register_component(
        container,
        Seq2SeqGenerator,
        BEAM_SEARCH_GENERATOR,
        config_kls=BeamSearchConfig,
        factory=create_seq2seq_generator,
    )


def _register_beam_search_algorithms(container: DependencyContainer) -> None:
    # Standard
    def create_standard_algorithm(
        resolver: DependencyResolver, config: None
    ) -> BeamSearchAlgorithm:
        return StandardBeamSearchAlgorithm()

    register_component(
        container,
        BeamSearchAlgorithm,
        STANDARD_BEAM_SEARCH_ALGO,
        config_kls=NoneType,
        factory=create_standard_algorithm,
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
