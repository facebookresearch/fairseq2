# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import Generic, Sequence, TypeVar, cast, final

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from typing_extensions import override

from fairseq2.assets import (
    AssetCardNotFoundError,
    AssetStore,
    FileAssetMetadataProvider,
    StandardMetadataFileLoader,
)
from fairseq2.checkpoint import (
    CheckpointError,
    CheckpointManager,
    FileCheckpointManager,
    FileCheckpointMetadataProvider,
)
from fairseq2.context import RuntimeContext
from fairseq2.data.text.tokenizers import (
    TextTokenizer,
    TextTokenizerHandler,
    UnknownTextTokenizerError,
    UnknownTextTokenizerFamilyError,
    get_text_tokenizer_family,
    resolve_text_tokenizer_reference,
)
from fairseq2.datasets import (
    DataReader,
    DatasetHandler,
    UnknownDatasetError,
    UnknownDatasetFamilyError,
    get_dataset_family,
)
from fairseq2.error import NotSupportedError, SetupError
from fairseq2.gang import Gangs, setup_default_gang, setup_parallel_gangs
from fairseq2.generation import (
    Seq2SeqGenerator,
    Seq2SeqGeneratorHandler,
    SequenceGenerator,
    SequenceGeneratorHandler,
    UnknownSeq2SeqGeneratorError,
    UnknownSequenceGeneratorError,
)
from fairseq2.logging import log
from fairseq2.metrics import MetricDescriptor, UnknownMetricDescriptorError
from fairseq2.metrics.recorders import (
    MetricRecorder,
    MetricRecorderHandler,
    UnknownMetricRecorderError,
)
from fairseq2.models import (
    ModelHandler,
    UnknownModelError,
    UnknownModelFamilyError,
    get_model_family,
)
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.fsdp import get_fsdp_wrap_policy
from fairseq2.nn.checkpointing import use_layerwise_activation_checkpointing
from fairseq2.nn.ddp import to_ddp
from fairseq2.nn.fsdp import to_fsdp
from fairseq2.nn.utils.module import broadcast_module, to_device
from fairseq2.optim import OptimizerHandler, UnknownOptimizerError
from fairseq2.optim.lr_scheduler import (
    LRScheduler,
    LRSchedulerHandler,
    NoopLR,
    UnknownLRSchedulerError,
)
from fairseq2.recipes.config import (
    AssetsSection,
    DatasetSection,
    EvalRecipeConfig,
    GangSection,
    GenerateRecipeConfig,
    MetricsSection,
    ModelSection,
    Seq2SeqGeneratorSection,
    SequenceGeneratorSection,
    TrainRecipeConfig,
)
from fairseq2.recipes.evaluator import Evaluator, EvalUnit
from fairseq2.recipes.generator import Generator, GeneratorUnit
from fairseq2.recipes.trainer import Trainer, TrainUnit
from fairseq2.registry import Provider
from fairseq2.typing import DataType
from fairseq2.utils.file import TorchTensorDumper, TorchTensorLoader
from fairseq2.utils.profiler import NoopProfiler, Profiler, TorchProfiler
from fairseq2.utils.yaml import StandardYamlDumper, StandardYamlLoader


def setup_gangs(context: RuntimeContext, config_section: GangSection) -> Gangs:
    log.info("Initializing the root gang.")

    timeout = timedelta(minutes=config_section.timeout)

    root_gang = setup_default_gang(
        device=context.device, timeout=timeout, monitored=config_section.monitored
    )

    log.info("Root gang initialized.")

    log.info("Initializing the parallel gangs.")

    gangs = setup_parallel_gangs(root_gang, tp_size=config_section.tensor_parallel_size)

    log.info("Parallel gangs initialized.")

    return gangs


def register_extra_asset_paths(
    context: RuntimeContext, config_section: AssetsSection
) -> None:
    file_system = context.file_system

    yaml_loader = StandardYamlLoader(file_system)

    metadata_file_loader = StandardMetadataFileLoader(yaml_loader)

    path = config_section.extra_path
    if path is not None:
        if not file_system.exists(path):
            log.warning("The '{}' path pointed to by the `extra_asset_card_path` configuration does not exist.", path)  # fmt: skip

            return

        path = file_system.resolve(path)

        context.asset_store.user_metadata_providers.append(
            FileAssetMetadataProvider(path, file_system, metadata_file_loader)
        )

    path = config_section.checkpoint_dir
    if path is not None:
        metadata_file = path.joinpath("model.yaml")
        if not file_system.exists(metadata_file):
            log.warning("The checkpoint metadata file (model.yaml) is not found under the '{}' directory. Make sure that the `checkpoint_search_dir` configuration points to the base checkpoint directory used during training.", path)  # fmt: skip

            return

        path = file_system.resolve(path)

        context.asset_store.user_metadata_providers.append(
            FileCheckpointMetadataProvider(path, file_system, metadata_file_loader)
        )


def create_checkpoint_manager(
    context: RuntimeContext, gangs: Gangs, output_dir: Path
) -> CheckpointManager:
    checkpoint_dir = output_dir.joinpath("checkpoints")

    file_system = context.file_system

    tensor_loader = TorchTensorLoader(file_system, restrict=False)

    tensor_dumper = TorchTensorDumper(file_system)

    yaml_dumper = StandardYamlDumper(file_system)

    return FileCheckpointManager(
        checkpoint_dir, gangs, file_system, tensor_loader, tensor_dumper, yaml_dumper
    )


def create_optimizer(
    context: RuntimeContext, config: TrainRecipeConfig, model: Module
) -> Optimizer:
    optimizer_handlers = context.get_registry(OptimizerHandler)

    factory = RecipeOptimizerFactory(optimizer_handlers)

    return factory.create(config, model)


@final
class RecipeOptimizerFactory:
    _optimizer_handlers: Provider[OptimizerHandler]

    def __init__(self, optimizer_handlers: Provider[OptimizerHandler]) -> None:
        self._optimizer_handlers = optimizer_handlers

    def create(self, config: TrainRecipeConfig, model: Module) -> Optimizer:
        name = config.optimizer.name

        try:
            handler = self._optimizer_handlers.get(name)
        except LookupError:
            raise UnknownOptimizerError(name) from None

        return handler.create(model.parameters(), config.optimizer.config)


def create_lr_scheduler(
    context: RuntimeContext, config: TrainRecipeConfig, optimizer: Optimizer
) -> LRScheduler:
    lr_scheduler_handlers = context.get_registry(LRSchedulerHandler)

    factory = RecipeLRSchedulerFactory(lr_scheduler_handlers)

    return factory.create(config, optimizer)


@final
class RecipeLRSchedulerFactory:
    _lr_scheduler_handlers: Provider[LRSchedulerHandler]

    def __init__(self, lr_scheduler_handlers: Provider[LRSchedulerHandler]) -> None:
        self._lr_scheduler_handlers = lr_scheduler_handlers

    def create(self, config: TrainRecipeConfig, optimizer: Optimizer) -> LRScheduler:
        name = config.lr_scheduler.name
        if name is None:
            return NoopLR(optimizer)

        try:
            handler = self._lr_scheduler_handlers.get(name)
        except LookupError:
            raise UnknownLRSchedulerError(name) from None

        return handler.create(
            optimizer, config.lr_scheduler.config, config.regime.num_steps
        )


BatchT = TypeVar("BatchT")


def create_trainer(
    context: RuntimeContext,
    config: TrainRecipeConfig,
    output_dir: Path,
    unit: TrainUnit[BatchT],
    data_reader: DataReader[BatchT],
    valid_units: Sequence[EvalUnit[BatchT]],
    valid_data_readers: Sequence[DataReader[BatchT]],
    gangs: Gangs,
    checkpoint_manager: CheckpointManager,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    seed: int,
) -> Trainer[BatchT]:
    metric_recorders = create_metric_recorders(context, config.metrics, output_dir)

    metric_name = config.regime.score_metric
    if metric_name is not None:
        metric_descriptors = context.get_registry(MetricDescriptor)

        try:
            score_metric_descriptor = metric_descriptors.get(metric_name)
        except LookupError:
            raise UnknownMetricDescriptorError(metric_name) from None
    else:
        score_metric_descriptor = None

    trainer = config.trainer

    profiler: Profiler

    if trainer.profile is not None:
        num_skip_steps, num_record_steps = trainer.profile

        profile_dir = output_dir.joinpath("tb")

        profiler = TorchProfiler(
            num_skip_steps, num_record_steps, profile_dir, gangs.root
        )
    else:
        profiler = NoopProfiler()

    # TODO: Fix once we support static mixed precision on one device.
    if trainer.mixed_precision == "static":
        amp = gangs.root.size == 1 or trainer.data_parallelism != "fsdp"
    else:
        amp = trainer.mixed_precision == "dynamic"

    regime = config.regime

    return Trainer[BatchT](
        unit=unit,
        data_reader=data_reader,
        root_gang=gangs.root,
        dp_gang=gangs.dp,
        tp_gang=gangs.tp,
        dtype=trainer.dtype,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        fp16_loss_scale=trainer.fp16_loss_scale,
        max_gradient_norm=trainer.max_gradient_norm,
        amp=amp,
        max_num_steps=regime.num_steps,
        max_num_data_epochs=regime.num_data_epochs,
        score_metric_descriptor=score_metric_descriptor,
        lower_better=regime.lower_score_better,
        valid_units=valid_units,
        valid_data_readers=valid_data_readers,
        validate_after_n_steps=regime.validate_after_n_steps,
        validate_every_n_steps=regime.validate_every_n_steps,
        validate_after_n_data_epochs=regime.validate_after_n_data_epochs,
        validate_every_n_data_epochs=regime.validate_every_n_data_epochs,
        checkpoint_manager=checkpoint_manager,
        checkpoint_after_n_steps=regime.checkpoint_after_n_steps,
        checkpoint_every_n_steps=regime.checkpoint_every_n_steps,
        checkpoint_after_n_data_epochs=regime.checkpoint_after_n_data_epochs,
        checkpoint_every_n_data_epochs=regime.checkpoint_every_n_data_epochs,
        keep_last_n_checkpoints=regime.keep_last_n_checkpoints,
        keep_best_n_checkpoints=regime.keep_best_n_checkpoints,
        keep_last_n_models=regime.keep_last_n_models,
        keep_best_n_models=regime.keep_best_n_models,
        metric_recorders=metric_recorders,
        publish_metrics_after_n_steps=regime.publish_metrics_after_n_steps,
        publish_metrics_every_n_steps=regime.publish_metrics_every_n_steps,
        publish_metrics_after_n_data_epochs=regime.publish_metrics_after_n_data_epochs,
        publish_metrics_every_n_data_epochs=regime.publish_metrics_every_n_data_epochs,
        profiler=profiler,
        anomaly_detection=trainer.anomaly_detection,
        wall_watch=context.wall_watch,
        seed=seed,
    )


def create_evaluator(
    context: RuntimeContext,
    config: EvalRecipeConfig,
    output_dir: Path,
    units: Sequence[EvalUnit[BatchT]],
    data_readers: Sequence[DataReader[BatchT]],
    gangs: Gangs,
    seed: int,
) -> Evaluator[BatchT]:
    metric_recorders = create_metric_recorders(context, config.metrics, output_dir)

    return Evaluator[BatchT](
        units=units,
        data_readers=data_readers,
        gangs=gangs,
        dtype=config.evaluator.dtype,
        amp=config.evaluator.amp,
        metric_recorders=metric_recorders,
        wall_watch=context.wall_watch,
        seed=seed,
    )


def create_generator(
    context: RuntimeContext,
    config: GenerateRecipeConfig,
    output_dir: Path,
    unit: GeneratorUnit[BatchT],
    data_reader: DataReader[BatchT],
    gangs: Gangs,
    seed: int,
) -> Generator[BatchT]:
    metric_recorders = create_metric_recorders(context, config.metrics, output_dir)

    return Generator[BatchT](
        unit=unit,
        data_reader=data_reader,
        gangs=gangs,
        dtype=config.generator.dtype,
        amp=config.generator.amp,
        metric_recorders=metric_recorders,
        wall_watch=context.wall_watch,
        seed=seed,
    )


def create_metric_recorders(
    context: RuntimeContext, config_section: MetricsSection, output_dir: Path
) -> list[MetricRecorder]:
    recorder_handlers = context.get_registry(MetricRecorderHandler)

    factory = RecipeMetricRecorderFactory(recorder_handlers)

    return factory.create(config_section, output_dir)


@final
class RecipeMetricRecorderFactory:
    _recorder_handlers: Provider[MetricRecorderHandler]

    def __init__(self, recorder_handlers: Provider[MetricRecorderHandler]) -> None:
        self._recorder_handlers = recorder_handlers

    def create(
        self, config_section: MetricsSection, output_dir: Path
    ) -> list[MetricRecorder]:
        recorders = []

        for name, config in config_section.recorders.items():
            try:
                recorder_handler = self._recorder_handlers.get(name)
            except LookupError:
                raise UnknownMetricRecorderError(name) from None

            recorder = recorder_handler.create(output_dir, config)

            recorders.append(recorder)

        return recorders


DatasetT = TypeVar("DatasetT")


def load_dataset(
    kls: type[DatasetT],
    context: RuntimeContext,
    config_section: DatasetSection,
    gangs: Gangs,
) -> DatasetT:
    dataset_handlers = context.get_registry(DatasetHandler)

    loader = RecipeDatasetLoader(kls, context.asset_store, dataset_handlers)

    dataset = loader.load(config_section, gangs)

    return cast(DatasetT, dataset)


@final
class RecipeDatasetLoader:
    _kls: type[object]
    _asset_store: AssetStore
    _dataset_handlers: Provider[DatasetHandler]

    def __init__(
        self,
        kls: type[object],
        asset_store: AssetStore,
        dataset_handlers: Provider[DatasetHandler],
    ) -> None:
        self._kls = kls
        self._asset_store = asset_store
        self._dataset_handlers = dataset_handlers

    def load(self, config_section: DatasetSection, gangs: Gangs) -> object:
        name = config_section.name
        path = config_section.path

        if path is not None:
            family = config_section.family

            try:
                handler = self._dataset_handlers.get(family)
            except LookupError:
                raise UnknownDatasetFamilyError(family) from None

            if not issubclass(handler.kls, self._kls):
                raise TypeError(
                    f"The dataset must be of type `{self._kls}`, but is of type `{handler.kls}` instead."
                )

            log.info("Loading the dataset.")

            dataset = handler.load_from_path(path, name=path.stem)
        elif name is not None:
            try:
                card = self._asset_store.retrieve_card(name)
            except AssetCardNotFoundError:
                raise UnknownDatasetError(name) from None

            family = get_dataset_family(card)

            try:
                handler = self._dataset_handlers.get(family)
            except LookupError:
                raise UnknownDatasetFamilyError(family, name) from None

            if not issubclass(handler.kls, self._kls):
                raise TypeError(
                    f"The '{name}' dataset must be of type `{self._kls}`, but is of type `{handler.kls}` instead."
                )

            log.info("Loading the '{}' dataset.", name)

            dataset = handler.load(card)
        else:
            raise ValueError(
                "Either `config_section.name` or `config_section.path` must be specified."
            )

        gangs.root.barrier()

        log.info("The dataset is loaded.")

        return dataset


def load_text_tokenizer(
    context: RuntimeContext, model_name: str | None, tokenizer_name: str | None = None
) -> TextTokenizer:
    tokenizer_handlers = context.get_registry(TextTokenizerHandler)

    loader = RecipeTextTokenizerLoader(context.asset_store, tokenizer_handlers)

    return loader.load(model_name, tokenizer_name)


@final
class RecipeTextTokenizerLoader:
    _asset_store: AssetStore
    _tokenizer_handlers: Provider[TextTokenizerHandler]

    def __init__(
        self,
        asset_store: AssetStore,
        tokenizer_handlers: Provider[TextTokenizerHandler],
    ) -> None:
        self._asset_store = asset_store
        self._tokenizer_handlers = tokenizer_handlers

    def load(self, model_name: str | None, tokenizer_name: str | None) -> TextTokenizer:
        name = tokenizer_name
        if name is None:
            name = model_name
            if name is None:
                raise ValueError(
                    "Either `model_name` or `tokenizer_name` must be specified."
                )

        try:
            card = self._asset_store.retrieve_card(name)
        except AssetCardNotFoundError:
            raise UnknownTextTokenizerError(name) from None

        card = resolve_text_tokenizer_reference(self._asset_store, card)

        family = get_text_tokenizer_family(card)

        try:
            handler = self._tokenizer_handlers.get(family)
        except LookupError:
            raise UnknownTextTokenizerFamilyError(family, name) from None

        log.info("Loading the '{}' tokenizer.", name)

        tokenizer = handler.load(card)

        log.info("The tokenizer is loaded.")

        return tokenizer


ModelT = TypeVar("ModelT", bound=Module)


def load_model(
    kls: type[ModelT],
    context: RuntimeContext,
    config: TrainRecipeConfig,
    gangs: Gangs,
    checkpoint_manager: CheckpointManager,
) -> ModelT:
    model_handlers = context.get_registry(ModelHandler)

    if config.model.name is not None:
        loader = RecipeModelLoader(
            kls, context.asset_store, model_handlers, checkpoint_manager
        )

        model = loader.load(config, gangs)
    elif config.model.family is not None:
        factory = RecipeModelFactory(kls, model_handlers)

        model = factory.create(config, gangs)
    else:
        raise ValueError(
            "Either `config.model.name` or `config.model.family` must be specified."
        )

    return cast(ModelT, model)


@final
class RecipeModelLoader:
    _kls: type[object]
    _asset_store: AssetStore
    _model_handlers: Provider[ModelHandler]
    _checkpoint_manager: CheckpointManager

    def __init__(
        self,
        kls: type[object],
        asset_store: AssetStore,
        model_handlers: Provider[ModelHandler],
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self._kls = kls
        self._asset_store = asset_store
        self._model_handlers = model_handlers
        self._checkpoint_manager = checkpoint_manager

    def load(self, config: TrainRecipeConfig, gangs: Gangs) -> Module:
        name = config.model.name
        if name is None:
            raise ValueError("`config.model.name` must not be `None`.")

        try:
            card = self._asset_store.retrieve_card(name)
        except AssetCardNotFoundError:
            raise UnknownModelError(name) from None

        family = get_model_family(card)

        try:
            handler = self._model_handlers.get(family)
        except LookupError:
            raise UnknownModelFamilyError(family, card.name) from None

        if not issubclass(handler.kls, self._kls):
            raise TypeError(
                f"The model must be of type `{self._kls}`, but is of type `{handler.kls}` instead."
            )

        model_config = config.model.config

        if not isinstance(config.model.config, handler.config_kls):
            raise TypeError(
                f"`config.model.config` must be of type `{handler.config_kls}`, but is of type `{type(model_config)}` instead."
            )

        trainer = config.trainer

        # Load the model.
        dtype = trainer.dtype if trainer.mixed_precision is None else torch.float32

        def create_model(meta: bool) -> Module:
            return handler.create(model_config, gangs, dtype, meta)

        # Shortcut if there is a training checkpoint. No need to load the model.
        if self._checkpoint_manager.has_checkpoint():
            model = create_model(handler.supports_meta)

            gangs.root.barrier()

            return model

        log.info("Loading the '{}' model on data parallel rank 0 (per shard).", name)  # fmt: skip

        if gangs.dp.rank == 0:
            model = handler.load(card, gangs, dtype, config.model.config)
        else:
            model = create_model(handler.supports_meta)

        gangs.root.barrier()

        log.info("The model is loaded on data parallel rank 0.")

        return model


@final
class RecipeModelFactory:
    _kls: type[object]
    _model_handlers: Provider[ModelHandler]

    def __init__(
        self, kls: type[object], model_handlers: Provider[ModelHandler]
    ) -> None:
        self._kls = kls
        self._model_handlers = model_handlers

    def create(self, config: TrainRecipeConfig, gangs: Gangs) -> Module:
        family = config.model.family
        if family is None:
            raise ValueError("`config.model.family` must not be `None`.")

        try:
            handler = self._model_handlers.get(family)
        except LookupError:
            raise UnknownModelFamilyError(family) from None

        if not issubclass(handler.kls, self._kls):
            raise TypeError(
                f"The model must be of type `{self._kls}`, but is of type `{handler.kls}` instead."
            )

        model_config = config.model.config

        if not isinstance(config.model.config, handler.config_kls):
            raise TypeError(
                f"`config.model.config` must be of type `{handler.config_kls}`, but is of type `{type(model_config)}` instead."
            )

        trainer = config.trainer

        # Create the model.
        dtype = trainer.dtype if trainer.mixed_precision is None else torch.float32

        try:
            return handler.create(model_config, gangs, dtype, handler.supports_meta)
        except NotSupportedError as ex:
            raise SetupError(
                "The model cannot be initialized due to an unsupported feature. See the nested exception for details."
            ) from ex


def save_checkpoint_card(
    context: RuntimeContext,
    config: TrainRecipeConfig,
    checkpoint_manager: CheckpointManager,
    tokenizer_name: str | None = None,
) -> None:
    model_family = config.model.family
    if model_family is None:
        raise ValueError("`config.model.family` must not be `None`.")

    model_config = config.model.config
    if model_config is None:
        raise ValueError("`config.model.config` must not be `None`.")

    if tokenizer_name is None:
        tokenizer_name = config.model.name
        if tokenizer_name is not None:
            try:
                card = context.asset_store.retrieve_card(tokenizer_name)
            except AssetCardNotFoundError:
                raise UnknownTextTokenizerError(tokenizer_name) from None

            card = resolve_text_tokenizer_reference(context.asset_store, card)

            tokenizer_name = card.name

    try:
        checkpoint_manager.save_checkpoint_card(
            model_family, model_config, tokenizer_name
        )
    except CheckpointError as ex:
        raise SetupError(
            "The checkpoint model card cannot be saved. See the nested exception for details."
        ) from ex


def load_eval_model(
    kls: type[ModelT],
    context: RuntimeContext,
    model_name: str,
    gangs: Gangs,
    dtype: DataType,
    mixed_precision: bool,
) -> ModelT:
    model_handlers = context.get_registry(ModelHandler)

    loader = RecipeEvalModelLoader(kls, context.asset_store, model_handlers)

    return loader.load(model_name, gangs, dtype, mp=mixed_precision)


class EvalModelLoader(ABC, Generic[ModelT]):
    @abstractmethod
    def load(
        self, model_name: str, gangs: Gangs, dtype: DataType, *, mp: bool = False
    ) -> ModelT:
        ...


@final
class RecipeEvalModelLoader(EvalModelLoader[ModelT]):
    _kls: type[ModelT]
    _asset_store: AssetStore
    _model_handlers: Provider[ModelHandler]

    def __init__(
        self,
        kls: type[ModelT],
        asset_store: AssetStore,
        model_handlers: Provider[ModelHandler],
    ) -> None:
        self._kls = kls
        self._asset_store = asset_store
        self._model_handlers = model_handlers

    @override
    def load(
        self, model_name: str, gangs: Gangs, dtype: DataType, *, mp: bool = False
    ) -> ModelT:
        try:
            card = self._asset_store.retrieve_card(model_name)
        except AssetCardNotFoundError:
            raise UnknownModelError(model_name) from None

        family = get_model_family(card)

        try:
            handler = self._model_handlers.get(family)
        except LookupError:
            raise UnknownModelFamilyError(family, model_name) from None

        if not issubclass(handler.kls, self._kls):
            raise TypeError(
                f"The model must be of type `{self._kls}`, but is of type `{handler.kls}` instead."
            )

        model_config = handler.load_config(card)

        # Load the model.
        log.info("Loading the '{}' model on data parallel rank 0 (per shard).", model_name)  # fmt: skip

        if mp:
            dtype = torch.float32

        if gangs.dp.rank == 0:
            model = handler.load(card, gangs, dtype, model_config)
        else:
            model = handler.create(model_config, gangs, dtype, handler.supports_meta)

        gangs.root.barrier()

        model.eval()

        log.info("The model is loaded on data parallel rank 0.")

        return cast(ModelT, model)


def broadcast_model(name: str, model: Module, gangs: Gangs) -> None:
    if gangs.dp.size == 1:
        return

    log.info("Broadcasting the '{}' model to all processes.", name)

    broadcast_module(model, gangs.dp)

    log.info("The model is broadcasted.")


def wrap_data_parallel(
    context: RuntimeContext,
    config: TrainRecipeConfig,
    model: Module,
    gangs: Gangs,
    checkpoint_manager: CheckpointManager,
    static_graph: bool = True,
) -> Module:
    if gangs.dp.size == 1:
        to_device(model, gangs.root.device)

        return model

    if config.trainer.data_parallelism == "ddp":
        return wrap_ddp(model, gangs, static_graph)

    if config.trainer.data_parallelism == "fsdp":
        return wrap_fsdp(config, model, gangs, checkpoint_manager, static_graph)

    raise ValueError("`data_parallelism` must be 'ddp' or 'fsdp'.")


def wrap_ddp(model: Module, gangs: Gangs, static_graph: bool) -> Module:
    log.info("Wrapping the model with DDP and broadcasting it to all processes.")

    model = to_ddp(
        model,
        gangs.dp,
        find_unused_parameters=not static_graph,
        static_graph=static_graph,
    )

    log.info("The model is wrapped with DDP and broadcasted.")

    return model


def wrap_fsdp(
    config: TrainRecipeConfig,
    model: Module,
    gangs: Gangs,
    checkpoint_manager: CheckpointManager,
    static_graph: bool,
) -> Module:
    if not static_graph:
        raise NotSupportedError("FSDP does not support non-static graphs.")

    has_checkpoint = checkpoint_manager.has_checkpoint()

    if has_checkpoint:
        log.info("Wrapping the model with FSDP.")
    else:
        log.info("Wrapping the model with FSDP and broadcasting it to all processes.")  # fmt: skip

    trainer = config.trainer

    if trainer.mixed_precision == "static":
        mixed_precision_dtype = trainer.dtype
    else:
        mixed_precision_dtype = None

    wrap_policy, ignored_modules = get_fsdp_wrap_policy(
        model, wrap_granularity=trainer.fsdp.granularity
    )

    model = to_fsdp(
        model,
        gangs.dp,
        wrap_policy,
        ignored_modules=ignored_modules,
        broadcast_state=not has_checkpoint,
        reshard_after_forward=trainer.fsdp.reshard_after_forward,
        mixed_precision_dtype=mixed_precision_dtype,
        fp32_reduce=trainer.fsdp.fp32_reduce,
    )

    if has_checkpoint:
        log.info("The model is wrapped with FSDP.")
    else:
        log.info("The model is wrapped with FSDP and broadcasted.")

    return model


def prepare_model(
    context: RuntimeContext, config: TrainRecipeConfig, model: Module, gangs: Gangs
) -> Module:
    if config.trainer.activation_checkpointing:
        use_layerwise_activation_checkpointing(model)

    return model


def create_seq_generator(
    context: RuntimeContext,
    config_section: SequenceGeneratorSection,
    model: DecoderModel,
) -> SequenceGenerator:
    handlers = context.get_registry(SequenceGeneratorHandler)

    try:
        handler = handlers.get(config_section.name)
    except LookupError:
        raise UnknownSequenceGeneratorError(config_section.name) from None

    return handler.create(model, config_section.config)


def create_seq2seq_generator(
    context: RuntimeContext,
    config_section: Seq2SeqGeneratorSection,
    model: EncoderDecoderModel,
) -> Seq2SeqGenerator:
    handlers = context.get_registry(Seq2SeqGeneratorHandler)

    try:
        handler = handlers.get(config_section.name)
    except LookupError:
        raise UnknownSeq2SeqGeneratorError(config_section.name) from None

    return handler.create(model, config_section.config)


def compile_model(
    context: RuntimeContext, config_section: ModelSection, model: ModelT
) -> ModelT:
    # TODO: implement!
    return model


def compile_eval_model(
    context: RuntimeContext, model_name: str, model: ModelT
) -> ModelT:
    # TODO: implement!
    return model


def check_model_type(model: Module, kls: type[Module]) -> None:
    """Check if a potentially DDP or FSDP wrapped `model` is of type `kls`."""
    if isinstance(model, (DDP, FSDP)):
        model = model.module

    if not isinstance(model, kls):
        raise TypeError(
            f"`model` must be of type `{kls}`, but is of type `{type(model)}` instead."
        )
