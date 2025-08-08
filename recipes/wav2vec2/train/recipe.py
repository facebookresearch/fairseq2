 Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from typing_extensions import override

from .criterion import Wav2Vec2Criterion
from .dataset import (
    WAV2VEC2_DATASET,
    Wav2Vec2Dataset,
    Wav2Vec2DatasetConfig,
    open_wav2vec2_train_dataset,
)
from .default_config import Wav2Vec2TrainConfig


@final
class Wav2Vec2TrainRecipe(TrainRecipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            WAV2VEC2_DATASET,
            Wav2Vec2TrainDataset,
            Wav2Vec2TrainDatasetConfig,
            opener=open_wav2vec2_train_dataset,
        )

    @override
    def create_trainer(self, context: RecipeContext) -> Trainer:
        config = context.config_as(Wav2Vec2TrainConfig)

        unit = Wav2Vec2TrainUnit(context.model)

        dataset = context.dataset_as(Wav2Vec2TrainDataset)

        # data_reader = dataset.create_reader(
        #     context.tokenizer,
        #     context.gangs,
        #     max_seq_len=config.dataset.max_seq_len,
        #     max_num_tokens=config.dataset.max_num_tokens,
        #     num_accumulate=config.trainer.grad_accumulation.num_batches,
        #     prefetch=config.dataset.prefetch,
        #     sync_ranks=config.dataset.sync_ranks,
        # )

        return context.create_trainer(unit, data_reader)

    @property
    @override
    def config_kls(self) -> type[object]:
        return Wav2Vec2TrainConfig


@final
class Wav2Vec2TrainUnit(TrainUnit[SequenceBatch]):
    _model: Model
    _criterion: Wav2Vec2Criterion

    def __init__(self, model: Model, criterion: Wav2Vec2Criterion) -> None:
        self._model = model
        self._criterion = criterion

    @override
    def __call__(
        self, batch: SequenceBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        return self._criterion(batch, metric_bag)

    @property
    @override
    def model(self) -> Model:
        return self._model
