# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import Any, Dict, final, TextIO

import torch
import torch.nn.functional as F

from fairseq2.data.text.tokenizers import TextTokenDecoder, TextTokenizer
from fairseq2.gang import Gang
from fairseq2.metrics import Max, Mean, MetricBag, Sum
from fairseq2.models.seq2seq import Seq2SeqBatch, SonarSpeechSeq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.recipes import BaseMetricBag, Model, UnitError
from fairseq2.recipes.wav2vec2.asr._train import Wav2Vec2AsrTrainConfig
from torch import Tensor
from torch.nn import CosineEmbeddingLoss, MSELoss
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from typing_extensions import override


@final
class SonarSpeechCriterion:
    _model: Model

    def __init__(self, model: Model | None = None) -> None:
        self._model = model
        self.mse_loss_fn = MSELoss(reduction="none")
        # self.cosine_loss_fn = CosineEmbeddingLoss(reduction="none")

    def __call__(
        self, batch: SonarSpeechSeq2SeqBatch, metric_bag: SonarSpeechMetricBag
    ) -> tuple[Tensor, int]:
        output = self._forward(batch)

        mse_loss = self.mse_loss_fn(output.text_embeddings, output.speech_embeddings)
        mse_loss = mse_loss.sum(dim=1)

        cosine_sim = F.cosine_similarity(
            output.speech_embeddings, output.text_embeddings
        )

        if not self._model.module.training:
            print(f"mse loss: {mse_loss.mean()}")
            print(f"cosine sim: {cosine_sim.mean()}")
            print(
                pairwise_cosine_similarity(
                    output.speech_embeddings, output.text_embeddings
                ).argmax(dim=-1)
            )

        loss = mse_loss.sum()

        metric_bag.update_mse_loss(batch, loss)
        metric_bag.update_cosine_sim(batch, cosine_sim.sum())
        metric_bag.update_batch_metrics(batch)

        return loss, batch.batch_size

    def _forward(
        self, batch: SequenceBatch | SonarSpeechSeq2SeqBatch
    ) -> SonarSpeechModelOutput:
        return self._model.module(batch)  # type: ignore[no-any-return]

    @property
    def model(self) -> Model:
        return self._model


class SonarSpeechMetricBag(BaseMetricBag):
    mse_loss: Mean

    def __init__(self, gang: Gang, train: bool = True) -> None:
        super().__init__(gang, train=train)

        self.device = gang.device

        self.register_metric("mse_loss", Mean(device=self.device), persistent=False)
        self.register_metric(
            "cosine_similarity", Mean(device=self.device), persistent=False
        )

    @torch.inference_mode()
    def update_mse_loss(self, batch: SonarSpeechSeq2SeqBatch, loss: Tensor) -> None:
        n = batch.batch_size

        self.mse_loss.update(loss.detach() / n, weight=n)

    @torch.inference_mode()
    def update_cosine_sim(self, batch: SonarSpeechSeq2SeqBatch, sim: Tensor) -> None:
        n = batch.batch_size

        self.cosine_similarity.update(sim.detach() / n, weight=n)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: SonarSpeechSeq2SeqBatch) -> None:
        num_examples = batch.batch_size

        num_elements = batch.num_source_elements()

        self.num_examples.update(num_examples)
        self.num_elements.update(num_elements)

        if self._train:
            assert self.total_num_examples is not None
            assert self.total_num_elements is not None

            self.total_num_examples.update(num_examples)
            self.total_num_elements.update(num_elements)
