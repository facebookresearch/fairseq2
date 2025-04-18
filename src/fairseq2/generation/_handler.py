# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod

from fairseq2.data import VocabularyInfo
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.encoder_decoder import EncoderDecoderModel

# isort: split

from fairseq2.generation._generator import Seq2SeqGenerator, SequenceGenerator


class SequenceGeneratorHandler(ABC):
    @abstractmethod
    def create(
        self, model: DecoderModel, vocab_info: VocabularyInfo, config: object
    ) -> SequenceGenerator: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class Seq2SeqGeneratorHandler(ABC):
    @abstractmethod
    def create(
        self,
        model: EncoderDecoderModel,
        target_vocab_info: VocabularyInfo,
        config: object,
    ) -> Seq2SeqGenerator: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...
