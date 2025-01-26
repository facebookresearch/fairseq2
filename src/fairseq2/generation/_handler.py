# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod

from fairseq2.generation._generator import Seq2SeqGenerator, SequenceGenerator
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.encoder_decoder import EncoderDecoderModel


class SequenceGeneratorHandler(ABC):
    @abstractmethod
    def create(self, model: DecoderModel, config: object) -> SequenceGenerator:
        ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]:
        ...


class UnknownSequenceGeneratorError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known sequence generator.")

        self.name = name


class Seq2SeqGeneratorHandler(ABC):
    @abstractmethod
    def create(self, model: EncoderDecoderModel, config: object) -> Seq2SeqGenerator:
        ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]:
        ...


class UnknownSeq2SeqGeneratorError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known sequence-to-sequence generator.")

        self.name = name
