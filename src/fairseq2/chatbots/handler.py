# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod

from fairseq2.chatbots.chatbot import Chatbot
from fairseq2.data.text import TextTokenizer
from fairseq2.generation.generator import SequenceGenerator


class ChatbotHandler(ABC):
    @abstractmethod
    def create(self, generator: SequenceGenerator, tokenizer: TextTokenizer) -> Chatbot:
        ...


class ChatbotNotFoundError(LookupError):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known chatbot.")

        self.name = name
