# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, TypeAlias, final

from typing_extensions import override

from fairseq2.chatbots.chatbot import Chatbot
from fairseq2.data.text import TextTokenizer
from fairseq2.generation.generator import SequenceGenerator
from fairseq2.utils.registry import Registry


class ChatbotHandler(ABC):
    @abstractmethod
    def make(self, generator: SequenceGenerator, tokenizer: TextTokenizer) -> Chatbot:
        ...


ChatbotRegistry: TypeAlias = Registry[ChatbotHandler]


class ChatbotFactory(Protocol):
    def __call__(
        self, generator: SequenceGenerator, tokenizer: TextTokenizer
    ) -> Chatbot:
        ...


@final
class StandardChatbotHandler(ChatbotHandler):
    def __init__(self, *, factory: ChatbotFactory) -> None:
        self._factory = factory

    @override
    def make(self, generator: SequenceGenerator, tokenizer: TextTokenizer) -> Chatbot:
        return self._factory(generator, tokenizer)
