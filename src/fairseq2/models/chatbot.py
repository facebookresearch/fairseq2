# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Protocol

from fairseq2.data.text import TextTokenizer
from fairseq2.generation import Chatbot, SequenceGenerator


class ChatbotFactory(Protocol):
    """Constructs instances of :class:`Chatbot`."""

    def __call__(
        self, generator: SequenceGenerator, tokenizer: TextTokenizer
    ) -> Chatbot:
        """
        :param generator:
            The sequence generator.
        :param tokenizer:
            The text tokenizer.
        """


class DelegatingChatbotFactory(ChatbotFactory):
    """Constructs instances of :class:`Chatbot` using registered factories."""

    _factories: Dict[str, ChatbotFactory]

    def __init__(self) -> None:
        self._factories = {}

    def __call__(
        self, generator: SequenceGenerator, tokenizer: TextTokenizer
    ) -> Chatbot:
        family = generator.model.family
        if family is None:
            raise ValueError("`generator.model.family` must not be `None`.")

        try:
            factory = self._factories[family]
        except KeyError:
            raise ValueError(
                f"`generator.model.family` must be a supported model family, but '{family}' has no registered chatbot."
            )

        return factory(generator, tokenizer)

    def register(self, family: str, factory: ChatbotFactory) -> None:
        """Register a chatbot factory to use with this factory.

        :param family:
            The model family supported by ``factory``.
        :param factory:
            The chatbot factory.
        """
        if family in self._factories:
            raise ValueError(
                f"`family` must be a unique model family name, but '{family}' has already a registered chatbot."
            )

        self._factories[family] = factory


create_chatbot = DelegatingChatbotFactory()
