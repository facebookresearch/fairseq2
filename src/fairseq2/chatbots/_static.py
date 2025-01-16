# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.chatbots._chatbot import Chatbot
from fairseq2.chatbots._handler import ChatbotHandler, ChatbotNotFoundError
from fairseq2.context import get_runtime_context
from fairseq2.data.text.tokenizers import TextTokenizer
from fairseq2.generation import SequenceGenerator


def create_chatbot(
    name: str, generator: SequenceGenerator, tokenizer: TextTokenizer
) -> Chatbot:
    context = get_runtime_context()

    registry = context.get_registry(ChatbotHandler)

    try:
        handler = registry.get(name)
    except LookupError:
        raise ChatbotNotFoundError(name) from None

    return handler.create(generator, tokenizer)
