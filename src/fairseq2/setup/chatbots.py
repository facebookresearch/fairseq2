# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.chatbots import ChatbotHandler
from fairseq2.chatbots.llama import LLaMAChatbotHandler
from fairseq2.chatbots.mistral import MistralChatbotHandler
from fairseq2.context import RuntimeContext
from fairseq2.models.llama import LLAMA_MODEL_FAMILY
from fairseq2.models.mistral import MISTRAL_MODEL_FAMILY


def _register_chatbots(context: RuntimeContext) -> None:
    registry = context.get_registry(ChatbotHandler)

    registry.register(LLAMA_MODEL_FAMILY, LLaMAChatbotHandler())
    registry.register(MISTRAL_MODEL_FAMILY, MistralChatbotHandler())
