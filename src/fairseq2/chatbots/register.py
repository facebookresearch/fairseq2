# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.chatbots.llama import make_llama_chatbot
from fairseq2.chatbots.mistral import make_mistral_chatbot
from fairseq2.chatbots.registry import ChatbotRegistry, StandardChatbotHandler
from fairseq2.extensions import run_extensions
from fairseq2.models.llama import LLAMA_FAMILY
from fairseq2.models.mistral import MISTRAL_FAMILY


def register_chatbots(registry: ChatbotRegistry) -> None:
    # LLaMA
    handler = StandardChatbotHandler(factory=make_llama_chatbot)

    registry.register(LLAMA_FAMILY, handler)

    # Mistral
    handler = StandardChatbotHandler(factory=make_mistral_chatbot)

    registry.register(MISTRAL_FAMILY, handler)

    # Extensions
    run_extensions("register_fairseq2_chatbots", registry)
