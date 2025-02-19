# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.chatbots.llama import register_llama_chatbot
from fairseq2.chatbots.mistral import register_mistral_chatbot
from fairseq2.context import RuntimeContext


def register_chatbots(context: RuntimeContext) -> None:
    register_llama_chatbot(context)
    register_mistral_chatbot(context)
