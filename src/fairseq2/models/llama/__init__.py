# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.llama.chatbot import LLaMA3Chatbot as LLaMA3Chatbot
from fairseq2.models.llama.chatbot import LLaMAChatbot as LLaMAChatbot
from fairseq2.models.llama.chatbot import create_llama_chatbot as create_llama_chatbot
from fairseq2.models.llama.factory import LLAMA_FAMILY as LLAMA_FAMILY
from fairseq2.models.llama.factory import LLaMABuilder as LLaMABuilder
from fairseq2.models.llama.factory import LLaMAConfig as LLaMAConfig
from fairseq2.models.llama.factory import create_llama_model as create_llama_model
from fairseq2.models.llama.factory import get_llama_lora_config as get_llama_lora_config
from fairseq2.models.llama.factory import (
    get_llama_lora_fa_config as get_llama_lora_fa_config,
)
from fairseq2.models.llama.factory import llama_arch as llama_arch
from fairseq2.models.llama.factory import llama_archs as llama_archs
from fairseq2.models.llama.loader import load_llama_config as load_llama_config
from fairseq2.models.llama.loader import load_llama_model as load_llama_model
from fairseq2.models.llama.loader import load_llama_tokenizer as load_llama_tokenizer
from fairseq2.models.llama.tokenizer import LLaMA3Tokenizer as LLaMA3Tokenizer

# isort: split

from fairseq2.dependency import DependencyContainer
from fairseq2.models.llama.archs import register_archs


def register_llama(container: DependencyContainer) -> None:
    register_archs()
