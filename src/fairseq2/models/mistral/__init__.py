# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.models.mistral.chatbot import MistralChatbot as MistralChatbot
from fairseq2.models.mistral.factory import MISTRAL_FAMILY as MISTRAL_FAMILY
from fairseq2.models.mistral.factory import MistralBuilder as MistralBuilder
from fairseq2.models.mistral.factory import MistralConfig as MistralConfig
from fairseq2.models.mistral.factory import create_mistral_model as create_mistral_model
from fairseq2.models.mistral.factory import mistral_arch as mistral_arch
from fairseq2.models.mistral.factory import mistral_archs as mistral_archs
from fairseq2.models.mistral.setup import load_mistral_config as load_mistral_config
from fairseq2.models.mistral.setup import load_mistral_model as load_mistral_model
from fairseq2.models.mistral.setup import (
    load_mistral_tokenizer as load_mistral_tokenizer,
)

# isort: split

from fairseq2.data.text import load_text_tokenizer
from fairseq2.models.chatbot import create_chatbot
from fairseq2.models.loader import load_model


def _register_mistral() -> None:
    load_model.register(MISTRAL_FAMILY, load_mistral_model)

    load_text_tokenizer.register(MISTRAL_FAMILY, load_mistral_tokenizer)

    create_chatbot.register(MISTRAL_FAMILY, MistralChatbot)
