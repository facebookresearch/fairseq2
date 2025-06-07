# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.chatbots.chatbot import (
    Chatbot,
    ChatDialogEncoder,
    StandardChatbot,
)
from fairseq2.chatbots.llama import LLaMA1DialogEncoder, LLaMA3DialogEncoder
from fairseq2.chatbots.mistral import MistralDialogEncoder
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.data.tokenizers.sentencepiece import BasicSentencePieceTokenizer
from fairseq2.dependency import DependencyNotFoundError, DependencyResolver
from fairseq2.error import ContractError
from fairseq2.generation import SequenceGenerator
from fairseq2.models.clm import CausalLM
from fairseq2.recipe.model import Model


def create_chatbot(resolver: DependencyResolver) -> Chatbot:
    model = resolver.resolve(Model)

    try:
        return resolver.resolve(Chatbot, key=model.handler.family)
    except DependencyNotFoundError:
        raise UnknownChatbotError(model.name) from None


class UnknownChatbotError(Exception):
    model_name: str

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"The '{model_name}' model does not have a chatbot implementation."
        )

        self.model_name = model_name


def create_llama_chatbot(resolver: DependencyResolver) -> Chatbot:
    model = resolver.resolve(Model)

    base_module = model.base_module
    if not isinstance(base_module, CausalLM):
        raise ContractError(
            f"`model.base_module` is expected to be of type `{CausalLM}`, but is of type `{type(base_module)}` instead."
        )

    tokenizer = resolver.resolve(Tokenizer)

    seq_generator = resolver.resolve(SequenceGenerator)

    dialog_encoder: ChatDialogEncoder

    if isinstance(tokenizer, BasicSentencePieceTokenizer):
        dialog_encoder = LLaMA1DialogEncoder(base_module, tokenizer)
    else:
        dialog_encoder = LLaMA3DialogEncoder(base_module, tokenizer)

    text_decoder = tokenizer.create_decoder(skip_special_tokens=True)

    return StandardChatbot(
        seq_generator, dialog_encoder, text_decoder, supports_system_prompt=True
    )


def create_mistral_chatbot(resolver: DependencyResolver) -> Chatbot:
    model = resolver.resolve(Model)

    base_module = model.base_module
    if not isinstance(base_module, CausalLM):
        raise ContractError(
            f"`model.base_module` is expected to be of type `{CausalLM}`, but is of type `{type(base_module)}` instead."
        )

    tokenizer = resolver.resolve(Tokenizer)

    seq_generator = resolver.resolve(SequenceGenerator)

    dialog_encoder = MistralDialogEncoder(base_module, tokenizer)

    text_decoder = tokenizer.create_decoder(skip_special_tokens=True)

    return StandardChatbot(
        seq_generator, dialog_encoder, text_decoder, supports_system_prompt=False
    )
