# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.models.chatbot import ChatbotFactory as ChatbotFactory
from fairseq2.models.chatbot import DelegatingChatbotFactory as DelegatingChatbotFactory
from fairseq2.models.chatbot import create_chatbot as create_chatbot
from fairseq2.models.config_loader import ModelConfigLoader as ModelConfigLoader
from fairseq2.models.config_loader import (
    StandardModelConfigLoader as StandardModelConfigLoader,
)
from fairseq2.models.loader import CheckpointConverter as CheckpointConverter
from fairseq2.models.loader import DelegatingModelLoader as DelegatingModelLoader
from fairseq2.models.loader import DenseModelFactory as DenseModelFactory
from fairseq2.models.loader import DenseModelLoader as DenseModelLoader
from fairseq2.models.loader import ModelLoader as ModelLoader
from fairseq2.models.loader import load_model as load_model
from fairseq2.models.model import Model as Model

# isort: split

from fairseq2.models.llama import _register_llama
from fairseq2.models.mistral import _register_mistral
from fairseq2.models.nllb import _register_nllb
from fairseq2.models.s2t_transformer import _register_s2t_transformer
from fairseq2.models.w2vbert import _register_w2vbert
from fairseq2.models.wav2vec2 import _register_wav2vec2
from fairseq2.models.wav2vec2.asr import _register_wav2vec2_asr


def _register_models() -> None:
    _register_llama()
    _register_mistral()
    _register_nllb()
    _register_s2t_transformer()
    _register_w2vbert()
    _register_wav2vec2()
    _register_wav2vec2_asr()
