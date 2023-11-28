# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final

from fairseq2.data.text import BasicSentencePieceTokenizer


@final
class MistralTokenizer(BasicSentencePieceTokenizer):
    """Represents the tokenizer used by Mistral models."""
