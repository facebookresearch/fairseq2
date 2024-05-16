# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final

from fairseq2.data.text import (
    default_raw_sentencepiece_tokenizer_loader,
    load_text_tokenizer,
)

LIBRISPEECH_ASR_FAMILY: Final = "librispeech_asr"


load_librispeech_asr_tokenizer = default_raw_sentencepiece_tokenizer_loader


def _register_librispeech_asr() -> None:
    load_text_tokenizer.register(LIBRISPEECH_ASR_FAMILY, load_librispeech_asr_tokenizer)
