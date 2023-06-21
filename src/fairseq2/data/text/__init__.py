# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.data.text.dict_model import DictDecoder as DictDecoder
from fairseq2.data.text.dict_model import DictEncoder as DictEncoder
from fairseq2.data.text.dict_model import DictModel as DictModel
from fairseq2.data.text.dict_tokenizer import DictTokenizer as DictTokenizer
from fairseq2.data.text.multilingual_tokenizer import (
    MultilingualTokenizer as MultilingualTokenizer,
)
from fairseq2.data.text.reader import LineEnding as LineEnding
from fairseq2.data.text.reader import read_text as read_text
from fairseq2.data.text.sentencepiece import (
    SentencePieceDecoder as SentencePieceDecoder,
)
from fairseq2.data.text.sentencepiece import (
    SentencePieceEncoder as SentencePieceEncoder,
)
from fairseq2.data.text.sentencepiece import SentencePieceModel as SentencePieceModel
from fairseq2.data.text.sentencepiece import (
    vocab_from_sentencepiece as vocab_from_sentencepiece,
)
from fairseq2.data.text.tokenizer import TokenDecoder as TokenDecoder
from fairseq2.data.text.tokenizer import TokenEncoder as TokenEncoder
from fairseq2.data.text.tokenizer import Tokenizer as Tokenizer
from fairseq2.data.text.tokenizer import VocabularyInfo as VocabularyInfo
