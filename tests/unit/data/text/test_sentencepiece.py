# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from pathlib import Path
from typing import ClassVar, Final, List, Optional, Sequence

import pytest
import torch

from fairseq2.data.text import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
)
from fairseq2.typing import DataType
from tests.common import assert_equal, device

TEST_SPM_PATH: Final = Path(__file__).parent.joinpath("test.spm")


class TestSentencePieceModel:
    sentences: ClassVar[List[str]]
    token_indices: ClassVar[List[List[int]]]

    @classmethod
    def setup_class(cls) -> None:
        cls.sentences = [
            "Hello world! How are you?",
            "What's up? Hope you are doing well today.",
        ]

        # fmt: off
        cls.token_indices = [
            [132, 30, 131, 114, 52, 418, 68, 166, 106, 40, 11],
            [169, 87, 5, 227, 11, 424, 294, 40, 106, 120, 26, 597, 19, 303, 4]
        ]
        # fmt: on

    def test_init_raises_error_if_model_file_is_not_found(self) -> None:
        with pytest.raises(RuntimeError, match="No such file or"):
            SentencePieceModel("<non-existent-file>")

    def test_init_constructs_model_without_pad_symbol(self) -> None:
        spm = SentencePieceModel(TEST_SPM_PATH)

        assert spm.pad_idx is None

    def test_init_adds_pad(self) -> None:
        spm = SentencePieceModel(TEST_SPM_PATH, control_symbols=["<pad>"])

        assert spm.pad_idx == 1000
        assert spm.unk_idx == 0

    def test_init_adds_pad_at_index_0(self) -> None:
        # Note that this is an undocumented feature and is not part of our
        # public API.
        spm = self.build_model()

        assert spm.pad_idx == 0
        assert spm.unk_idx == 1

    def test_index_to_token_raises_error_if_out_of_range(self) -> None:
        spm = self.build_model()

        with pytest.raises(ValueError):
            spm.index_to_token(4867847)

    def test_encodes_decodes_sentence_correctly(self) -> None:
        spm = self.build_model()

        encoder = SentencePieceEncoder(spm, device=device)
        decoder = SentencePieceDecoder(spm)

        indices = encoder(self.sentences[0])

        # Assert encoder.
        assert_equal(indices, self.token_indices[0])

        sentences = decoder(indices)

        # Assert decoder
        assert isinstance(sentences, list)

        assert len(sentences) == 1

        assert sentences[0] == self.sentences[0]

    def test_decode_ignores_control_token(self) -> None:
        spm = self.build_model(control_symbols=["<foo>"])

        encoder = SentencePieceEncoder(spm, device=device)
        decoder = SentencePieceDecoder(spm)

        indices = encoder(self.sentences[0])

        # Assert encoder.
        assert_equal(indices, self.token_indices[0])

        # We inject a dummy <foo> token to the returned tokens.
        foo_idx = spm.token_to_index("<foo>")

        foo = torch.full((1,), foo_idx, device=device, dtype=indices.dtype)

        indices = torch.cat([indices[:2], foo, indices[2:]])

        # We expect the decoder to ignore the <foo> tokens.
        sentences = decoder(indices)

        # Assert decoder.
        assert isinstance(sentences, list)

        assert len(sentences) == 1

        assert sentences[0] == self.sentences[0]

    def test_encode_adds_prefix_and_suffix_tokens_correctly(self) -> None:
        spm = self.build_model(control_symbols=["<foo1>", "<foo2>", "<foo3>"])

        encoder = SentencePieceEncoder(
            spm,
            device=device,
            prefix_tokens=["<foo1>", "<s>"],
            suffix_tokens=["<foo2>", "</s>", "<foo3>"],
        )
        decoder = SentencePieceDecoder(spm)

        indices = encoder(self.sentences[0])

        # Assert encoder.
        foo1_idx = spm.token_to_index("<foo1>")
        foo2_idx = spm.token_to_index("<foo2>")
        foo3_idx = spm.token_to_index("<foo3>")

        e = (
            [foo1_idx, spm.bos_idx]
            + self.token_indices[0]
            + [foo2_idx, spm.eos_idx, foo3_idx]
        )

        assert_equal(indices, e)

        # We expect the decoder to ignore the prefix and suffix tokens.
        sentences = decoder(indices)

        # Assert decoder.
        assert sentences[0] == self.sentences[0]

    def test_encodes_decodes_in_reverse_correctly(self) -> None:
        spm = self.build_model()

        encoder = SentencePieceEncoder(spm, device=device, reverse=True)
        decoder = SentencePieceDecoder(spm, reverse=True)

        indices = encoder(self.sentences[0])

        # Assert encoder.
        assert_equal(indices, self.token_indices[0][::-1])

        sentences = decoder(indices)

        # Assert decoder.
        assert isinstance(sentences, list)

        assert len(sentences) == 1

        assert sentences[0] == self.sentences[0]

    @pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
    def test_decodes_batch_correctly(self, dtype: DataType) -> None:
        spm = self.build_model(control_symbols=["<foo>"])

        decoder = SentencePieceDecoder(spm)

        i1 = torch.tensor(self.token_indices[0], device=device, dtype=dtype)
        i2 = torch.tensor(self.token_indices[1], device=device, dtype=dtype)

        batch = torch.nn.utils.rnn.pad_sequence([i1, i2], batch_first=True)

        sentences = decoder(batch)

        assert isinstance(sentences, list)

        assert len(sentences) == 2

        assert sentences == self.sentences

    @pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
    def test_decode_raises_error_if_data_type_is_not_supported(
        self, dtype: DataType
    ) -> None:
        spm = self.build_model(control_symbols=["<foo>"])

        decoder = SentencePieceDecoder(spm)

        indices = torch.zeros((10,), device=device, dtype=dtype)

        with pytest.raises(ValueError):
            decoder(indices)

    def test_pickles_correctly(self) -> None:
        spm = self.build_model(control_symbols=["<foo1>", "<foo2>", "<foo3>"])

        foo1_idx = spm.token_to_index("<foo1>")
        foo2_idx = spm.token_to_index("<foo2>")
        foo3_idx = spm.token_to_index("<foo3>")

        dmp = pickle.dumps(spm)

        # We expect that the entire state of the model including our control
        # symbols are pickled.
        spm = pickle.loads(dmp)

        assert foo1_idx == spm.token_to_index("<foo1>")
        assert foo2_idx == spm.token_to_index("<foo2>")
        assert foo3_idx == spm.token_to_index("<foo3>")

    @staticmethod
    def build_model(
        control_symbols: Optional[Sequence[str]] = None,
    ) -> SentencePieceModel:
        ct = ["<pad>@0"]

        if control_symbols is not None:
            ct += control_symbols

        return SentencePieceModel(TEST_SPM_PATH, control_symbols=ct)
