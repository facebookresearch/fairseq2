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

from fairseq2.data import CString
from fairseq2.data.text import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
)
from fairseq2.typing import DataType
from tests.common import assert_equal, device

TEST_SPM_PATH: Final = Path(__file__).parent.joinpath("test.spm")


class TestSentencePieceModel:
    sentence: ClassVar[str]
    token_indices: ClassVar[List[int]]

    @classmethod
    def setup_class(cls) -> None:
        cls.sentence = "What's up? Hope you are doing well today."

        # fmt: off
        cls.token_indices = [
            169, 87, 5, 227, 11, 424, 294, 40, 106, 120, 26, 597, 19, 303, 4
        ]
        # fmt: on

    def test_init_works(self) -> None:
        model = SentencePieceModel(TEST_SPM_PATH)

        assert model.pad_idx is None

    def test_init_works_when_pad_symbol_is_specified(self) -> None:
        model = SentencePieceModel(TEST_SPM_PATH, control_symbols=["<pad>"])

        assert model.pad_idx == 1000
        assert model.unk_idx == 0

    def test_init_works_when_pad_at_0_symbol_is_specified(self) -> None:
        # Note that this is an undocumented feature and is not part of our
        # public API.
        model = self.build_model()

        assert model.pad_idx == 0
        assert model.unk_idx == 1

    def test_init_raises_error_when_model_file_is_not_found(self) -> None:
        with pytest.raises(RuntimeError, match="No such file or"):
            SentencePieceModel("<non-existent-file>")

    def test_index_to_token_raises_error_when_input_is_out_of_range(self) -> None:
        model = self.build_model()

        with pytest.raises(
            ValueError,
            match=r"^`idx` must be less than vocabulary size \(1001\), but is 1005 instead\.$",
        ):
            model.index_to_token(1005)

    def test_encode_decode_work(self) -> None:
        model = self.build_model()

        encoder = SentencePieceEncoder(model, device=device)
        decoder = SentencePieceDecoder(model)

        indices = encoder(self.sentence)

        # Assert encoder.
        assert_equal(indices, self.token_indices)

        sentence = decoder(indices)

        # Assert decoder
        assert isinstance(sentence, CString)

        assert sentence == self.sentence

    def test_encode_decode_work_when_reverse_is_true(self) -> None:
        model = self.build_model()

        encoder = SentencePieceEncoder(model, device=device, reverse=True)
        decoder = SentencePieceDecoder(model, reverse=True)

        indices = encoder(self.sentence)

        # Assert encoder.
        assert_equal(indices, self.token_indices[::-1])

        sentence = decoder(indices)

        # Assert decoder.
        assert isinstance(sentence, CString)

        assert sentence == self.sentence

    def test_decode_works_when_control_symbols_are_specified(self) -> None:
        model = self.build_model(control_symbols=["<foo>"])

        encoder = SentencePieceEncoder(model, device=device)
        decoder = SentencePieceDecoder(model)

        indices = encoder(self.sentence)

        # Assert encoder.
        assert_equal(indices, self.token_indices)

        # We inject a dummy <foo> token to the returned tokens.
        foo_idx = model.token_to_index("<foo>")

        foo = torch.full((1,), foo_idx, device=device, dtype=indices.dtype)

        indices = torch.cat([indices[:2], foo, indices[2:]])

        # We expect the decoder to ignore the <foo> tokens.
        sentence = decoder(indices)

        # Assert decoder.
        assert isinstance(sentence, CString)

        assert sentence == self.sentence

    def test_encode_works_when_prefix_and_suffix_tokens_are_specified(self) -> None:
        model = self.build_model(control_symbols=["<foo1>", "<foo2>", "<foo3>"])

        encoder = SentencePieceEncoder(
            model,
            device=device,
            prefix_tokens=["<foo1>", "<s>"],
            suffix_tokens=["<foo2>", "</s>", "<foo3>"],
        )
        decoder = SentencePieceDecoder(model)

        indices = encoder(self.sentence)

        # Assert encoder.
        foo1_idx = model.token_to_index("<foo1>")
        foo2_idx = model.token_to_index("<foo2>")
        foo3_idx = model.token_to_index("<foo3>")

        bos_idx = model.bos_idx
        eos_idx = model.eos_idx

        e = [foo1_idx, bos_idx] + self.token_indices + [foo2_idx, eos_idx, foo3_idx]

        assert_equal(indices, e)

        # We expect the decoder to ignore the prefix and suffix tokens.
        sentence = decoder(indices)

        # Assert decoder.
        assert sentence == self.sentence

    @pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
    def test_decode_raises_error_when_data_type_is_not_supported(
        self, dtype: DataType
    ) -> None:
        model = self.build_model()

        decoder = SentencePieceDecoder(model)

        indices = torch.zeros((10,), device=device, dtype=dtype)

        with pytest.raises(
            ValueError,
            match=r"^`sp_decoder` supports only `torch.int16`, `torch.int32`, and `torch.int64` data types\.$",
        ):
            decoder(indices)

    @pytest.mark.parametrize("shape", [(), (4, 4, 4)])
    def test_decode_raises_error_when_input_has_more_than_1_dimension(
        self, shape: Sequence[int]
    ) -> None:
        model = self.build_model()

        decoder = SentencePieceDecoder(model)

        indices = torch.zeros(shape, device=device)

        with pytest.raises(
            ValueError,
            match=rf"^The input tensor must be one dimensional, but has {len(shape)} dimension\(s\) instead\.$",
        ):
            decoder(indices)

    def test_pickle_works(self) -> None:
        model = self.build_model(control_symbols=["<foo1>", "<foo2>", "<foo3>"])

        foo1_idx = model.token_to_index("<foo1>")
        foo2_idx = model.token_to_index("<foo2>")
        foo3_idx = model.token_to_index("<foo3>")

        dump = pickle.dumps(model)

        del model

        # We expect that the entire state of the model including our control
        # symbols are pickled.
        model = pickle.loads(dump)

        assert foo1_idx == model.token_to_index("<foo1>")
        assert foo2_idx == model.token_to_index("<foo2>")
        assert foo3_idx == model.token_to_index("<foo3>")

    @staticmethod
    def build_model(
        control_symbols: Optional[Sequence[str]] = None,
    ) -> SentencePieceModel:
        symbols = ["<pad>@0"]

        if control_symbols is not None:
            symbols += control_symbols

        return SentencePieceModel(TEST_SPM_PATH, control_symbols=symbols)
