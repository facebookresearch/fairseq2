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
import torch.nn.functional as F
from torch import Tensor

from fairseq2.data.text import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
)
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

    def test_init_adds_pad(self) -> None:
        spm = SentencePieceModel(TEST_SPM_PATH, control_tokens=["<pad>"])

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

    @pytest.mark.parametrize("left_pad", [False, True])
    def test_encodes_decodes_batch_with_single_sentence_correctly(
        self, left_pad: bool
    ) -> None:
        spm = self.build_model()

        encoder = SentencePieceEncoder(spm, device=device, left_pad=left_pad)
        decoder = SentencePieceDecoder(spm)

        expected_indices = self.build_batch(self.token_indices[:1], left_pad=left_pad)

        indices = encoder(self.sentences[:1])

        assert_equal(indices, expected_indices)

        sentences = decoder(indices)

        assert isinstance(sentences, list)

        assert sentences[0] == self.sentences[0]

    @pytest.mark.parametrize("left_pad", [False, True])
    def test_encodes_decodes_batch_correctly(self, left_pad: bool) -> None:
        spm = self.build_model()

        encoder = SentencePieceEncoder(spm, device=device, left_pad=left_pad)
        decoder = SentencePieceDecoder(spm)

        expected_indices = self.build_batch(self.token_indices, left_pad=left_pad)

        indices = encoder(self.sentences)

        assert_equal(indices, expected_indices)

        sentences = decoder(indices)

        assert sentences == self.sentences

    def test_decode_ignores_control_token(self) -> None:
        spm = self.build_model(control_tokens=["<foo>"])

        encoder = SentencePieceEncoder(spm, device=device)
        decoder = SentencePieceDecoder(spm)

        expected_indices = self.build_batch(self.token_indices)

        indices = encoder(self.sentences)

        assert_equal(indices, expected_indices)

        # We inject a dummy <foo> token to the returned tokens.
        foo_idx = spm.token_to_index("<foo>")

        foo_indices = torch.full(
            (len(indices), 1), foo_idx, device=device, dtype=indices.dtype
        )

        indices = torch.cat([indices[:, :2], foo_indices, indices[:, 2:]], dim=1)

        # We expect the decoder to ignore the <foo> tokens.
        sentences = decoder(indices)

        assert sentences == self.sentences

    @pytest.mark.parametrize("left_pad", [False, True])
    def test_encode_adds_prefix_and_suffix_tokens_correctly(
        self, left_pad: bool
    ) -> None:
        spm = self.build_model(control_tokens=["<foo1>", "<foo2>", "<foo3>"])

        encoder = SentencePieceEncoder(
            spm,
            device=device,
            left_pad=left_pad,
            prefix_tokens=["<foo1>", "<s>"],
            suffix_tokens=["<foo2>", "</s>", "<foo3>"],
        )
        decoder = SentencePieceDecoder(spm)

        foo1_idx = spm.token_to_index("<foo1>")
        foo2_idx = spm.token_to_index("<foo2>")
        foo3_idx = spm.token_to_index("<foo3>")

        expected_indices = self.build_batch(
            self.token_indices,
            left_pad=left_pad,
            prefix_token_indices=[foo1_idx, spm.bos_idx],
            suffix_token_indices=[foo2_idx, spm.eos_idx, foo3_idx],
        )

        indices = encoder(self.sentences)

        assert_equal(indices, expected_indices)

        # We expect the decoder to ignore the prefix and suffix tokens.
        sentences = decoder(indices)

        assert sentences == self.sentences

    @pytest.mark.parametrize("left_pad", [False, True])
    def test_encodes_decodes_in_reverse_correctly(self, left_pad: bool) -> None:
        spm = self.build_model()

        encoder = SentencePieceEncoder(
            spm, device=device, left_pad=left_pad, reverse=True
        )
        decoder = SentencePieceDecoder(spm, reverse=True)

        expected_indices = self.build_batch(
            self.token_indices, left_pad=left_pad, reverse=True
        )

        indices = encoder(self.sentences)

        assert_equal(indices, expected_indices)

        sentences = decoder(indices)

        assert sentences == self.sentences

    def test_encode_handles_dtype_correctly(self) -> None:
        spm = self.build_model()

        # We use int64 instead of the default int32.
        encoder = SentencePieceEncoder(spm, device=device, dtype=torch.int64)
        decoder = SentencePieceDecoder(spm)

        expected_indices = self.build_batch(self.token_indices, dtype=torch.int64)

        indices = encoder(self.sentences)

        assert_equal(indices, expected_indices)

        sentences = decoder(indices)

        assert sentences == self.sentences

    @pytest.mark.parametrize("batch_size", [16])
    def test_encode_handles_batch_size_correctly(self, batch_size: int) -> None:
        spm = self.build_model()

        encoder = SentencePieceEncoder(spm, device=device, batch_size=batch_size)

        expected_indices = self.build_batch(self.token_indices)

        # If the specified batch_size is larger than the actual batch size, we
        # expect the encoder to pad the batch dimension.
        pad_len = batch_size - len(self.token_indices)
        if pad_len > 0:
            expected_indices = F.pad(expected_indices, (0, 0, 0, pad_len))

        indices = encoder(self.sentences)

        assert_equal(indices, expected_indices)

    @pytest.mark.parametrize(
        "pad_to_len,left_pad", [(1, False), (64, False), (0, True), (64, True)]
    )
    def test_encode_handles_pad_to_length_correctly(
        self, pad_to_len: int, left_pad: bool
    ) -> None:
        spm = self.build_model()

        encoder = SentencePieceEncoder(
            spm, device=device, left_pad=left_pad, pad_to_length=pad_to_len
        )
        decoder = SentencePieceDecoder(spm)

        expected_indices = self.build_batch(self.token_indices, left_pad=left_pad)

        # If the specified pad_to_length is larger than the size of the sequence
        # dimension, we expect the encoder to pad the sequence dimension.
        pad_len = pad_to_len - expected_indices.size(1)
        if pad_len > 0:
            if left_pad:
                expected_indices = F.pad(expected_indices, (pad_len, 0))
            else:
                expected_indices = F.pad(expected_indices, (0, pad_len))

        indices = encoder(self.sentences)

        assert_equal(indices, expected_indices)

        sentences = decoder(indices)

        assert sentences == self.sentences

    @pytest.mark.parametrize(
        "pad_to_mul,left_pad", [(6, False), (32, False), (6, True), (32, True)]
    )
    def test_encode_handles_pad_to_multiple_correctly(
        self, pad_to_mul: int, left_pad: bool
    ) -> None:
        spm = self.build_model()

        pad_to_len = 17

        encoder = SentencePieceEncoder(
            spm,
            device=device,
            left_pad=left_pad,
            pad_to_length=pad_to_len,
            pad_to_multiple=pad_to_mul,
        )

        expected_indices = self.build_batch(self.token_indices, left_pad=left_pad)

        # Compute the expected padded size of the sequence dimension using the
        # specified pad_to_multiple value.
        seq_dim = max(expected_indices.size(1), pad_to_len)

        r = seq_dim % pad_to_mul
        if r == 0:
            padded_seq_dim = seq_dim
        else:
            padded_seq_dim = seq_dim - r + pad_to_mul

        # We expect the encoder to pad the sequence dimension to padded_seq_dim.
        pad_len = padded_seq_dim - expected_indices.size(1)
        if pad_len > 0:
            if left_pad:
                expected_indices = F.pad(expected_indices, (pad_len, 0))
            else:
                expected_indices = F.pad(expected_indices, (0, pad_len))

        indices = encoder(self.sentences)

        assert_equal(indices, expected_indices)

    def test_pickles_correctly(self) -> None:
        spm = self.build_model(control_tokens=["<foo1>", "<foo2>", "<foo3>"])

        dmp = pickle.dumps(spm)

        # We expect that the entire state of the model including our control
        # tokens are pickled.
        spm = pickle.loads(dmp)

        encoder = SentencePieceEncoder(
            spm,
            device=device,
            prefix_tokens=["<foo1>", "<s>"],
            suffix_tokens=["<foo2>", "</s>", "<foo3>"],
        )
        decoder = SentencePieceDecoder(spm)

        foo1_idx = spm.token_to_index("<foo1>")
        foo2_idx = spm.token_to_index("<foo2>")
        foo3_idx = spm.token_to_index("<foo3>")

        expected_indices = self.build_batch(
            self.token_indices,
            prefix_token_indices=[foo1_idx, spm.bos_idx],
            suffix_token_indices=[foo2_idx, spm.eos_idx, foo3_idx],
        )

        indices = encoder(self.sentences)

        assert_equal(indices, expected_indices)

        # We expect the decoder to ignore the prefix and suffix tokens.
        sentences = decoder(indices)

        assert sentences == self.sentences

    @staticmethod
    def build_model(
        control_tokens: Optional[Sequence[str]] = None,
    ) -> SentencePieceModel:
        ct = ["<pad>@0"]

        if control_tokens is not None:
            ct += control_tokens

        return SentencePieceModel(TEST_SPM_PATH, control_tokens=ct)

    @staticmethod
    def build_batch(
        batch: Sequence[List[int]],
        reverse: bool = False,
        left_pad: bool = False,
        prefix_token_indices: Optional[List[int]] = None,
        suffix_token_indices: Optional[List[int]] = None,
        dtype: torch.dtype = torch.int32,
    ) -> Tensor:
        prefix = prefix_token_indices or []
        suffix = suffix_token_indices or []

        b: List[Tensor] = []

        for token_indices in batch:
            token_indices = prefix + token_indices + suffix

            b.append(torch.tensor(token_indices, device=device, dtype=dtype))

        if reverse:
            b = [t.flip(dims=[0]) for t in b]

        if left_pad:
            b = [t.flip(dims=[0]) for t in b]

        t = torch.nn.utils.rnn.pad_sequence(b, batch_first=True)

        if left_pad:
            t = t.flip(dims=[1])

        return t
