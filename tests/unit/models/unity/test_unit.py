# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairseq2.models.unity import UnitTokenizer
from tests.common import assert_equal, device


class TestUnitTokenizer:
    def test_init_works(self) -> None:
        tokenizer = UnitTokenizer(num_units=100, langs=["eng", "deu", "fra"])

        assert tokenizer.num_units == 100

        assert tokenizer.langs == {"eng": 0, "deu": 1, "fra": 2}

        assert tokenizer.vocabulary_info.size == 112

    def test_lang_to_index_works(self) -> None:
        tokenizer = UnitTokenizer(num_units=100, langs=["eng", "deu", "fra"])

        assert tokenizer.lang_to_index("deu") == 105


class TestUnitEncoder:
    def test_init_raises_error_when_lang_is_not_supported(self) -> None:
        tokenizer = UnitTokenizer(num_units=100, langs=["eng", "deu", "fra"])

        with pytest.raises(
            ValueError,
            match=r"^`lang` must be one of the supported languages\, but is 'xyz' instead\. Supported languages: eng, deu, fra$",
        ):
            tokenizer.create_encoder(lang="xyz")

    def test_call_works(self) -> None:
        tokenizer = UnitTokenizer(num_units=100, langs=["eng", "deu", "fra"])

        prefix = torch.tensor([2, 105], device=device, dtype=torch.int)

        encoder = tokenizer.create_encoder(lang="deu")

        # Empty units.
        units = torch.ones((1, 0), device=device, dtype=torch.int)

        assert_equal(encoder(units), prefix.expand(1, -1))

        # Batched units.
        units = torch.ones((6, 4), device=device, dtype=torch.int)

        assert_equal(
            encoder(units), torch.cat([prefix.expand(6, -1), units + 4], dim=1)
        )

    def test_call_works_when_units_have_unks(self) -> None:
        tokenizer = UnitTokenizer(num_units=100, langs=["eng", "deu", "fra"])

        encoder = tokenizer.create_encoder(lang="deu")

        units = torch.ones((6, 4), device=device, dtype=torch.int)

        units[1, 3] = 100
        units[2, 1] = 101

        token_indices = encoder(units)

        assert token_indices[1, 5].item() == tokenizer.vocabulary_info.unk_idx
        assert token_indices[2, 3].item() == tokenizer.vocabulary_info.unk_idx


class TestUnitDecoder:
    def test_call_works(self) -> None:
        tokenizer = UnitTokenizer(num_units=100, langs=["eng", "deu", "fra"])

        encoder = tokenizer.create_encoder(lang="deu")
        decoder = tokenizer.create_decoder()

        assert tokenizer.vocabulary_info.eos_idx is not None
        assert tokenizer.vocabulary_info.pad_idx is not None

        units1 = torch.ones((6, 4), device=device, dtype=torch.int)

        encoded_units = encoder(units1)

        encoded_units[2, 2] = tokenizer.vocabulary_info.eos_idx

        units2 = decoder(encoded_units)

        units1[2, 2] = tokenizer.vocabulary_info.pad_idx

        prefix = torch.tensor([105], device=device, dtype=torch.int)

        assert_equal(torch.cat([prefix.expand(6, -1), units1], dim=1), units2)
