# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest

from fairseq2.data.text import SentencePieceModel

NLLB_SPM_PATH = Path(
    "/large_experiments/seamless/nllb/opensource/spm_200/sentencepiece.source.256000.model"
)


# TODO: Remove this limitation once we can download the models!
@pytest.mark.skipif(not NLLB_SPM_PATH.exists(), reason="needs to run on FAIR cluster")
class TestSentencePieceModel:
    def test_pad_is_correctly_added(self) -> None:
        spm = SentencePieceModel(NLLB_SPM_PATH, control_tokens=["<pad>"])

        assert spm.pad_idx == 256000
        assert spm.unk_idx == 0

    def test_pad_is_correctly_added_at_index_0(self) -> None:
        # Note that this is an undocumented feature and is not part of our
        # public API.
        spm = SentencePieceModel(NLLB_SPM_PATH, control_tokens=["<pad>@0"])

        assert spm.pad_idx == 0
        assert spm.unk_idx == 1
