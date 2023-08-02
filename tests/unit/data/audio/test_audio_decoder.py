# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Final

import pytest
import torch

from fairseq2.data.audio import AudioDecoder
from fairseq2.memory import MemoryBlock
from fairseq2.typing import DataType
from tests.common import assert_close, device

TEST_OGG_PATH: Final = Path(__file__).parent.joinpath("test.ogg")


class TestAudioDecoder:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.int64])
    def test_init_raises_error_when_data_type_is_not_supported(
        self, dtype: DataType
    ) -> None:
        with pytest.raises(
            ValueError,
            match=r"^`audio_decoder` supports only `torch.float32`, `torch.int32`, and `torch.int16` data types\.$",
        ):
            AudioDecoder(dtype=dtype)

    def test_call_works(self) -> None:
        decoder = AudioDecoder(device=device)

        with TEST_OGG_PATH.open("rb") as fb:
            block = MemoryBlock(fb.read())

        output = decoder(block)

        assert output["format"] == 0x200060  # OGG Vorbis

        assert output["sample_rate"] == 16000

        waveform = output["waveform"]

        assert waveform.shape == (28800, 1)

        assert waveform.dtype == torch.float32

        assert waveform.device == device

        assert_close(waveform[0][0], torch.tensor(9.0017202e-6, device=device))

        assert_close(waveform.sum(), torch.tensor(-0.753374, device=device))

    @pytest.mark.parametrize(
        "value,type_name", [(None, "pyobj"), (123, "int"), ("s", "string")]
    )
    def test_call_raises_error_when_input_is_not_memory_block(
        self, value: Any, type_name: str
    ) -> None:
        decoder = AudioDecoder()

        with pytest.raises(
            ValueError,
            match=rf"^The input data must be of type `memory_block`, but is of type `{type_name}` instead\.$",
        ):
            decoder(value)

    def test_call_raises_error_when_input_is_empty(self) -> None:
        decoder = AudioDecoder()

        empty_block = MemoryBlock()

        with pytest.raises(
            ValueError,
            match=r"^The input memory block has zero length and cannot be decoded as audio\.$",
        ):
            decoder(empty_block)

    def test_call_raises_error_when_input_is_invalid(self) -> None:
        decoder = AudioDecoder()

        block = MemoryBlock(b"foo")

        with pytest.raises(
            ValueError,
            match=r"^The input audio cannot be decoded. See nested exception for details\.$",
        ):
            decoder(block)
