# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Final

import pytest
import torch
from fairseq2n import supports_image

from fairseq2.data.image import ImageDecoder
from fairseq2.memory import MemoryBlock
from tests.common import assert_close, device

TEST_PNG_PATH: Final = Path(__file__).parent.joinpath("test.png")
TEST_JPG_PATH: Final = Path(__file__).parent.joinpath("test.jpg")
TEST_CORRUPT_JPG_PATH: Final = Path(__file__).parent.joinpath("test_corrupt.jpg")
TEST_CORRUPT_PNG_PATH: Final = Path(__file__).parent.joinpath("test_corrupt.png")


@pytest.mark.skipif(
    not supports_image(), reason="fairseq2n is not built with JPEG/PNG decoding support"
)
class TestImageDecoder:
    def test_init(self) -> None:
        decoder = ImageDecoder()
        assert isinstance(decoder, ImageDecoder)

    def test_call_works_on_png(self) -> None:
        decoder = ImageDecoder(device=device)

        with TEST_PNG_PATH.open("rb") as fb:
            block = MemoryBlock(fb.read())

        output = decoder(block)

        assert output["bit_depth"] == 8.0

        assert output["color_type"] == 6.0

        assert output["channels"] == 4.0

        assert output["height"] == 70.0

        assert output["width"] == 70.0

        image = output["image"]

        assert image.shape == torch.Size([70, 70, 4])

        assert image.dtype == torch.uint8

        assert image.device == device

        assert_close(image.sum(), torch.tensor(4656924, device=device))

    #    def test_call_works_on_jpg(self) -> None:
    #        decoder = ImageDecoder(device=device)
    #
    #        with TEST_JPG_PATH.open("rb") as fb:
    #            block = MemoryBlock(fb.read())
    #
    #        output = decoder(block)
    #
    #        assert output["bit_depth"] == 8.0
    #
    #        assert output["channels"] == 3.0
    #
    #        assert output["height"] == 50.0
    #
    #        assert output["width"] == 50.0
    #
    #        image = output["image"]
    #
    #        assert image.shape == torch.Size([50, 50, 3])
    #
    #        assert image.dtype == torch.uint8
    #
    #        assert image.device == device
    #
    #        assert_close(image.sum(), torch.tensor(1747686, device=device))

    def test_call_raises_error_when_input_is_corrupted_png(self) -> None:
        decoder = ImageDecoder(device=device)

        with TEST_CORRUPT_PNG_PATH.open("rb") as fb:
            block = MemoryBlock(fb.read())

        with pytest.raises(
            RuntimeError,
            match="libpng internal error.",
        ):
            decoder(block)

    #    def test_call_raises_error_when_input_is_corrupted_jpg(self) -> None:
    #        decoder = ImageDecoder(device=device)
    #
    #        with TEST_CORRUPT_JPG_PATH.open("rb") as fb:
    #            block = MemoryBlock(fb.read())
    #
    #        with pytest.raises(
    #            RuntimeError,
    #            match="JPEG decompression failed.",
    #        ):
    #            decoder(block)
    #
    @pytest.mark.parametrize(
        "value,type_name", [(None, "pyobj"), (123, "int"), ("s", "string")]
    )
    def test_call_raises_error_when_input_is_not_memory_block(
        self, value: Any, type_name: str
    ) -> None:
        decoder = ImageDecoder()

        with pytest.raises(
            ValueError,
            match=rf"^The input data must be of type `memory_block`, but is of type `{type_name}` instead\.$",
        ):
            decoder(value)

    def test_call_raises_error_when_input_is_empty(self) -> None:
        decoder = ImageDecoder()

        empty_block = MemoryBlock()

        with pytest.raises(
            ValueError,
            match=r"^The input memory block has zero length and cannot be decoded\.$",
        ):
            decoder(empty_block)

    def test_call_raises_error_when_input_is_invalid(self) -> None:
        decoder = ImageDecoder()

        block = MemoryBlock(b"foo")

        with pytest.raises(
            ValueError,
            match=r"^Unsupported image file. Only jpeg and png are currently supported\.$",
        ):
            decoder(block)
