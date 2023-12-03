# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Final

import pytest
import torch

from fairseq2.data.video import VideoDecoder
from fairseq2.memory import MemoryBlock
from fairseq2.typing import DataType
from tests.common import assert_close, device

TEST_MP4_PATH: Final = Path(__file__).parent.joinpath("test.mp4")
TEST_UNSUPPORTED_CODEC_PATH: Final = Path(__file__).parent.joinpath("test2.MP4")


class TestVideoDecoder:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.int64])
    def test_init_raises_error_when_data_type_is_not_supported(
        self, dtype: DataType
    ) -> None:
        with pytest.raises(
            ValueError,
            match=r"^`video_decoder` supports only `torch.int16` and `torch.uint8` data types\.$",
        ):
            VideoDecoder(dtype=dtype)

    def test_call_works(self) -> None:
        decoder = VideoDecoder(device=device)

        with TEST_MP4_PATH.open("rb") as fb:
            block = MemoryBlock(fb.read())

        output = decoder(block)

        video = output["video"]["0"]["all_video_frames"]

        assert  video.shape == torch.Size([24, 2160, 3840, 3])

        assert  video.dtype == torch.uint8

        assert  video.device == device

        assert_close(video[0][0][0][0], torch.tensor(80, dtype=torch.uint8))

        assert_close(video.sum(), torch.tensor(3816444602, device=device))

        assert_close(output["video"]["0"]["frame_pts"].sum(), torch.tensor(11511489))

        assert_close(output["video"]["0"]["timebase"], torch.tensor([1, 24000], dtype=torch.int32))

        assert_close(output["video"]["0"]["fps"][0], torch.tensor(23.9760))

        assert_close(output["video"]["0"]["duration"][0], torch.tensor(1024000))

    def test_call_raises_error_when_codec_is_not_supported(self) -> None:
        decoder = VideoDecoder()

        with TEST_UNSUPPORTED_CODEC_PATH.open("rb") as fb:
            block = MemoryBlock(fb.read())

        with pytest.raises(
            RuntimeError,
            match=r"^Failed to find decoder for stream 2",
        ):
            decoder(block)

    @pytest.mark.parametrize(
        "value,type_name", [(None, "pyobj"), (123, "int"), ("s", "string")]
    )
    def test_call_raises_error_when_input_is_not_memory_block(
        self, value: Any, type_name: str
    ) -> None:
        decoder = VideoDecoder()

        with pytest.raises(
            ValueError,
            match=rf"^The input data must be of type `memory_block`, but is of type `{type_name}` instead\.$",
        ):
            decoder(value)

    def test_call_raises_error_when_input_is_empty(self) -> None:
        decoder = VideoDecoder()

        empty_block = MemoryBlock()

        with pytest.raises(
            ValueError,
            match=r"^The input memory block has zero length and cannot be decoded\.$",
        ):
            decoder(empty_block)

    def test_call_raises_error_when_input_is_invalid(self) -> None:
        decoder = VideoDecoder()

        block = MemoryBlock(b"foo")

        with pytest.raises(
            ValueError,
            match=r"^Failed to open input\.$",
        ):
            decoder(block)
