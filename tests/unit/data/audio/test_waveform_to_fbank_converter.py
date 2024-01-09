# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Final, Sequence

import pytest
import torch

from fairseq2.data.audio import (
    AudioDecoder,
    AudioDecoderOutput,
    WaveformToFbankConverter,
)
from fairseq2.memory import MemoryBlock
from tests.common import assert_equal, device

TEST_OGG_PATH: Final = Path(__file__).parent.joinpath("test.ogg")


class TestWaveformToFbankConverter:
    def test_call_works(self) -> None:
        audio = self.get_audio()

        converter = WaveformToFbankConverter(channel_last=True)

        output = converter(audio)

        fbank = output["fbank"]

        assert fbank.dtype == torch.float32

        assert fbank.shape == (178, 80)

        assert fbank.mean() == pytest.approx(9.8117, abs=0.0001)

    def test_call_works_when_input_waveform_is_strided(self) -> None:
        audio = self.get_audio()

        audio["waveform"] = audio["waveform"].transpose(0, 1)

        converter = WaveformToFbankConverter()

        output = converter(audio)

        fbank = output["fbank"]

        assert fbank.dtype == torch.float32

        assert fbank.shape == (178, 80)

        assert fbank.mean() == pytest.approx(9.8117, abs=0.0001)

    def test_call_works_when_standardize_is_true(self) -> None:
        audio = self.get_audio()

        converter = WaveformToFbankConverter(channel_last=True, standardize=True)

        output = converter(audio)

        fbank = output["fbank"]

        assert fbank.dtype == torch.float32

        assert fbank.shape == (178, 80)

        assert fbank.mean() == pytest.approx(0.0, abs=0.001)

        assert fbank.std(unbiased=True) == pytest.approx(1.0, abs=0.01)

    def test_call_works_when_keep_waveform_is_true(self) -> None:
        audio = self.get_audio()

        converter = WaveformToFbankConverter(channel_last=True, keep_waveform=True)

        output = converter(audio)

        assert_equal(output["waveform"], audio["waveform"])

    def test_call_works_when_data_type_is_specified(self) -> None:
        audio = self.get_audio()

        converter = WaveformToFbankConverter(channel_last=True, dtype=torch.float64)

        output = converter(audio)

        fbank = output["fbank"]

        assert fbank.dtype == torch.float64

    def test_call_raises_error_when_input_is_not_dict(self) -> None:
        converter = WaveformToFbankConverter()

        with pytest.raises(
            ValueError,
            match=r"^The input data must be of type `dict` containing a waveform tensor and its sample rate, but is of type `string` instead\.$",
        ):
            converter("foo")  # type: ignore[arg-type]

    def test_call_raises_error_when_input_does_not_contain_sample_rate(self) -> None:
        converter = WaveformToFbankConverter()

        with pytest.raises(
            ValueError,
            match=r"^The input dictionary must contain the waveform sample rate under a key named `sample_rate`, but does not contain such key\.$",
        ):
            converter({"waveform": torch.zeros((), device=device)})  # type: ignore[typeddict-item]

    def test_call_raises_error_when_input_does_not_contain_waveform(self) -> None:
        converter = WaveformToFbankConverter()

        with pytest.raises(
            ValueError,
            match=r"^The input dictionary must contain the waveform under a key named `waveform`, but does not contain such key\.$",
        ):
            converter({"sample_rate": 16000.0})  # type: ignore[typeddict-item]

    def test_call_raises_error_when_sample_rate_is_not_float_or_int(self) -> None:
        converter = WaveformToFbankConverter()

        with pytest.raises(
            ValueError,
            match=r"^The input sample rate must be of type `float` or `int`, but is of type `string` instead\.$",
        ):
            converter({"waveform": torch.zeros((), device=device), "sample_rate": "foo"})  # type: ignore[typeddict-item]

    def test_call_raises_error_when_sample_rate_is_too_large(self) -> None:
        converter = WaveformToFbankConverter()

        with pytest.raises(
            ValueError,
            match=r"^The input sample rate must be representable in single precision \(32-bit\), but is 3\.41E\+38 instead\.$",
        ):
            converter(
                {"waveform": torch.zeros((), device=device), "sample_rate": 3.41e38}
            )

    def test_call_raises_error_when_sample_rate_is_less_than_100(self) -> None:
        audio = self.get_audio()

        audio["sample_rate"] = 99.0

        converter = WaveformToFbankConverter(channel_last=True)

        with pytest.raises(
            ValueError,
            match=r"^The input sample rate must be greater than or equal to 100, but is 99 instead\.$",
        ):
            converter(audio)

    def test_call_raises_error_when_inputs_have_different_sample_rates(self) -> None:
        audio = self.get_audio()

        converter = WaveformToFbankConverter(channel_last=True)

        converter(audio)

        audio["sample_rate"] = 18000

        with pytest.raises(
            ValueError,
            match=r"^The input waveform must have a sample rate of 16000, but has a sample rate of 18000 instead\.$",
        ):
            converter(audio)

    def test_call_raises_error_when_waveform_is_not_tensor(self) -> None:
        converter = WaveformToFbankConverter()

        with pytest.raises(
            ValueError,
            match=r"^The input waveform must be of type `torch\.Tensor`, but is of type `string` instead\.$",
        ):
            converter({"waveform": "foo", "sample_rate": 16000.0})  # type: ignore[typeddict-item]

    @pytest.mark.parametrize("shape", [(), (4,), (4, 4, 4)])
    def test_call_raises_error_when_waveform_is_not_two_dimensional(
        self, shape: Sequence[int]
    ) -> None:
        converter = WaveformToFbankConverter()

        waveform = torch.empty(shape, device=device)

        with pytest.raises(
            ValueError,
            match=rf"^The input waveform must be two dimensional, but has {len(shape)} dimension\(s\) instead\.$",
        ):
            converter({"waveform": waveform, "sample_rate": 16000.0})

    @staticmethod
    def get_audio() -> AudioDecoderOutput:
        decoder = AudioDecoder(dtype=torch.int16, device=device)

        with TEST_OGG_PATH.open("rb") as fb:
            block = MemoryBlock(fb.read())

        return decoder(block)
