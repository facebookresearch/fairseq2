# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

from fairseq2.metrics.formatters import (
    format_as_percentage,
    scale_as_percentage,
)


class TestScaleAsPercentage:
    def test_scales_float(self) -> None:
        assert scale_as_percentage(0.42) == 42.0

    def test_scales_zero(self) -> None:
        assert scale_as_percentage(0.0) == 0.0

    def test_scales_one(self) -> None:
        assert scale_as_percentage(1.0) == 100.0

    def test_scales_int(self) -> None:
        assert scale_as_percentage(1) == 100.0

    def test_returns_non_numeric_unchanged(self) -> None:
        assert scale_as_percentage("hello") == "hello"

    def test_returns_none_unchanged(self) -> None:
        assert scale_as_percentage(None) is None

    def test_scales_tensor(self) -> None:
        t = torch.tensor(0.42)
        result = scale_as_percentage(t)
        assert torch.is_tensor(result)
        assert float(result) == pytest.approx(42.0)


class TestFormatAsPercentage:
    def test_formats_float(self) -> None:
        assert format_as_percentage(0.4212) == "42.12%"

    def test_formats_zero(self) -> None:
        assert format_as_percentage(0.0) == "0.00%"

    def test_formats_one(self) -> None:
        assert format_as_percentage(1.0) == "100.00%"

    def test_formats_tensor(self) -> None:
        assert format_as_percentage(torch.tensor(0.5)) == "50.00%"

    def test_formats_string_number(self) -> None:
        assert format_as_percentage("0.75") == "75.00%"

    def test_formats_non_numeric_string(self) -> None:
        assert format_as_percentage("abc") == "abc"

    def test_keeps_two_decimal_places(self) -> None:
        assert format_as_percentage(0.4213) == "42.13%"
