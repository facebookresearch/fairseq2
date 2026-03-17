# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from fairseq2.error import NotSupportedError
from fairseq2.models.hg.converter import (
    HuggingFaceConfig,
    _LegacyHuggingFaceConverter,
    get_hugging_face_converter,
    save_hugging_face_model,
)


class TestLegacyHuggingFaceConverter:
    """Test the _LegacyHuggingFaceConverter class."""

    def test_to_hg_config_raises_not_supported(self) -> None:
        """Legacy converter's to_hg_config always raises NotSupportedError."""
        exporter = MagicMock()
        converter = _LegacyHuggingFaceConverter(exporter)
        with pytest.raises(NotSupportedError):
            converter.to_hg_config(object())

    def test_to_hg_state_dict_raises_not_supported(self) -> None:
        """Legacy converter's to_hg_state_dict always raises NotSupportedError."""
        exporter = MagicMock()
        converter = _LegacyHuggingFaceConverter(exporter)
        with pytest.raises(NotSupportedError):
            converter.to_hg_state_dict({}, object())


class TestGetHuggingFaceConverter:
    """Test the get_hugging_face_converter function."""

    @patch("fairseq2.models.hg.converter.get_dependency_resolver")
    def test_raises_not_supported_for_unknown_family(
        self, mock_get_resolver: MagicMock
    ) -> None:
        """Raises NotSupportedError when model family has no converter."""
        mock_resolver = MagicMock()
        mock_resolver.maybe_resolve.return_value = None
        mock_get_resolver.return_value = mock_resolver

        with pytest.raises(
            NotSupportedError, match="does not support Hugging Face conversion"
        ):
            get_hugging_face_converter("nonexistent_family")

    @patch("fairseq2.models.hg.converter.get_dependency_resolver")
    def test_returns_converter_for_known_family(
        self, mock_get_resolver: MagicMock
    ) -> None:
        """Returns converter when model family is registered."""
        mock_converter = MagicMock()
        mock_resolver = MagicMock()
        mock_resolver.maybe_resolve.return_value = mock_converter
        mock_get_resolver.return_value = mock_resolver

        result = get_hugging_face_converter("known_family")
        assert result is mock_converter


class TestSaveHuggingFaceModel:
    """Test the save_hugging_face_model function."""

    def test_invalid_config_class_name_raises_type_error(self) -> None:
        """TypeError when kls_name doesn't exist in transformers module."""
        config = HuggingFaceConfig(
            data={}, kls_name="CompletelyFakeConfig999", arch="FakeArch"
        )
        with pytest.raises(TypeError, match="is not a type"):
            save_hugging_face_model(Path("/tmp/test"), {}, config)

    @patch("fairseq2.models.hg.converter.transformers")
    def test_non_pretrained_config_raises_type_error(
        self, mock_transformers: MagicMock
    ) -> None:
        """TypeError when config class is not a PretrainedConfig subclass."""

        class NotAConfig:
            pass

        mock_transformers.NotAConfig = NotAConfig

        config = HuggingFaceConfig(data={}, kls_name="NotAConfig", arch="FakeArch")
        with pytest.raises(TypeError, match="expected to be a subclass"):
            save_hugging_face_model(Path("/tmp/test"), {}, config)

    @patch("fairseq2.models.hg.converter.huggingface_hub")
    @patch("fairseq2.models.hg.converter.transformers")
    def test_invalid_config_attribute_raises_value_error(
        self, mock_transformers: MagicMock, mock_hf_hub: MagicMock
    ) -> None:
        """ValueError when config data key doesn't exist on HF config class."""
        from transformers import PretrainedConfig

        class FakeConfig(PretrainedConfig):
            model_type = "fake"

        mock_transformers.FakeConfig = FakeConfig

        config = HuggingFaceConfig(
            data={"nonexistent_attr": 42}, kls_name="FakeConfig", arch="FakeArch"
        )
        with pytest.raises(ValueError, match="does not have an attribute"):
            save_hugging_face_model(Path("/tmp/test"), {}, config)

    @patch("fairseq2.models.hg.converter.huggingface_hub")
    @patch("fairseq2.models.hg.converter.transformers")
    def test_non_tensor_state_dict_raises_type_error(
        self, mock_transformers: MagicMock, mock_hf_hub: MagicMock
    ) -> None:
        """TypeError when state dict contains non-Tensor values."""
        from transformers import PretrainedConfig

        class FakeConfig(PretrainedConfig):
            model_type = "fake"

        mock_transformers.FakeConfig = FakeConfig

        state_dict: dict[str, Any] = {"layer.weight": "not_a_tensor"}
        config = HuggingFaceConfig(data={}, kls_name="FakeConfig", arch="FakeArch")
        with pytest.raises(TypeError, match="must be of type"):
            save_hugging_face_model(Path("/tmp/test"), state_dict, config)

    @patch("fairseq2.models.hg.converter.huggingface_hub")
    @patch("fairseq2.models.hg.converter.transformers")
    def test_valid_save(
        self, mock_transformers: MagicMock, mock_hf_hub: MagicMock
    ) -> None:
        """Successful save with valid inputs."""
        from transformers import PretrainedConfig

        class FakeConfig(PretrainedConfig):
            model_type = "fake"

        mock_transformers.FakeConfig = FakeConfig

        state_dict = {"layer.weight": torch.randn(10, 10)}
        config = HuggingFaceConfig(data={}, kls_name="FakeConfig", arch="FakeArch")
        save_hugging_face_model(Path("/tmp/test"), state_dict, config)

        mock_hf_hub.save_torch_state_dict.assert_called_once()
