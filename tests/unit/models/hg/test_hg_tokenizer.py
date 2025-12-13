# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor

from fairseq2.data.tokenizers import VocabularyInfo
from fairseq2.device import CPU
from fairseq2.models.hg_qwen_omni.tokenizer import (
    HgTokenizer,
    HgTokenizerConfig,
    load_hg_tokenizer,
)


class TestHgTokenizerConfig:
    """Test the HgTokenizerConfig dataclass."""

    def test_config_creation_with_defaults(self) -> None:
        """Test creating config with default values."""
        config = HgTokenizerConfig()

        assert config.unk_token is None
        assert config.bos_token is None
        assert config.eos_token is None
        assert config.pad_token is None
        assert config.boh_token is None
        assert config.eoh_token is None

    def test_config_creation_with_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = HgTokenizerConfig(
            unk_token="<unk>",
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            boh_token="<boh>",
            eoh_token="<eoh>",
        )

        assert config.unk_token == "<unk>"
        assert config.bos_token == "<bos>"
        assert config.eos_token == "<eos>"
        assert config.pad_token == "<pad>"
        assert config.boh_token == "<boh>"
        assert config.eoh_token == "<eoh>"

    def test_config_equality(self) -> None:
        """Test config equality comparison."""
        config1 = HgTokenizerConfig(
            unk_token="<unk>",
            bos_token="<bos>",
        )
        config2 = HgTokenizerConfig(
            unk_token="<unk>",
            bos_token="<bos>",
        )
        config3 = HgTokenizerConfig(
            unk_token="<unk>",
            bos_token="<s>",  # Different bos_token
        )

        assert config1 == config2
        assert config1 != config3


class TestHgTokenizer:
    """Test the HgTokenizer class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_model = MagicMock()
        self.mock_vocab_info = MagicMock(spec=VocabularyInfo)
        self.mock_model.vocab_info = self.mock_vocab_info
        self.tokenizer = HgTokenizer(self.mock_model)

    def test_init(self) -> None:
        """Test tokenizer initialization."""
        assert self.tokenizer._model is self.mock_model
        assert self.tokenizer._encoder is None
        assert self.tokenizer._decoder is None

    @patch("fairseq2.models.hg_qwen_omni.tokenizer.HuggingFaceTokenEncoder")
    def test_create_encoder(self, mock_encoder_class: MagicMock) -> None:
        """Test encoder creation."""
        mock_encoder = MagicMock()
        mock_encoder_class.return_value = mock_encoder

        encoder = self.tokenizer.create_encoder()

        mock_encoder_class.assert_called_once_with(
            self.mock_model, device=None, pin_memory=False
        )
        assert encoder is mock_encoder
        assert self.tokenizer._encoder is mock_encoder

    @patch("fairseq2.models.hg_qwen_omni.tokenizer.HuggingFaceTokenEncoder")
    def test_create_encoder_with_device(self, mock_encoder_class: MagicMock) -> None:
        """Test encoder creation with device."""
        mock_encoder = MagicMock()
        mock_encoder_class.return_value = mock_encoder

        encoder = self.tokenizer.create_encoder(device=CPU, pin_memory=True)

        mock_encoder_class.assert_called_once_with(
            self.mock_model, device=CPU, pin_memory=True
        )
        assert encoder is mock_encoder

    @patch("fairseq2.models.hg_qwen_omni.tokenizer.HuggingFaceTokenEncoder")
    def test_create_raw_encoder(self, mock_encoder_class: MagicMock) -> None:
        """Test raw encoder creation."""
        mock_encoder = MagicMock()
        mock_encoder_class.return_value = mock_encoder

        encoder = self.tokenizer.create_raw_encoder()

        mock_encoder_class.assert_called_once_with(
            self.mock_model, device=None, pin_memory=False
        )
        assert encoder is mock_encoder
        assert self.tokenizer._encoder is mock_encoder

    def test_create_raw_encoder_cached(self) -> None:
        """Test that raw encoder is cached."""
        mock_encoder = MagicMock()
        self.tokenizer._encoder = mock_encoder

        encoder = self.tokenizer.create_raw_encoder()

        assert encoder is mock_encoder

    @patch("fairseq2.models.hg_qwen_omni.tokenizer.HuggingFaceTokenDecoder")
    def test_create_decoder(self, mock_decoder_class: MagicMock) -> None:
        """Test decoder creation."""
        mock_decoder = MagicMock()
        mock_decoder_class.return_value = mock_decoder

        decoder = self.tokenizer.create_decoder()

        mock_decoder_class.assert_called_once_with(
            self.mock_model, skip_special_tokens=False
        )
        assert decoder is mock_decoder
        assert self.tokenizer._decoder is mock_decoder

    @patch("fairseq2.models.hg_qwen_omni.tokenizer.HuggingFaceTokenDecoder")
    def test_create_decoder_skip_special_tokens(
        self, mock_decoder_class: MagicMock
    ) -> None:
        """Test decoder creation with skip_special_tokens."""
        mock_decoder = MagicMock()
        mock_decoder_class.return_value = mock_decoder

        decoder = self.tokenizer.create_decoder(skip_special_tokens=True)

        mock_decoder_class.assert_called_once_with(
            self.mock_model, skip_special_tokens=True
        )
        assert decoder is mock_decoder

    def test_create_decoder_cached(self) -> None:
        """Test that decoder is cached."""
        mock_decoder = MagicMock()
        self.tokenizer._decoder = mock_decoder

        decoder = self.tokenizer.create_decoder()

        assert decoder is mock_decoder

    def test_encode(self) -> None:
        """Test text encoding."""
        mock_encoder = MagicMock()
        mock_tensor = MagicMock(spec=Tensor)
        mock_encoder.return_value = mock_tensor
        self.tokenizer._encoder = mock_encoder

        result = self.tokenizer.encode("Hello world")

        mock_encoder.assert_called_once_with("Hello world")
        assert result is mock_tensor

    def test_decode(self) -> None:
        """Test token decoding."""
        mock_decoder = MagicMock()
        mock_decoder.return_value = "Hello world"
        self.tokenizer._decoder = mock_decoder

        input_tensor = torch.as_tensor([1, 2, 3])
        result = self.tokenizer.decode(input_tensor)

        # Verify decoder was called once
        mock_decoder.assert_called_once()
        # Verify the tensor argument by checking it's equivalent
        call_args = mock_decoder.call_args[0][0]
        assert torch.equal(call_args, input_tensor)
        assert result == "Hello world"

    def test_vocab_info(self) -> None:
        """Test vocab info property."""
        assert self.tokenizer.vocab_info is self.mock_vocab_info

    def test_token_properties_with_attributes(self) -> None:
        """Test token properties when attributes exist."""
        mock_tok = MagicMock()
        mock_tok.unk_token = "<unk>"
        mock_tok.bos_token = "<bos>"
        mock_tok.eos_token = "<eos>"
        mock_tok.pad_token = "<pad>"
        mock_tok.boh_token = "<boh>"
        mock_tok.eoh_token = "<eoh>"
        self.mock_model._tok = mock_tok

        assert self.tokenizer.unk_token == "<unk>"
        assert self.tokenizer.bos_token == "<bos>"
        assert self.tokenizer.eos_token == "<eos>"
        assert self.tokenizer.pad_token == "<pad>"
        assert self.tokenizer.boh_token == "<boh>"
        assert self.tokenizer.eoh_token == "<eoh>"

    def test_token_properties_without_attributes(self) -> None:
        """Test token properties when attributes don't exist."""
        mock_tok = MagicMock()
        # Remove attributes to test hasattr check
        for attr in [
            "unk_token",
            "bos_token",
            "eos_token",
            "pad_token",
            "boh_token",
            "eoh_token",
        ]:
            if hasattr(mock_tok, attr):
                delattr(mock_tok, attr)
        self.mock_model._tok = mock_tok

        assert self.tokenizer.unk_token is None
        assert self.tokenizer.bos_token is None
        assert self.tokenizer.eos_token is None
        assert self.tokenizer.pad_token is None
        assert self.tokenizer.boh_token is None
        assert self.tokenizer.eoh_token is None

    def test_raw_property(self) -> None:
        """Test raw tokenizer property."""
        mock_tok = MagicMock()
        self.mock_model._tok = mock_tok

        assert self.tokenizer.raw is mock_tok

    def test_model_property(self) -> None:
        """Test model property."""
        assert self.tokenizer.model is self.mock_model


class TestLoadHgTokenizer:
    """Test the load_hg_tokenizer function."""

    @patch("fairseq2.models.hg_qwen_omni.tokenizer.load_hg_token_model")
    def test_load_hg_tokenizer_basic(self, mock_load_model: MagicMock) -> None:
        """Test basic tokenizer loading."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        path = Path("/test/path")
        config = HgTokenizerConfig()
        result = load_hg_tokenizer(path, config)

        mock_load_model.assert_called_once_with(
            path,
            unk_token=None,
            bos_token=None,
            eos_token=None,
            pad_token=None,
            boh_token=None,
            eoh_token=None,
        )
        assert isinstance(result, HgTokenizer)
        assert result._model is mock_model

    @patch("fairseq2.models.hg_qwen_omni.tokenizer.load_hg_token_model")
    def test_load_hg_tokenizer_with_tokens(self, mock_load_model: MagicMock) -> None:
        """Test tokenizer loading with custom tokens."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        path = Path("/test/path")
        config = HgTokenizerConfig(
            unk_token="<unk>",
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            boh_token="<boh>",
            eoh_token="<eoh>",
        )
        result = load_hg_tokenizer(path, config)

        mock_load_model.assert_called_once_with(
            path,
            unk_token="<unk>",
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            boh_token="<boh>",
            eoh_token="<eoh>",
        )
        assert isinstance(result, HgTokenizer)
        assert result._model is mock_model

    @patch("fairseq2.models.hg_qwen_omni.tokenizer.load_hg_token_model")
    def test_load_hg_tokenizer_propagates_errors(
        self, mock_load_model: MagicMock
    ) -> None:
        """Test that load_hg_tokenizer propagates errors."""
        mock_load_model.side_effect = ValueError("Token model loading failed")

        path = Path("/test/path")
        config = HgTokenizerConfig()

        with pytest.raises(ValueError) as exc_info:
            load_hg_tokenizer(path, config)

        assert str(exc_info.value) == "Token model loading failed"
