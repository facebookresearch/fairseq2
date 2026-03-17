# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for the HuggingFace model API surface.

These tests verify that the fairseq2 API correctly delegates to HuggingFace
Auto classes (AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, etc.)
without downloading real models. All HF calls are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from fairseq2.models.hg.adapter import HgCausalLMAdapter
from fairseq2.models.hg.api import (
    load_causal_lm,
    load_hg_model_simple,
    load_hg_tokenizer_simple,
    load_seq2seq_lm,
)
from fairseq2.models.hg.config import HuggingFaceModelConfig
from fairseq2.models.hg.factory import (
    _get_auto_model_class,
    create_hg_model,
)


def _make_mock_hf_causal_model():
    """Create a real Module-based mock for causal LM (needed by add_module)."""

    class FakeCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(50257, 768)
            self.config = MagicMock()
            self.config.max_position_embeddings = 1024

        def forward(self, **kwargs):
            output = MagicMock()
            output.loss = torch.tensor(2.5)
            input_ids = kwargs.get("input_ids", torch.zeros(1, 5, dtype=torch.long))
            output.logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 50257)
            return output

    return FakeCausalLM()


def _make_mock_hf_seq2seq_model():
    """Create a mock HF seq2seq model."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.__class__.__name__ = "T5Config"
    model.config.is_encoder_decoder = True
    return model


class TestLoadCausalLmAPI:
    """Test that load_causal_lm correctly delegates to AutoModelForCausalLM."""

    @patch("fairseq2.models.hg.factory._get_model_path")
    @patch("fairseq2.models.hg.factory._has_transformers", True)
    @patch("fairseq2.models.hg.factory.AutoConfig")
    @patch("fairseq2.models.hg.factory.AutoModelForCausalLM")
    def test_load_causal_lm_uses_auto_causal_lm(
        self,
        mock_auto_causal: MagicMock,
        mock_auto_config: MagicMock,
        mock_get_path: MagicMock,
    ) -> None:
        """load_causal_lm should use AutoModelForCausalLM.from_pretrained."""
        mock_get_path.return_value = Path("/fake/gpt2")
        mock_hf_config = MagicMock()
        mock_hf_config.__class__.__name__ = "GPT2Config"
        mock_auto_config.from_pretrained.return_value = mock_hf_config

        mock_model = _make_mock_hf_causal_model()
        mock_auto_causal.from_pretrained.return_value = mock_model

        result = load_causal_lm("gpt2")

        mock_auto_config.from_pretrained.assert_called_once_with("gpt2")
        mock_auto_causal.from_pretrained.assert_called_once()
        # Causal LM models should be wrapped in HgCausalLMAdapter
        assert isinstance(result, HgCausalLMAdapter)

    @patch("fairseq2.models.hg.factory._get_model_path")
    @patch("fairseq2.models.hg.factory._has_transformers", True)
    @patch("fairseq2.models.hg.factory.AutoConfig")
    @patch("fairseq2.models.hg.factory.AutoModelForCausalLM")
    def test_load_causal_lm_passes_kwargs(
        self,
        mock_auto_causal: MagicMock,
        mock_auto_config: MagicMock,
        mock_get_path: MagicMock,
    ) -> None:
        """Extra kwargs should be forwarded to from_pretrained."""
        mock_get_path.return_value = Path("/fake/model")
        mock_hf_config = MagicMock()
        mock_hf_config.__class__.__name__ = "LlamaConfig"
        mock_auto_config.from_pretrained.return_value = mock_hf_config

        mock_model = _make_mock_hf_causal_model()
        mock_auto_causal.from_pretrained.return_value = mock_model

        load_causal_lm("meta-llama/Llama-2-7b", dtype=torch.float16)

        call_kwargs = mock_auto_causal.from_pretrained.call_args[1]
        assert call_kwargs.get("dtype") == torch.float16


class TestLoadSeq2SeqLmAPI:
    """Test that load_seq2seq_lm correctly delegates to AutoModelForSeq2SeqLM."""

    @patch("fairseq2.models.hg.factory._get_model_path")
    @patch("fairseq2.models.hg.factory._has_transformers", True)
    @patch("fairseq2.models.hg.factory.AutoConfig")
    @patch("fairseq2.models.hg.factory.AutoModelForSeq2SeqLM")
    def test_load_seq2seq_uses_auto_seq2seq(
        self,
        mock_auto_seq2seq: MagicMock,
        mock_auto_config: MagicMock,
        mock_get_path: MagicMock,
    ) -> None:
        """load_seq2seq_lm should use AutoModelForSeq2SeqLM.from_pretrained."""
        mock_get_path.return_value = Path("/fake/t5")
        mock_hf_config = MagicMock()
        mock_hf_config.__class__.__name__ = "T5Config"
        mock_auto_config.from_pretrained.return_value = mock_hf_config

        mock_model = _make_mock_hf_seq2seq_model()
        mock_auto_seq2seq.from_pretrained.return_value = mock_model

        result = load_seq2seq_lm("google/t5-efficient-tiny")

        mock_auto_config.from_pretrained.assert_called_once_with(
            "google/t5-efficient-tiny"
        )
        mock_auto_seq2seq.from_pretrained.assert_called_once()
        # Seq2seq models should NOT be wrapped in HgCausalLMAdapter
        assert not isinstance(result, HgCausalLMAdapter)


class TestLoadHgModelSimpleAPI:
    """Test load_hg_model_simple with different model_type values."""

    @patch("fairseq2.models.hg.factory._get_model_path")
    @patch("fairseq2.models.hg.factory._has_transformers", True)
    @patch("fairseq2.models.hg.factory.AutoConfig")
    @patch("fairseq2.models.hg.factory.AutoModel")
    def test_auto_model_type_uses_auto_model(
        self,
        mock_auto_model: MagicMock,
        mock_auto_config: MagicMock,
        mock_get_path: MagicMock,
    ) -> None:
        """model_type='auto' should use AutoModel."""
        mock_get_path.return_value = Path("/fake/bert")
        mock_hf_config = MagicMock()
        mock_hf_config.__class__.__name__ = "BertConfig"
        mock_auto_config.from_pretrained.return_value = mock_hf_config

        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        result = load_hg_model_simple("bert-base-uncased", model_type="auto")

        mock_auto_model.from_pretrained.assert_called_once()

    def test_config_construction_causal_lm(self) -> None:
        """load_causal_lm should produce config with model_type='causal_lm'."""
        # Verify the config is constructed correctly without calling HF
        config = HuggingFaceModelConfig(hf_name="gpt2", model_type="causal_lm")
        assert config.model_type == "causal_lm"
        assert config.hf_name == "gpt2"

    def test_config_construction_seq2seq(self) -> None:
        """load_seq2seq_lm should produce config with model_type='seq2seq_lm'."""
        config = HuggingFaceModelConfig(
            hf_name="google/t5-efficient-tiny", model_type="seq2seq_lm"
        )
        assert config.model_type == "seq2seq_lm"


class TestAutoModelClassSelection:
    """Test that the correct Auto class is selected based on model_type."""

    @patch("fairseq2.models.hg.factory._has_transformers", True)
    @patch("fairseq2.models.hg.factory.AutoModelForCausalLM")
    def test_causal_lm_selects_auto_causal(self, mock_cls: MagicMock) -> None:
        config = HuggingFaceModelConfig(hf_name="gpt2", model_type="causal_lm")
        result = _get_auto_model_class(config, MagicMock())
        assert result is mock_cls

    @patch("fairseq2.models.hg.factory._has_transformers", True)
    @patch("fairseq2.models.hg.factory.AutoModelForSeq2SeqLM")
    def test_seq2seq_selects_auto_seq2seq(self, mock_cls: MagicMock) -> None:
        config = HuggingFaceModelConfig(hf_name="t5", model_type="seq2seq_lm")
        result = _get_auto_model_class(config, MagicMock())
        assert result is mock_cls

    @patch("fairseq2.models.hg.factory._has_transformers", True)
    @patch("fairseq2.models.hg.factory.AutoModel")
    def test_auto_selects_auto_model(self, mock_cls: MagicMock) -> None:
        config = HuggingFaceModelConfig(hf_name="bert", model_type="auto")
        result = _get_auto_model_class(config, MagicMock())
        assert result is mock_cls

    @patch("fairseq2.models.hg.factory._has_transformers", True)
    @patch("fairseq2.models.hg.factory.AutoModelForSeq2SeqLM")
    def test_encoder_decoder_auto_detects_seq2seq(self, mock_cls: MagicMock) -> None:
        config = HuggingFaceModelConfig(hf_name="bart", model_type="unknown")
        hf_config = MagicMock()
        hf_config.is_encoder_decoder = True
        result = _get_auto_model_class(config, hf_config)
        assert result is mock_cls

    @patch("fairseq2.models.hg.factory._has_transformers", True)
    @patch("fairseq2.models.hg.factory.AutoModelForCausalLM")
    def test_decoder_only_auto_detects_causal(self, mock_cls: MagicMock) -> None:
        config = HuggingFaceModelConfig(hf_name="gpt2", model_type="unknown")
        hf_config = MagicMock()
        hf_config.is_encoder_decoder = False
        result = _get_auto_model_class(config, hf_config)
        assert result is mock_cls


class TestLoadTokenizerAPI:
    """Test that load_hg_tokenizer_simple delegates to AutoTokenizer."""

    @patch("fairseq2.data.tokenizers.hg.AutoTokenizer")
    def test_load_tokenizer_calls_auto_tokenizer(
        self, mock_auto_tok: MagicMock
    ) -> None:
        """load_hg_tokenizer_simple should call AutoTokenizer.from_pretrained."""
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50257
        mock_tok.unk_token_id = 0
        mock_tok.bos_token_id = 1
        mock_tok.eos_token_id = 2
        mock_tok.pad_token_id = None
        mock_auto_tok.from_pretrained.return_value = mock_tok

        tokenizer = load_hg_tokenizer_simple("gpt2")

        mock_auto_tok.from_pretrained.assert_called_once()
        call_args = mock_auto_tok.from_pretrained.call_args
        assert str(call_args[0][0]) == "gpt2"

    @patch("fairseq2.data.tokenizers.hg.AutoTokenizer")
    def test_tokenizer_vocab_info_populated(
        self, mock_auto_tok: MagicMock
    ) -> None:
        """Tokenizer vocab_info should reflect the HF tokenizer's properties."""
        mock_tok = MagicMock()
        mock_tok.vocab_size = 32000
        mock_tok.unk_token_id = 0
        mock_tok.bos_token_id = 1
        mock_tok.eos_token_id = 2
        mock_tok.pad_token_id = None
        mock_auto_tok.from_pretrained.return_value = mock_tok

        tokenizer = load_hg_tokenizer_simple("meta-llama/Llama-2-7b")

        vocab_info = tokenizer.vocab_info
        assert vocab_info.size == 32000
        assert vocab_info.bos_idx == 1
        assert vocab_info.eos_idx == 2

    @patch("fairseq2.data.tokenizers.hg.AutoTokenizer")
    def test_tokenizer_custom_special_tokens(
        self, mock_auto_tok: MagicMock
    ) -> None:
        """Custom special tokens should be resolved via convert_tokens_to_ids."""
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50257
        mock_tok.unk_token_id = None
        mock_tok.bos_token_id = None
        mock_tok.eos_token_id = None
        mock_tok.pad_token_id = None
        mock_tok.convert_tokens_to_ids.side_effect = lambda t: {
            "<pad>": 50256,
            "<eos>": 50255,
        }.get(t, 0)
        mock_auto_tok.from_pretrained.return_value = mock_tok

        tokenizer = load_hg_tokenizer_simple(
            "gpt2", pad_token="<pad>", eos_token="<eos>"
        )

        vocab_info = tokenizer.vocab_info
        assert vocab_info.pad_idx == 50256
        assert vocab_info.eos_idx == 50255

    @patch("fairseq2.data.tokenizers.hg.AutoTokenizer")
    def test_vocab_size_uses_base_vocab_not_len(
        self, mock_auto_tok: MagicMock
    ) -> None:
        """vocab_info.size should use tok.vocab_size (base vocab), not len(tok) (base + added tokens).

        HF tokenizers distinguish between base vocab (vocab_size property) and
        total vocab including added tokens (len()). Model embeddings are sized to
        base vocab, so we must use vocab_size to avoid spurious mismatches.
        """
        mock_tok = MagicMock()
        # Simulate an instruct model: base vocab is 32000, but added tokens make len() = 32010
        mock_tok.vocab_size = 32000
        mock_tok.__len__ = MagicMock(return_value=32010)
        mock_tok.unk_token_id = 0
        mock_tok.bos_token_id = 1
        mock_tok.eos_token_id = 2
        mock_tok.pad_token_id = None
        mock_auto_tok.from_pretrained.return_value = mock_tok

        tokenizer = load_hg_tokenizer_simple("meta-llama/Llama-2-7b-chat-hf")

        # Must use base vocab (32000), NOT len(tok) (32010)
        assert tokenizer.vocab_info.size == 32000


class TestCausalLMAdapterWiring:
    """Test that the factory correctly wraps causal LM models in HgCausalLMAdapter."""

    @patch("fairseq2.models.hg.factory._get_model_path")
    @patch("fairseq2.models.hg.factory._has_transformers", True)
    @patch("fairseq2.models.hg.factory.AutoConfig")
    @patch("fairseq2.models.hg.factory.AutoModelForCausalLM")
    def test_causal_lm_wrapped_in_adapter(
        self,
        mock_auto_causal: MagicMock,
        mock_auto_config: MagicMock,
        mock_get_path: MagicMock,
    ) -> None:
        """Models loaded with model_type='causal_lm' should be HgCausalLMAdapter."""
        mock_get_path.return_value = Path("/fake/gpt2")
        mock_hf_config = MagicMock()
        mock_hf_config.__class__.__name__ = "GPT2Config"
        mock_auto_config.from_pretrained.return_value = mock_hf_config

        mock_model = _make_mock_hf_causal_model()
        mock_auto_causal.from_pretrained.return_value = mock_model

        config = HuggingFaceModelConfig(hf_name="gpt2", model_type="causal_lm")
        result = create_hg_model(config)

        assert isinstance(result, HgCausalLMAdapter)
        assert result.decoder_frontend.embed.num_embeddings == 50257

    @patch("fairseq2.models.hg.factory._get_model_path")
    @patch("fairseq2.models.hg.factory._has_transformers", True)
    @patch("fairseq2.models.hg.factory.AutoConfig")
    @patch("fairseq2.models.hg.factory.AutoModelForSeq2SeqLM")
    def test_seq2seq_not_wrapped(
        self,
        mock_auto_seq2seq: MagicMock,
        mock_auto_config: MagicMock,
        mock_get_path: MagicMock,
    ) -> None:
        """Models loaded with model_type='seq2seq_lm' should NOT be wrapped."""
        mock_get_path.return_value = Path("/fake/t5")
        mock_hf_config = MagicMock()
        mock_hf_config.__class__.__name__ = "T5Config"
        mock_auto_config.from_pretrained.return_value = mock_hf_config

        mock_model = _make_mock_hf_seq2seq_model()
        mock_auto_seq2seq.from_pretrained.return_value = mock_model

        config = HuggingFaceModelConfig(
            hf_name="google/t5-efficient-tiny", model_type="seq2seq_lm"
        )
        result = create_hg_model(config)

        assert not isinstance(result, HgCausalLMAdapter)
