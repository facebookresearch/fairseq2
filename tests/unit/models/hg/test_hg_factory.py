# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import torch

from fairseq2.gang import get_current_gangs, get_default_gangs
from fairseq2.models.hg_qwen_omni.config import HuggingFaceModelConfig
from fairseq2.models.hg_qwen_omni.factory import (
    HgFactory,
    HuggingFaceModelError,
    _get_auto_model_class,
    _get_model_info,
    _get_model_path,
    _import_class_from_transformers,
    _prepare_load_kwargs,
    _set_device_kwargs,
    _set_dtype_kwargs,
    create_hg_model,
    register_hg_model_class,
)


class TestRegisterHgModelClass:
    """Test the register_hg_model_class function."""

    @patch("fairseq2.models.hg_qwen_omni.factory._has_transformers", True)
    def test_register_model_class_with_strings(self) -> None:
        """Test registering model class with string names."""
        register_hg_model_class(
            config_class_name="TestConfig",
            model_class="TestModel",
            tokenizer_class="TestTokenizer",
            processor_class="TestProcessor",
        )

        # Import the registry to check
        from fairseq2.models.hg_qwen_omni.factory import _USER_REGISTRY

        assert "TestConfig" in _USER_REGISTRY
        entry = _USER_REGISTRY["TestConfig"]
        assert entry["model_class"] == "TestModel"
        assert entry["tokenizer_class"] == "TestTokenizer"
        assert entry["processor_class"] == "TestProcessor"

    @patch("fairseq2.models.hg_qwen_omni.factory._has_transformers", True)
    def test_register_model_class_with_types(self) -> None:
        """Test registering model class with actual types."""
        # Use string names instead of actual classes to avoid type issues
        register_hg_model_class(
            config_class_name="TestConfig2",
            model_class="MockModel",
            tokenizer_class="MockTokenizer",
        )

        from fairseq2.models.hg_qwen_omni.factory import _USER_REGISTRY

        assert "TestConfig2" in _USER_REGISTRY
        entry = _USER_REGISTRY["TestConfig2"]
        assert entry["model_class"] == "MockModel"
        assert entry["tokenizer_class"] == "MockTokenizer"

    @patch("fairseq2.models.hg_qwen_omni.factory._has_transformers", True)
    def test_register_model_class_minimal(self) -> None:
        """Test registering model class with minimal params."""
        register_hg_model_class(
            config_class_name="TestConfig3",
            model_class="MinimalModel",
        )

        from fairseq2.models.hg_qwen_omni.factory import _USER_REGISTRY

        assert "TestConfig3" in _USER_REGISTRY
        entry = _USER_REGISTRY["TestConfig3"]
        assert entry["model_class"] == "MinimalModel"
        assert "tokenizer_class" not in entry
        assert "processor_class" not in entry

    @patch("fairseq2.models.hg_qwen_omni.factory._has_transformers", False)
    def test_register_model_class_no_transformers(self) -> None:
        """Test error when transformers is not available."""
        with pytest.raises(Exception) as exc_info:
            register_hg_model_class("TestConfig", "TestModel")

        assert "transformers" in str(exc_info.value).lower()


class TestCreateHgModel:
    """Test the create_hg_model function."""

    @patch("fairseq2.models.hg_qwen_omni.factory.HgFactory")
    def test_create_hg_model(self, mock_factory_class: MagicMock) -> None:
        """Test create_hg_model delegates to HgFactory."""
        mock_factory = MagicMock()
        mock_model = MagicMock()
        mock_factory.create_model.return_value = mock_model
        mock_factory_class.return_value = mock_factory

        config = HuggingFaceModelConfig(hf_name="gpt2")
        result = create_hg_model(config)

        mock_factory_class.assert_called_once_with(config)
        mock_factory.create_model.assert_called_once()
        assert result is mock_model


class TestHgFactory:
    """Test the HgFactory class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.config = HuggingFaceModelConfig(hf_name="gpt2")
        self.root_gangs_object = get_default_gangs()
        with self.root_gangs_object:
            self.gangs = get_current_gangs()
        self.gangs = get_current_gangs()
        self.factory = HgFactory(self.config)

    @patch("fairseq2.models.hg_qwen_omni.factory._has_transformers", False)
    def test_create_model_no_transformers(self) -> None:
        """Test error when transformers is not available."""
        with pytest.raises(Exception) as exc_info:
            self.factory.create_model()

        assert "transformers" in str(exc_info.value).lower()

    @patch("fairseq2.models.hg_qwen_omni.factory._has_transformers", True)
    @patch("fairseq2.models.hg_qwen_omni.factory.AutoConfig")
    @patch("fairseq2.models.hg_qwen_omni.factory._get_model_info")
    @patch("fairseq2.models.hg_qwen_omni.factory._load_auto_model")
    def test_create_model_auto_model(
        self,
        mock_load_auto: MagicMock,
        mock_get_info: MagicMock,
        mock_auto_config: MagicMock,
    ) -> None:
        """Test creating model using auto model."""
        mock_hf_config = MagicMock()
        mock_hf_config.__class__.__name__ = "GPT2Config"
        mock_auto_config.from_pretrained.return_value = mock_hf_config
        mock_get_info.return_value = None
        mock_model = MagicMock()
        mock_load_auto.return_value = mock_model

        result = self.factory.create_model()

        mock_auto_config.from_pretrained.assert_called_once_with("gpt2")
        mock_get_info.assert_called_once_with("GPT2Config", self.config)
        mock_load_auto.assert_called_once_with("gpt2", self.config, mock_hf_config)
        assert result is mock_model

    @patch("fairseq2.models.hg_qwen_omni.factory._has_transformers", True)
    @patch("fairseq2.models.hg_qwen_omni.factory.AutoConfig")
    @patch("fairseq2.models.hg_qwen_omni.factory._get_model_info")
    @patch("fairseq2.models.hg_qwen_omni.factory._load_special_model")
    def test_create_model_special_model(
        self,
        mock_load_special: MagicMock,
        mock_get_info: MagicMock,
        mock_auto_config: MagicMock,
    ) -> None:
        """Test creating model using special model."""
        mock_hf_config = MagicMock()
        mock_hf_config.__class__.__name__ = "Qwen2_5OmniConfig"
        mock_auto_config.from_pretrained.return_value = mock_hf_config
        model_class = "Qwen2_5OmniForConditionalGeneration"
        mock_model_info = {"model_class": model_class}
        mock_get_info.return_value = mock_model_info
        mock_model = MagicMock()
        mock_load_special.return_value = mock_model

        result = self.factory.create_model()

        mock_load_special.assert_called_once_with(
            "gpt2", self.config, mock_model_info, None
        )
        assert result is mock_model

    def test_create_model_special_model_with_gangs(self) -> None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            import torch.distributed as dist

            from fairseq2.assets import get_asset_store
            from fairseq2.device import get_default_device
            from fairseq2.gang import (
                ProcessGroupGang,
                create_parallel_gangs,
            )
            from fairseq2.models.hg_qwen_omni import get_hg_model_hub

            world_size = torch.cuda.device_count()
            device = get_default_device()
            root_gang = ProcessGroupGang.create_default_process_group(device)
            gangs = create_parallel_gangs(root_gang, tp_size=world_size//2)

            card = get_asset_store().retrieve_card("hg_qwen25_omni_3b")
            dist.barrier()
            get_hg_model_hub().load_model(card, gangs=gangs)
            dist.barrier()
            gangs.close()

    @patch("fairseq2.models.hg_qwen_omni.factory._has_transformers", True)
    @patch("fairseq2.models.hg_qwen_omni.factory.AutoConfig")
    def test_create_model_not_found_error(self, mock_auto_config: MagicMock) -> None:
        """Test error handling for model not found."""
        error = Exception("404 Not Found")
        mock_auto_config.from_pretrained.side_effect = error

        with pytest.raises(HuggingFaceModelError) as exc_info:
            self.factory.create_model()

        assert exc_info.value.model_name == "gpt2"
        assert "not found" in str(exc_info.value).lower()

    @patch("fairseq2.models.hg_qwen_omni.factory._has_transformers", True)
    @patch("fairseq2.models.hg_qwen_omni.factory.AutoConfig")
    def test_create_model_generic_error(self, mock_auto_config: MagicMock) -> None:
        """Test error handling for generic errors."""
        error = Exception("Generic error")
        mock_auto_config.from_pretrained.side_effect = error

        with pytest.raises(HuggingFaceModelError) as exc_info:
            self.factory.create_model()

        assert exc_info.value.model_name == "gpt2"
        assert "Failed to load model" in str(exc_info.value)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_model_info_user_registry(self) -> None:
        """Test _get_model_info with user registry."""
        from fairseq2.models.hg_qwen_omni.factory import _USER_REGISTRY

        # Add test entry
        _USER_REGISTRY["TestConfig"] = {"model_class": "TestModel"}

        config = HuggingFaceModelConfig(hf_name="test")
        result = _get_model_info("TestConfig", config)

        assert result == {"model_class": "TestModel"}

        # Clean up
        del _USER_REGISTRY["TestConfig"]

    def test_get_model_info_builtin_registry(self) -> None:
        """Test _get_model_info with built-in registry."""
        config = HuggingFaceModelConfig(hf_name="test")
        result = _get_model_info("Qwen2_5OmniConfig", config)

        assert result is not None
        assert result["model_class"] == "Qwen2_5OmniForConditionalGeneration"

    def test_get_model_info_custom_config(self) -> None:
        """Test _get_model_info with custom config."""
        config = HuggingFaceModelConfig(
            hf_name="test",
            model_type="custom",
            custom_model_class="CustomModel",
            custom_processor_class="CustomProcessor",
        )
        result = _get_model_info("UnknownConfig", config)

        assert result == {
            "model_class": "CustomModel",
            "processor_class": "CustomProcessor",
        }

    def test_get_model_info_no_match(self) -> None:
        """Test _get_model_info with no match."""
        config = HuggingFaceModelConfig(hf_name="test")
        result = _get_model_info("UnknownConfig", config)

        assert result is None

    @patch("fairseq2.models.hg_qwen_omni.factory._has_transformers", True)
    @patch("fairseq2.models.hg_qwen_omni.factory.AutoModelForCausalLM")
    def test_get_auto_model_class_causal_lm(self, mock_auto_causal: MagicMock) -> None:
        """Test _get_auto_model_class for causal LM."""
        config = HuggingFaceModelConfig(hf_name="test", model_type="causal_lm")
        hf_config = MagicMock()
        result = _get_auto_model_class(config, hf_config)

        assert result is mock_auto_causal

    @patch("fairseq2.models.hg_qwen_omni.factory._has_transformers", True)
    @patch("fairseq2.models.hg_qwen_omni.factory.AutoModelForSeq2SeqLM")
    def test_get_auto_model_class_seq2seq_lm(
        self, mock_auto_seq2seq: MagicMock
    ) -> None:
        """Test _get_auto_model_class for seq2seq LM."""
        config = HuggingFaceModelConfig(hf_name="test", model_type="seq2seq_lm")
        hf_config = MagicMock()
        result = _get_auto_model_class(config, hf_config)

        assert result is mock_auto_seq2seq

    @patch("fairseq2.models.hg_qwen_omni.factory._has_transformers", True)
    @patch("fairseq2.models.hg_qwen_omni.factory.AutoModel")
    def test_get_auto_model_class_auto(self, mock_auto: MagicMock) -> None:
        """Test _get_auto_model_class for auto."""
        config = HuggingFaceModelConfig(hf_name="test", model_type="auto")
        hf_config = MagicMock()
        result = _get_auto_model_class(config, hf_config)

        assert result is mock_auto

    @patch("fairseq2.models.hg_qwen_omni.factory._has_transformers", True)
    @patch("fairseq2.models.hg_qwen_omni.factory.AutoModelForSeq2SeqLM")
    def test_get_auto_model_class_encoder_decoder(
        self, mock_auto_seq2seq: MagicMock
    ) -> None:
        """Test _get_auto_model_class for encoder-decoder models."""
        config = HuggingFaceModelConfig(hf_name="test", model_type="unknown")
        hf_config = MagicMock()
        hf_config.is_encoder_decoder = True
        result = _get_auto_model_class(config, hf_config)

        assert result is mock_auto_seq2seq

    @patch("fairseq2.models.hg_qwen_omni.factory._has_transformers", True)
    @patch("fairseq2.models.hg_qwen_omni.factory.AutoModelForCausalLM")
    def test_get_auto_model_class_decoder_only(
        self, mock_auto_causal: MagicMock
    ) -> None:
        """Test _get_auto_model_class for decoder-only models."""
        config = HuggingFaceModelConfig(hf_name="test", model_type="unknown")
        hf_config = MagicMock()
        hf_config.is_encoder_decoder = False
        result = _get_auto_model_class(config, hf_config)

        assert result is mock_auto_causal


class TestPrepareLoadKwargs:
    """Test the _prepare_load_kwargs function."""

    @patch("fairseq2.models.hg_qwen_omni.factory._get_model_path")
    @patch("fairseq2.models.hg_qwen_omni.factory._set_dtype_kwargs")
    @patch("fairseq2.models.hg_qwen_omni.factory._set_device_kwargs")
    def test_prepare_load_kwargs_basic(
        self,
        mock_set_device: MagicMock,
        mock_set_dtype: MagicMock,
        mock_get_path: MagicMock,
    ) -> None:
        """Test basic kwargs preparation."""
        mock_path = Path("/test/path")
        mock_get_path.return_value = mock_path

        config = HuggingFaceModelConfig(hf_name="gpt2")
        result = _prepare_load_kwargs(config)

        expected_base = {
            "pretrained_model_name_or_path": mock_path,
            "use_safetensors": True,
        }

        # Check base kwargs
        for key, value in expected_base.items():
            assert result[key] == value

        # Check that helper functions were called
        mock_get_path.assert_called_once_with(config)
        mock_set_dtype.assert_called_once_with(config, result)
        mock_set_device.assert_called_once_with(config, result)

    @patch("fairseq2.models.hg_qwen_omni.factory._get_model_path")
    @patch("fairseq2.models.hg_qwen_omni.factory._set_dtype_kwargs")
    @patch("fairseq2.models.hg_qwen_omni.factory._set_device_kwargs")
    def test_prepare_load_kwargs_with_trust_remote_code(
        self,
        mock_set_device: MagicMock,
        mock_set_dtype: MagicMock,
        mock_get_path: MagicMock,
    ) -> None:
        """Test kwargs with trust_remote_code."""
        mock_get_path.return_value = Path("/test/path")

        config = HuggingFaceModelConfig(hf_name="gpt2", trust_remote_code=True)
        result = _prepare_load_kwargs(config)

        assert result["trust_remote_code"] is True

    @patch("fairseq2.models.hg_qwen_omni.factory._get_model_path")
    @patch("fairseq2.models.hg_qwen_omni.factory._set_dtype_kwargs")
    @patch("fairseq2.models.hg_qwen_omni.factory._set_device_kwargs")
    def test_prepare_load_kwargs_with_additional_kwargs(
        self,
        mock_set_device: MagicMock,
        mock_set_dtype: MagicMock,
        mock_get_path: MagicMock,
    ) -> None:
        """Test kwargs with additional load_kwargs."""
        mock_get_path.return_value = Path("/test/path")

        load_kwargs: Dict[str, Any] = {"temperature": 0.7, "max_length": 100}
        config = HuggingFaceModelConfig(hf_name="gpt2", load_kwargs=load_kwargs)
        result = _prepare_load_kwargs(config)

        assert result["temperature"] == 0.7
        assert result["max_length"] == 100


class TestSetDtypeKwargs:
    """Test the _set_dtype_kwargs function."""

    def test_set_dtype_kwargs_auto(self) -> None:
        """Test dtype auto."""
        config = HuggingFaceModelConfig(hf_name="test", dtype="auto")
        kwargs: Dict[str, Any] = {}
        _set_dtype_kwargs(config, kwargs)

        assert kwargs["dtype"] == "auto"

    def test_set_dtype_kwargs_float16(self) -> None:
        """Test dtype float16."""
        config = HuggingFaceModelConfig(hf_name="test", dtype="float16")
        kwargs: Dict[str, Any] = {}
        _set_dtype_kwargs(config, kwargs)

        assert kwargs["dtype"] == torch.float16

    def test_set_dtype_kwargs_bfloat16(self) -> None:
        """Test dtype bfloat16."""
        config = HuggingFaceModelConfig(hf_name="test", dtype="bfloat16")
        kwargs: Dict[str, Any] = {}
        _set_dtype_kwargs(config, kwargs)

        assert kwargs["dtype"] == torch.bfloat16


class TestSetDeviceKwargs:
    """Test the _set_device_kwargs function."""

    def test_set_device_kwargs_not_auto(self) -> None:
        """Test device kwargs when not auto."""
        config = HuggingFaceModelConfig(hf_name="test", device="cpu")
        kwargs: Dict[str, Any] = {}
        _set_device_kwargs(config, kwargs)

        assert "device_map" not in kwargs

    def test_set_device_kwargs_auto_with_accelerate(self) -> None:
        """Test device kwargs auto with accelerate available."""
        # Mock successful accelerate import
        with patch("builtins.__import__") as mock_import:

            def side_effect(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "accelerate":
                    return MagicMock()  # Mock accelerate module
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            config = HuggingFaceModelConfig(hf_name="test", device="auto")
            kwargs: Dict[str, Any] = {}
            _set_device_kwargs(config, kwargs)

            assert kwargs["device_map"] == "auto"

    def test_set_device_kwargs_auto_no_accelerate(self) -> None:
        """Test device kwargs auto without accelerate."""
        # Mock failed accelerate import
        with patch("builtins.__import__") as mock_import:

            def side_effect(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "accelerate":
                    raise ImportError("No module named 'accelerate'")
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            config = HuggingFaceModelConfig(hf_name="test", device="auto")
            kwargs: Dict[str, Any] = {}
            _set_device_kwargs(config, kwargs)

            assert "device_map" not in kwargs


class TestGetModelPath:
    """Test the _get_model_path function."""

    @patch("fairseq2.models.hg_qwen_omni.factory.HuggingFaceHub")
    @patch("fairseq2.models.hg_qwen_omni.factory.Uri.maybe_parse")
    def test_get_model_path(
        self, mock_uri_parse: MagicMock, mock_hub_class: MagicMock
    ) -> None:
        """Test getting model path."""
        mock_hub = MagicMock()
        mock_path = Path("/test/model/path")
        mock_hub.download_model.return_value = mock_path
        mock_hub_class.return_value = mock_hub
        mock_uri = MagicMock()
        mock_uri.scheme = "hg"
        mock_uri_parse.return_value = mock_uri

        config = HuggingFaceModelConfig(hf_name="gpt2")
        result = _get_model_path(config)

        mock_uri_parse.assert_called_once_with("hg://gpt2")
        mock_hub.download_model.assert_called_once_with(mock_uri, "gpt2")
        assert result == mock_path


class TestImportClassFromTransformers:
    """Test the _import_class_from_transformers function."""

    @patch("fairseq2.models.hg_qwen_omni.factory.importlib.import_module")
    def test_import_class_success(self, mock_import: MagicMock) -> None:
        """Test successful class import."""
        mock_module = MagicMock()
        mock_class = MagicMock()
        mock_module.TestClass = mock_class
        mock_import.return_value = mock_module

        result = _import_class_from_transformers("TestClass")

        mock_import.assert_called_once_with("transformers")
        assert result is mock_class

    @patch("fairseq2.models.hg_qwen_omni.factory.importlib.import_module")
    def test_import_class_not_found(self, mock_import: MagicMock) -> None:
        """Test class not found error."""
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        del mock_module.TestClass  # Ensure attribute doesn't exist

        with pytest.raises(ImportError) as exc_info:
            _import_class_from_transformers("TestClass")

        assert "TestClass" in str(exc_info.value)
        assert "not found" in str(exc_info.value)


if __name__ == "__main__":
    """
    Hardware dependent test:
    If more than one GPU is present, model sharding with tp
    can be tested.

    Example run:
        `torchrun --nproc-per-node 8 test_hg_factory.py`
    """

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        mg_factory = TestHgFactory()
        mg_factory.setup_method()
        mg_factory.test_create_model_special_model_with_gangs()
