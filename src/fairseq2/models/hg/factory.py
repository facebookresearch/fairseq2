# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Factory functions for creating HuggingFace models.

This module provides the core factory functionality for creating HuggingFace
models within fairseq2. It handles model loading, device placement, custom
model classes, and error handling.

Key Components:
    - HgFactory: Main factory class for model creation
    - HuggingFaceModelError: Custom exception for model loading errors
    - Model registries for special cases and custom models
    - Utility functions for device and dtype handling

The factory supports:
    - Auto model detection and loading
    - Custom model classes for unsupported architectures
    - Device placement with accelerate integration
    - Processor and tokenizer handling
    - Comprehensive error handling and logging
"""

from __future__ import annotations

import importlib
import urllib
from pathlib import Path
from typing import Any, Dict, Type

from tqdm.auto import tqdm

from fairseq2.assets import HuggingFaceHub
from fairseq2.error import NotSupportedError, OperationalError
from fairseq2.logging import log
from fairseq2.models.hg.config import HuggingFaceModelConfig
from fairseq2.utils.uri import Uri
from fairseq2.gang import Gangs, Gang, GangError, maybe_get_current_gangs

try:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoProcessor,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
except ImportError:
    _has_transformers = False
else:
    _has_transformers = True


# Registry for special cases where AutoModel doesn't work
_HF_SPECIAL_MODELS: Dict[str, Dict[str, str]] = {
    "Qwen2_5OmniConfig": {
        "model_class": "Qwen2_5OmniForConditionalGeneration",
        "processor_class": "Qwen2_5OmniProcessor",
    },
    # Add more special cases as needed
}
"""Built-in registry for models that require special handling."""

# User-defined registry for custom model mappings
_USER_REGISTRY: Dict[str, Dict[str, Any]] = {}
"""User-defined registry for custom model class mappings."""


class HuggingFaceModelError(Exception):
    """
    Exception raised when HuggingFace model loading fails.

    This exception provides detailed information about model loading failures,
    including the model name and specific error details.

    Attributes:
        model_name: The name of the model that failed to load
        message: Detailed error message
    """

    def __init__(self, model_name: str, message: str) -> None:
        super().__init__(message)
        self.model_name = model_name


def register_hg_model_class(
    config_class_name: str,
    model_class: Type[PreTrainedModel] | str,
    tokenizer_class: Type[PreTrainedTokenizer] | str | None = None,
    processor_class: str | None = None,
) -> None:
    """
    Register a custom model class for models not supported by Auto classes.

    This function allows registration of custom model classes that cannot be
    loaded automatically by HuggingFace's Auto classes. This is useful for
    new or experimental model architectures.

    :param config_class_name: The name of the config class (e.g., 'Qwen2_5OmniConfig')
    :param model_class: The model class or its string name
    :param tokenizer_class: The tokenizer class or its string name (optional)
    :param processor_class: The processor class or its string name (optional)
    :raises: OperationalError: If transformers library is not available

    Example:
        Register a custom model::

            register_hg_model_class(
                "Qwen2_5OmniConfig",
                "Qwen2_5OmniForConditionalGeneration",
                processor_class="Qwen2_5OmniProcessor",
            )
    """
    if not _has_transformers:
        raise OperationalError(
            "HuggingFace Transformers is not available. Install it with "
            "`pip install transformers`."
        )

    entry = {}

    if isinstance(model_class, str):
        entry["model_class"] = model_class
    else:
        entry["model_class"] = model_class.__name__

    if tokenizer_class is not None:
        if isinstance(tokenizer_class, str):
            entry["tokenizer_class"] = tokenizer_class
        else:
            entry["tokenizer_class"] = tokenizer_class.__name__

    if processor_class is not None:
        if isinstance(processor_class, str):
            entry["processor_class"] = processor_class
        else:
            entry["processor_class"] = processor_class.__name__

    _USER_REGISTRY[config_class_name] = entry

    log.info(f"Registered custom HF model mapping: {config_class_name} -> {entry}")


def create_hg_model(
    config: HuggingFaceModelConfig,
) -> Any:
    """
    Create a HuggingFace model from configuration.

    This factory loads models directly from HuggingFace Hub with transformers.

    :param config: HuggingFace model configuration
    :returns: HuggingFace PreTrainedModel
    :raises: OperationalError: If transformers library is not available
    :raises: HuggingFaceModelError: If model loading fails
    :raises: NotSupportedError: If transformers library is not available
    """
    gangs = maybe_get_current_gangs()
    
    return HgFactory(config, gangs).create_model()


class HgFactory:
    """
    Factory for creating HuggingFace models.

    This class handles the logic of loading HuggingFace models,
    including device placement, dtype conversion, custom classes, and
    error handling.

    Args:
        config: The HuggingFace model configuration
    """

    def __init__(self, config: HuggingFaceModelConfig, gangs: Gangs) -> None:
        """Initialize the factory with configuration."""
        self._config = config
        self._gangs = gangs

    def create_model(self) -> Any:
        """Create the model according to the configuration.

        Returns:
            PreTrainedModel: The loaded model

        Raises:
            NotSupportedError: If transformers is not available
            HuggingFaceModelError: If model loading fails
        """
        config = self._config
        gangs = self._gangs

        log.info(f"Creating HuggingFace model: {config.hf_name}")

        if not _has_transformers:
            raise NotSupportedError(
                "HuggingFace Transformers is not available. Install it with "
                "`pip install transformers`."
            )

        name = config.hf_name

        try:
            # Get the model configuration first
            hf_config = AutoConfig.from_pretrained(name)
            config_class_name = hf_config.__class__.__name__

            log.info(
                f"Loading HuggingFace model '{name}' with config "
                f"{config_class_name}"
            )

            # Check if this is a special case model
            model_info = _get_model_info(config_class_name, config)

            if model_info:
                return _load_special_model(name, config, model_info, gangs)
            else:
                return _load_auto_model(name, config, hf_config, gangs)

        except Exception as ex:
            if "not found" in str(ex).lower() or "404" in str(ex):
                raise HuggingFaceModelError(
                    name,
                    f"Model '{name}' not found. Please ensure the model name "
                    f"is correct and refer to HuggingFace documentation: "
                    f"https://huggingface.co/docs/transformers/model_doc/auto",
                ) from ex
            else:
                raise HuggingFaceModelError(
                    name, f"Failed to load model '{name}': {str(ex)}"
                ) from ex


def _get_model_info(
    config_class_name: str, config: HuggingFaceModelConfig
) -> Dict[str, str] | None:
    """Get model info from registries or config."""

    # Check user registry first
    if config_class_name in _USER_REGISTRY:
        return _USER_REGISTRY[config_class_name]

    # Check built-in special models
    if config_class_name in _HF_SPECIAL_MODELS:
        return _HF_SPECIAL_MODELS[config_class_name]

    # Check if config specifies custom classes
    if config.model_type == "custom" and config.custom_model_class:
        info = {"model_class": config.custom_model_class}
        if config.custom_processor_class:
            info["processor_class"] = config.custom_processor_class
        return info

    return None

def _replace_layer(
        obj:object, path:str, value:object
) -> None:
    """Replace a potentially indexed, nested object atrribute
    by parsing an object path string

    :param obj: The object to be modified, e.g. model (`QwenModel`)

    :param path: The path of the attribute to be modified, e.g. ``model.thinker.fc2``

    :param value: The object to substitute, e.g. a `RowShardedLinear` layer
    """
    parts = path.split('.')
    for i, part in enumerate(parts[:-1]):
        # If the part is an integer, treat as index
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    # Handle trailing indices if present
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        setattr(obj, last, value)

def _shard_qwen_omni_model(model:Qwen2_5OmniForConditionalGeneration, gangs:Gangs) -> Qwen2_5OmniForConditionalGeneration:
    """
    Shard a QwenOmni HuggingFace checkpoint to provided gangs, replacing
    layers with fairseq2 compatible linear layers

    :param model: The model to shard
    
    :param gangs: The gangs to use when sharding

    :returns: The sharded model with replaced layers
    """
    qkv_pattern_c = re.compile('[.]q_|[.]k_|[.]v_|_[qkv]$|[.][qkv]$')
    out_pattern_c = re.compile('out|[.]o_|_o$|[.]o$|[.]proj$')
    col_ffn_pattern_c = re.compile('ff.*[02]|fc[1]|mlp.*[01]|[.]gate_|[.]down_')
    row_ffn_pattern_c = re.compile('ff.*[13]|up_|fc2|mlp.*[2]')

    fs_model = copy.deepcopy(model)

    progress_bar = tqdm(len(model.modules))
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            output_dim, input_dim, bias, dtype = module.out_features, module.in_features, module.bias is not None, module.weight.dtype
            fs_proj = Linear(input_dim=input_dim, output_dim=output_dim, bias=bias, dtype=dtype)
            state_dict = module.state_dict()
            fs_proj.load_state_dict(state_dict)
            if qkv_pattern_c.search(name):
                sharded_proj = ColumnShardedLinear.from_linear(fs_proj, gang=gangs))
            elif out_pattern_c.search(name):
                sharded_proj = RowShardedLinear.from_linear(fs_proj, gang=gangs))
            elif col_ffn_pattern_c.search(name):
                sharded_proj = ColumnShardedLinear.from_linear(fs_proj, gang=gang))
            elif row_ffn_pattern_c.search(name):
                sharded_proj = RowShardedLinear.from_linear(fs_proj, gang=gangs))
            else:
                continue

            replace_layer(fs_model, name, sharded_proj)

        progress_bar.update(1)

    return fs_model

def _load_special_model(
    name: str, config: HuggingFaceModelConfig, model_info: Dict[str, str], gangs: Gang | None = None
) -> Any:
    """Load a model using special/custom classes."""

    log.info(f"Loading special model '{name}' using custom classes")

    # Prepare kwargs for from_pretrained
    load_kwargs = _prepare_load_kwargs(config)

    # Needed for Qwen Omni 2.5 in transformers v2.51.1
    load_kwargs.pop("safe_serialization")

    # Import and load the model class
    model_class_name = model_info["model_class"]
    try:
        model_class = _import_class_from_transformers(model_class_name)
        model = model_class.from_pretrained(**load_kwargs)
        
    except Exception as ex:
        raise HuggingFaceModelError(
            name,
            f"Failed to load model using custom class '{model_class_name}': {str(ex)}",
        ) from ex

    # Shard the model according to available gangs
    if gangs.tp_size > 1:
        try:
            model = _shard_qwen_omni_model(model, gangs)
            print("Model successfully sharded!")
        except Exception as e:
            print(f"Error sharding the model. Is special model type supported? (Qwen2.5-Omni) {e}")

    # Load tokenizer/processor
    if "processor_class" in model_info:
        processor_class_name = model_info["processor_class"]
        try:
            processor_class = _import_class_from_transformers(processor_class_name)
            processor = processor_class.from_pretrained(name)
            model.processor = processor
        except Exception as ex:
            log.warning(
                f"Failed to load processor '{processor_class_name}', skipping: {str(ex)}"
            )

    return model


def _load_auto_model(name: str, config: HuggingFaceModelConfig, hf_config: Any) -> Any:
    """Load a model using Auto classes."""

    log.info(f"Loading model '{name}' using Auto classes")

    # Prepare kwargs for from_pretrained
    load_kwargs = _prepare_load_kwargs(config)

    # Determine which AutoModel class to use
    auto_model_class = _get_auto_model_class(config, hf_config)
    
    try:
        model = auto_model_class.from_pretrained(**load_kwargs)
    except Exception as ex:
        # Check if this might be an unsupported model
        if "does not appear to have a file named config.json" not in str(ex):
            raise HuggingFaceModelError(
                name,
                f"Model '{name}' is not supported by HuggingFace AutoModel.\nError: {str(ex)}.\nPlease refer to https://huggingface.co/docs/transformers/model_doc/auto\nfor supported architectures or register a custom class.",  # fmt: skip
            ) from ex
        else:
            raise

    # Load tokenizer/processor
    if config.use_processor:
        try:
            processor = AutoProcessor.from_pretrained(name)
            # Set processor to model
            model.processor = processor
        except Exception:
            log.warning(f"AutoProcessor failed for '{name}', skipping")

    return model


def _get_auto_model_class(config: HuggingFaceModelConfig, hf_config: Any) -> Any:
    """Determine which AutoModel class to use."""

    if config.model_type == "causal_lm":
        return AutoModelForCausalLM
    elif config.model_type == "seq2seq_lm":
        return AutoModelForSeq2SeqLM
    elif config.model_type == "auto":
        return AutoModel
    else:
        # Auto-detect based on config
        if hasattr(hf_config, "is_encoder_decoder") and hf_config.is_encoder_decoder:
            return AutoModelForSeq2SeqLM
        else:
            # Default to causal LM for decoder-only models
            return AutoModelForCausalLM


def _prepare_load_kwargs(config: HuggingFaceModelConfig) -> Dict[str, Any]:
    """Prepare kwargs for from_pretrained call."""

    kwargs: Dict[str, Any] = {}

    # Get the local model path (download if necessary)
    model_path = _get_model_path(config)
    kwargs["pretrained_model_name_or_path"] = model_path
    kwargs["use_safetensors"] = True
    kwargs["safe_serialization"] = True

    # Set dtype
    _set_dtype_kwargs(config, kwargs)

    # Set trust_remote_code
    if config.trust_remote_code:
        kwargs["trust_remote_code"] = True

    # Handle device_map for auto device placement
    _set_device_kwargs(config, kwargs)

    # Add any additional kwargs from config
    if config.load_kwargs:
        kwargs.update(config.load_kwargs)

    return kwargs


def _get_model_path(config: HuggingFaceModelConfig) -> Path:
    """
    Download and return the local path to the HuggingFace model.
    Uses caching to avoid re-downloading the same model.

    .. note::
        This function requires internet access on the first call to download the model.
        Subsequent calls will use the cached version in ``$HF_HOME``.

    :param config: HuggingFace model configuration
    :returns: Local path to the downloaded model
    """
    uri = Uri.maybe_parse(f"hg://{config.hf_name}")

    if uri is None or uri.scheme != "hg":
        raise ValueError(f"Invalid HuggingFace model URI: {config.hf_name}")

    hf_hub = HuggingFaceHub()
    path = hf_hub.download_model(uri, config.hf_name)

    return path


def _set_dtype_kwargs(config: HuggingFaceModelConfig, kwargs: Dict[str, Any]) -> None:
    """Set dtype-related kwargs."""
    if config.dtype != "auto":
        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if config.dtype in dtype_map:
            kwargs["dtype"] = dtype_map[config.dtype]
    else:
        kwargs["dtype"] = "auto"


def _set_device_kwargs(config: HuggingFaceModelConfig, kwargs: Dict[str, Any]) -> None:
    """Set device-related kwargs."""
    if config.device == "auto":
        try:
            import accelerate  # type: ignore[import-untyped]  # noqa: F401

            kwargs["device_map"] = "auto"
        except ImportError:
            log.warning(
                "accelerate library not found. Cannot use device_map='auto'. Install with `pip install accelerate`."
            )


def _import_class_from_transformers(class_name: str) -> Any:
    """Import a class from the transformers library."""

    try:
        # Try importing from transformers directly
        transformers_module = importlib.import_module("transformers")
        return getattr(transformers_module, class_name)
    except AttributeError:
        # If not found in main module, it might be in a submodule
        # This is a simplified approach
        raise ImportError(
            f"Class '{class_name}' not found in transformers library. Make sure you have the correct version installed."  # fmt: skip
        )
