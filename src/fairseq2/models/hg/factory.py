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
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any, Dict, Type

import torch
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from fairseq2.assets import HuggingFaceHub
from fairseq2.device import detect_default_device
from fairseq2.error import NotSupportedError, OperationalError
from fairseq2.file_system import LocalFileSystem
from fairseq2.gang import (
    Gangs,
    get_current_gangs,
)
from fairseq2.logging import log
from fairseq2.models.hg.adapter import wrap_hg_model_if_causal_lm
from fairseq2.models.hg.config import HuggingFaceModelConfig
from fairseq2.nn import Linear
from fairseq2.utils.uri import Uri

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
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        DiTAttention,
        Qwen2_5OmniAttention,
        Qwen2_5OmniAudioAttention,
        Qwen2_5OmniForConditionalGeneration,
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
    """Exception raised when HuggingFace model loading fails."""

    def __init__(self, model_name: str, message: str) -> None:
        super().__init__(message)
        self.model_name = model_name


def _class_name(cls: type | str) -> str:
    """Return the class name string, whether given a type or already a string."""
    return cls if isinstance(cls, str) else cls.__name__


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

    entry = {"model_class": _class_name(model_class)}

    if tokenizer_class is not None:
        entry["tokenizer_class"] = _class_name(tokenizer_class)

    if processor_class is not None:
        entry["processor_class"] = _class_name(processor_class)

    _USER_REGISTRY[config_class_name] = entry

    log.info(f"Registered custom HF model mapping: {config_class_name} -> {entry}")


def create_hg_model(config: HuggingFaceModelConfig) -> Any:
    """
    Create a HuggingFace model from configuration.

    This factory loads models directly from HuggingFace Hub with transformers.

    :param config: HuggingFace model configuration
    :returns: HuggingFace PreTrainedModel
    :raises: OperationalError: If transformers library is not available
    :raises: HuggingFaceModelError: If model loading fails
    :raises: NotSupportedError: If transformers library is not available
    """

    if torch.cuda.is_available():
        gangs = get_current_gangs()
        return HgFactory(config, gangs).create_model()
    else:
        return HgFactory(config).create_model()


class HgFactory:
    """Factory for creating HuggingFace models.

    :param config: The HuggingFace model configuration.
    :param gangs: The gangs to use for distributed model loading.
    """

    def __init__(
        self, config: HuggingFaceModelConfig, gangs: Gangs | None = None
    ) -> None:
        """Initialize the factory with configuration."""
        self._config = config
        self._gangs = gangs

    def create_model(self) -> Any:
        """Create the model according to the configuration.

        :returns: The loaded model.
        :raises NotSupportedError: If transformers is not available.
        :raises HuggingFaceModelError: If model loading fails.
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
                f"Loading HuggingFace model '{name}' with config {config_class_name}"
            )

            # Check if this is a special case model
            model_info = _get_model_info(config_class_name, config)

            if model_info:
                model = _load_special_model(name, config, model_info, gangs)
            else:
                model = _load_auto_model(name, config, hf_config)

            # Wrap causal LM models with fairseq2 adapter
            model = wrap_hg_model_if_causal_lm(model, config)

            return model

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
    layers_indexable: Qwen2_5OmniForConditionalGeneration, path: str, value: object
) -> None:
    """Replace a potentially indexed, nested object attribute
    by parsing an object path string.

    :param layers_indexable: The object to be modified, e.g. model.
    :param path: The path of the attribute to be modified, e.g. ``model.thinker.fc2``.
    :param value: The object to substitute, e.g. a ``RowShardedLinear`` layer.
    """
    parts = path.split(".")
    for i, part in enumerate(parts[:-1]):
        # If the part is an integer, treat as index
        if part.isdigit():
            layers_indexable = layers_indexable[int(part)]  # type: ignore
        else:
            layers_indexable = getattr(layers_indexable, part)
    # Handle trailing indices if present
    last = parts[-1]
    if last.isdigit():
        layers_indexable[int(last)] = value  # type: ignore
    else:
        setattr(layers_indexable, last, value)


def _simple_shard_qwen_omni_model(
    model: Qwen2_5OmniForConditionalGeneration, gangs: Gangs
) -> Qwen2_5OmniForConditionalGeneration:
    """
    Shard a QwenOmni HuggingFace checkpoint to provided gangs, replacing
    layers with fairseq2 compatible linear layers (non-row/column)

    :param model: The model to shard

    :param gangs: The gangs to use when sharding

    :returns: The sharded model with replaced layers
    """

    qkv_pattern_c = re.compile("[.]q_|[.]k_|[.]v_|_[qkv]$|[.][qkv]$")
    out_pattern_c = re.compile("out|[.]o_|_o$|[.]o$|[.]proj$")

    # Collect replacements first to avoid mutating the model while iterating.
    replacements: list[tuple[str, object]] = []

    gang = gangs.tp

    for name, module in model.named_modules():
        # Place attention heads on a single device
        if (
            isinstance(module, Qwen2_5OmniAudioAttention)
            or isinstance(module, Qwen2_5OmniAttention)
            or isinstance(module, DiTAttention)
        ):
            replacements.append((name, module.to(gang.device)))

        # Place gated and projection layers, MLP, FFN according to gang sharding strategy
        elif isinstance(module, torch.nn.Linear):
            output_dim, input_dim, bias, dtype = (
                module.out_features,
                module.in_features,
                module.bias is not None,
                module.weight.dtype,
            )
            if not qkv_pattern_c.search(name) and not out_pattern_c.search(name):
                fs_proj = Linear(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias=bias,
                    dtype=dtype,
                    device=gang.device,
                )
                fs_proj.load_state_dict(module.state_dict())
                replacements.append((name, fs_proj))

    # Apply all replacements in-place.
    for name, replacement in tqdm(replacements):
        _replace_layer(model, name, replacement)

    return model


def _load_special_model(
    name: str,
    config: HuggingFaceModelConfig,
    model_info: Dict[str, str],
    gangs: Gangs | None = None,
) -> Any:
    """Load a model using special/custom classes."""

    log.info(f"Loading special model '{name}' using custom classes")

    if gangs is not None:
        device = gangs.root.device
    else:
        device = detect_default_device()

    # Prepare kwargs for from_pretrained
    load_kwargs = _prepare_load_kwargs(config)

    load_kwargs["device_map"] = device
    load_kwargs["ignore_mismatched_sizes"] = True

    # Import and load the model class
    model_class_name = model_info["model_class"]
    try:
        model_class = _import_class_from_transformers(model_class_name)
        try:
            model = model_class.from_pretrained(
                **load_kwargs, attn_implementation="flash_attention_2"
            )
            log.info("Using flash_attention_2.")
        except Exception:
            log.info("Couldn't load flash_attention_2. Using classic SDPA.")
            model = model_class.from_pretrained(**load_kwargs)

    except Exception as ex:
        raise HuggingFaceModelError(
            name,
            f"Failed to load model using custom class '{model_class_name}': {str(ex)}",
        ) from ex

    # Shard the model according to available gangs
    if gangs is not None and gangs.tp.size > 1:
        try:
            log.info(f"Sharding model with {gangs.tp.size} gangs...")
            model = _simple_shard_qwen_omni_model(model, gangs)
            log.info("Model successfully sharded!")

        except Exception as e:
            log.warning(
                f"Error sharding the model. Is special model type supported? (Qwen2.5-Omni) {e}"
            )

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


def _is_multimodal_config(hf_config: Any) -> bool:
    """Check if a HuggingFace config indicates a multimodal model."""
    if hasattr(hf_config, "processor_class") and hf_config.processor_class:
        return True
    auto_map = getattr(hf_config, "auto_map", {}) or {}
    if any("Processor" in v for v in auto_map.values() if isinstance(v, str)):
        return True
    return False


def _load_auto_model(name: str, config: HuggingFaceModelConfig, hf_config: Any) -> Any:
    """Load a model using Auto classes."""

    log.info(f"Loading model '{name}' using Auto classes")

    load_kwargs = _prepare_load_kwargs(config)

    auto_model_class = _get_auto_model_class(config, hf_config)

    try:
        model = auto_model_class.from_pretrained(**load_kwargs)
    except Exception as ex:
        if _is_multimodal_config(hf_config):
            raise HuggingFaceModelError(
                name,
                f"Model '{name}' appears to be a multimodal model that requires "
                f"a processor and cannot be loaded via Auto classes. "
                f"Register it with register_hg_model_class() or use "
                f"model_type='custom' with the appropriate model class.",
            ) from ex
        if "does not appear to have a file named config.json" not in str(ex):
            raise HuggingFaceModelError(
                name,
                f"Model '{name}' is not supported by HuggingFace AutoModel.\nError: {str(ex)}.\nPlease refer to https://huggingface.co/docs/transformers/model_doc/auto\nfor supported architectures or register a custom class.",  # fmt: skip
            ) from ex
        else:
            raise

    if config.use_processor:
        try:
            processor = AutoProcessor.from_pretrained(name)
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
        # Try auto-detection based on config
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

    hf_hub = HuggingFaceHub(LocalFileSystem())
    path = hf_hub.download_model(uri, config.hf_name)

    return path


def _set_dtype_kwargs(config: HuggingFaceModelConfig, kwargs: Dict[str, Any]) -> None:
    """Set dtype-related kwargs."""
    if config.dtype is not None:
        kwargs["dtype"] = config.dtype
    else:
        kwargs["dtype"] = "auto"


def _set_device_kwargs(config: HuggingFaceModelConfig, kwargs: Dict[str, Any]) -> None:
    """Set device-related kwargs."""
    if config.device == "auto":
        try:
            import accelerate  # type: ignore[import-untyped,import-not-found]  # noqa: F401

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
        raise ImportError(
            f"Class '{class_name}' not found in transformers library. Make sure you have the correct version installed."  # fmt: skip
        )
