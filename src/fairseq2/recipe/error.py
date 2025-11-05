# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, final

from torch.nn import Module

from fairseq2.device import Device


class RecipeConfigParseError(Exception):
    pass


class RecipeError(Exception):
    pass


class BeamSearchAlgorithmNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known beam search algorithm.")

        self.name = name


class DeviceTypeNotSupportedError(Exception):
    def __init__(self, device: Device) -> None:
        super().__init__(
            f"Only `cpu` and `cuda` devices are supported, but the device of the process is `{device}`."
        )

        self.device = device
        self.supported_devices = {"cpu", "cuda"}


class FSDPNotSupportedError(Exception):
    def __init__(self) -> None:
        super().__init__("Model does not support FSDP.")


class GangTopologyError(Exception):
    def __init__(self, world_size: int, tp_size: int) -> None:
        super().__init__(
            f"`gang.tensor_parallel_size` must be a factor of the number of processes in the gang ({world_size}), but is {tp_size} instead."
        )

        self.world_size = world_size
        self.tp_size = tp_size


class HSDPTopologyError(Exception):
    def __init__(self, local_world_size: int, dp_size: int) -> None:
        super().__init__(
            f"Local world size must be a factor of the number of processes in the data parallel gang ({dp_size}), but is {local_world_size} instead."
        )

        self.local_world_size = local_world_size
        self.dp_size = dp_size


class HuggingFaceNotSupportedError(Exception):
    def __init__(self) -> None:
        super().__init__("Model does not support Hugging Face conversion.")


class LayerwiseACNotSupportedError(Exception):
    def __init__(self) -> None:
        super().__init__("Model does not support layerwise activation checkpointing.")


class LRSchedulerNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known learning rate scheduler.")

        self.name = name


class ManualGradScalingNotSupportedError(Exception):
    def __init__(self) -> None:
        super().__init__(
            "Selected optimizer configuration does not support manual fp16 gradient scaling required for FSDP."
        )


class MinimumLossScaleReachedError(Exception):
    def __init__(self, step_nr: int) -> None:
        super().__init__(f"Loss is scaled down to minimum at step {step_nr}.")

        self.step_nr = step_nr


class MetricNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known metric.")

        self.name = name


class ModelCheckpointNotFoundError(Exception):
    def __init__(self, path: Path) -> None:
        super().__init__(f"{path} does not point to a model checkpoint.")

        self.path = path


class ModelTypeNotValidError(Exception):
    def __init__(self, kls: type[Module], expected_kls: type[Module]) -> None:
        super().__init__(
            f"Model must be of type `{expected_kls}`, but is of type `{kls}` instead."
        )

        self.kls = kls
        self.expected_kls = expected_kls


class OptimizerNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known optimizer.")

        self.name = name


class SamplerNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known sampler.")

        self.name = name


class SequenceGeneratorNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known sequence generator.")

        self.name = name


class SplitNotKnownError(Exception):
    def __init__(
        self, dataset_name: str, split: str, available_splits: Sequence[str]
    ) -> None:
        super().__init__(f"{split} is not a known split of the {dataset_name} dataset.")

        self.dataset_name = dataset_name
        self.split = split
        self.available_splits = available_splits


class TokenizerModelNotFoundError(Exception):
    def __init__(self, path: Path) -> None:
        super().__init__(f"{path} does not point to a tokenizer model.")

        self.path = path


class TorchCompileError(Exception):
    def __init__(self) -> None:
        super().__init__("torch.compile() failed.")


class TorchCompileNotSupportedError(Exception):
    def __init__(self) -> None:
        super().__init__("Model does not support torch.compile().")


class TorchDistributedNotAvailableError(Exception):
    def __init__(self) -> None:
        super().__init__("torch.distributed is not available.")


class WandbInitializationError(Exception):
    def __init__(self) -> None:
        super().__init__("Weights & Biases client cannot be initialized.")


@final
class ErrorContext:
    _CONFIG_SECTION_ATTR_NAME: Final = "__fs2_config_section__"

    @classmethod
    def set_config_section_name(cls, ex: Exception, name: str) -> None:
        setattr(ex, cls._CONFIG_SECTION_ATTR_NAME, name)

    @classmethod
    def maybe_get_config_section_name(cls, ex: Exception) -> str | None:
        return getattr(ex, cls._CONFIG_SECTION_ATTR_NAME, None)
