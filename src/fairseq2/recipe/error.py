# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

from fairseq2.error import NotSupportedError


class ModelParallelismNotSupportedError(NotSupportedError):
    model_name: str

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"The '{model_name}' model does not support model parallelism."
        )

        self.model_name = model_name


class TorchCompileNotSupportedError(NotSupportedError):
    model_name: str

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"The '{model_name}' model does not support `torch.compile()`."
        )

        self.model_name = model_name


class ActivationCheckpointingNotSupportedError(NotSupportedError):
    model_name: str

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"The '{model_name}' model does not support activation checkpointing."
        )

        self.model_name = model_name


class FSDPNotSupportedError(NotSupportedError):
    model_name: str

    def __init__(self, model_name: str) -> None:
        super().__init__(f"The '{model_name}' model does not support FSDP.")

        self.model_name = model_name


class HuggingFaceNotSupportedError(NotSupportedError):
    model_name: str

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"The '{model_name}' model does not support Hugging Face conversion."
        )

        self.model_name = model_name


class HybridShardingNotSupportedError(NotSupportedError):
    def __init__(self) -> None:
        super().__init__(
            "Hybrid sharded data parallelism is not supported when model parallelism is enabled."
        )


class ModelInitializationError(Exception):
    model_name: str

    def __init__(self, model_name: str, message: str) -> None:
        super().__init__(message)

        self.model_name = model_name


class ModelNotFoundError(Exception):
    path: Path

    def __init__(self, path: Path) -> None:
        super().__init__(f"The '{path}' path does not point to a model.")

        self.path = path


class DatasetNotFoundError(Exception):
    path: Path

    def __init__(self, path: Path) -> None:
        super().__init__(f"The '{path}' path does not point to a dataset.")

        self.path = path


class TokenizerNotFoundError(Exception):
    path: Path

    def __init__(self, path: Path) -> None:
        super().__init__(f"The '{path}' path does not point to a tokenizer.")

        self.path = path


class UnspecifiedNumberOfStepsError(Exception):
    lr_scheduler_name: str

    def __init__(self, lr_scheduler_name: str) -> None:
        super().__init__(
            f"`regime.num_steps` must be specified for the '{lr_scheduler_name}' learning rate scheduler."
        )

        self.lr_scheduler_name = lr_scheduler_name


class UnknownOptimizerError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known optimizer.")

        self.name = name


class UnknownLRSchedulerError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known learning rate scheduler.")

        self.name = name


class UnknownSequenceGeneratorError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known sequence generator.")

        self.name = name


class UnknownSamplerError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known sampler.")

        self.name = name


class UnknownBeamSearchAlgorithmError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known beam search algorithm.")

        self.name = name


class UnknownMetricDescriptorError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known metric.")

        self.name = name
