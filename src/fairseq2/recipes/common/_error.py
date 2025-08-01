# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

from fairseq2.error import NotSupportedError


class ModelPathNotFoundError(Exception):
    model_name: str
    path: Path

    def __init__(self, model_name: str, path: Path) -> None:
        super().__init__(
            f"The '{model_name}' model cannot be found at the '{path}' path."
        )

        self.model_name = model_name
        self.path = path


class DatasetPathNotFoundError(Exception):
    dataset_name: str
    path: Path

    def __init__(self, dataset_name: str, path: Path) -> None:
        super().__init__(
            f"The '{dataset_name}' dataset cannot be found at the '{path}' path."
        )

        self.dataset_name = dataset_name
        self.path = path


class ModelParallelismNotSupportedError(NotSupportedError):
    model_name: str

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"The '{model_name}' model does not support model parallelism."
        )

        self.model_name = model_name


class ModelCompilationNotSupportedError(NotSupportedError):
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
