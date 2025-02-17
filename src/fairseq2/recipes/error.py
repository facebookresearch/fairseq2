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


class InvalidCheckpointPathError(Exception):
    pathname: str

    def __init__(self, pathname: str) -> None:
        super().__init__(f"'{pathname}' does not represent a valid file system path.")

        self.pathname = pathname


class ModelParallelismNotSupportedError(NotSupportedError):
    family: str
    model_name: str

    def __init__(self, family: str, model_name: str) -> None:
        super().__init__(
            f"The '{family}' family of the '{model_name}' model does not support non-data parallelism."
        )

        self.family = family
        self.model_name = model_name


class StaticGraphNotSupportedError(NotSupportedError):
    data_parallelism: str

    def __init__(self, data_parallelism: str) -> None:
        super().__init__(
            f"{data_parallelism} does not support non-static model graphs."
        )

        self.data_parallelism = data_parallelism


class HybridShardingNotSupportedError(NotSupportedError):
    data_parallelism: str

    def __init__(self, data_parallelism: str) -> None:
        super().__init__(
            f"{data_parallelism} with hybrid sharding does not support non-data parallelism."
        )

        self.data_parallelism = data_parallelism


class UnitError(Exception):
    pass
