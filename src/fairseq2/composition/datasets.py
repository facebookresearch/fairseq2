# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Protocol, TypeVar

from fairseq2.datasets import DatasetFamily, DatasetOpener, StandardDatasetFamily
from fairseq2.error import InternalError
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    wire_object,
)

DatasetT_co = TypeVar("DatasetT_co", covariant=True)

DatasetConfigT_contra = TypeVar("DatasetConfigT_contra", contravariant=True)


class AdvancedDatasetOpener(Protocol[DatasetConfigT_contra, DatasetT_co]):
    def __call__(
        self, resolver: DependencyResolver, config: DatasetConfigT_contra
    ) -> DatasetT_co: ...


DatasetT = TypeVar("DatasetT")

DatasetConfigT = TypeVar("DatasetConfigT")


def register_dataset_family(
    container: DependencyContainer,
    name: str,
    kls: type[DatasetT],
    config_kls: type[DatasetConfigT],
    *,
    opener: DatasetOpener[DatasetConfigT, DatasetT] | None = None,
    advanced_opener: AdvancedDatasetOpener[DatasetConfigT, DatasetT] | None = None,
) -> None:
    if advanced_opener is not None:
        if opener is not None:
            raise ValueError(
                "`opener` and `advanced_opener` must not be specified at the same time."
            )
    elif opener is None:
        raise ValueError("`opener` or `advanced_opener` must be specified.")

    def create_family(resolver: DependencyResolver) -> DatasetFamily:
        nonlocal opener

        if advanced_opener is not None:

            def open_dataset(config: DatasetConfigT) -> DatasetT:
                return advanced_opener(resolver, config)

            opener = open_dataset
        elif opener is None:
            raise InternalError("`opener` is `None`.")

        return wire_object(
            resolver,
            StandardDatasetFamily,
            name=name,
            kls=kls,
            config_kls=config_kls,
            opener=opener,
        )

    container.register(DatasetFamily, create_family, key=name)


def _register_dataset_families(container: DependencyContainer) -> None:
    pass
