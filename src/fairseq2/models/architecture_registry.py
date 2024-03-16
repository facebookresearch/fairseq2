# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import AbstractSet, Callable, Dict, Generic, Protocol, TypeVar, final

ModelConfigT = TypeVar("ModelConfigT")

ModelConfigT_co = TypeVar("ModelConfigT_co", covariant=True)


class ModelConfigFactory(Protocol[ModelConfigT_co]):
    """Constructs instances of ``ModelConfigT``."""

    def __call__(self) -> ModelConfigT_co:
        ...


@final
class ModelArchitectureRegistry(Generic[ModelConfigT]):
    """Holds the architectures and associated configurations of a model family."""

    _configs: Dict[str, ModelConfigFactory[ModelConfigT]]

    def __init__(self) -> None:
        self._configs = {}

    def get_config(self, arch: str) -> ModelConfigT:
        """Return the model configuration of ``arch``."""
        try:
            return self._configs[arch]()
        except KeyError:
            raise ValueError(
                f"`arch` must be a registered model architecture, but is '{arch}' instead."
            )

    def register(
        self, arch: str, config_factory: ModelConfigFactory[ModelConfigT]
    ) -> None:
        """Register a new architecture.

        :param arch:
            The model architecture.
        :param config_factory:
            The factory to construct model configurations.
        """
        if arch in self._configs:
            raise ValueError(
                f"`arch` must be a unique model architecture, but '{arch}' is already registered."
            )

        self._configs[arch] = config_factory

    def decorator(
        self, arch: str
    ) -> Callable[[ModelConfigFactory[ModelConfigT]], ModelConfigFactory[ModelConfigT]]:
        """Register ``arch`` with the decorated model configuration factory."""

        def register(
            config_factory: ModelConfigFactory[ModelConfigT],
        ) -> ModelConfigFactory[ModelConfigT]:
            self.register(arch, config_factory)

            return config_factory

        return register

    def names(self) -> AbstractSet[str]:
        """Return the names of all model architectures."""
        return self._configs.keys()
