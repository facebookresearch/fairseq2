# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import AbstractSet, Callable, Dict, Generic, Protocol, TypeVar

ModelConfigT = TypeVar("ModelConfigT", covariant=True)


class ModelConfigFactory(Protocol[ModelConfigT]):
    """Constructs instances of ``ModelConfigT``."""

    def __call__(self) -> ModelConfigT:
        ...


class ArchitectureRegistry(Generic[ModelConfigT]):
    """Represents a registry of model architectures."""

    model_type: str
    configs: Dict[str, ModelConfigFactory[ModelConfigT]]

    def __init__(self, model_type: str) -> None:
        """
        :param model_type:
            The type of the model for which architectures will be registered.
        """
        self.model_type = model_type

        self.configs = {}

    def register(
        self, arch_name: str, config_factory: ModelConfigFactory[ModelConfigT]
    ) -> None:
        """Register a new architecture.

        :param arch_name:
            The name of the architecture.
        :param config_factory:
            The factory to construct model configurations.
        """
        if arch_name in self.configs:
            raise ValueError(
                f"The architecture name '{arch_name}' is already registered for '{self.model_type}'."
            )

        self.configs[arch_name] = config_factory

    def get_config(self, arch_name: str) -> ModelConfigT:
        """Return the model configuration of the specified architecture.

        :param arch_name:
            The name of the architecture.
        """
        try:
            return self.configs[arch_name]()
        except KeyError:
            raise ValueError(
                f"The registry of '{self.model_type}' does not contain an architecture named '{arch_name}'."
            )

    def names(self) -> AbstractSet[str]:
        """Return the names of all supported architectures."""
        return self.configs.keys()

    def decorator(
        self, arch_name: str
    ) -> Callable[[ModelConfigFactory[ModelConfigT]], ModelConfigFactory[ModelConfigT]]:
        """Register the specified architecture with the decorated model
        configuration factory.

        :param arch_name:
            The name of the architecture.
        """

        def register(
            config_factory: ModelConfigFactory[ModelConfigT],
        ) -> ModelConfigFactory[ModelConfigT]:
            self.register(arch_name, config_factory)

            return config_factory

        return register
