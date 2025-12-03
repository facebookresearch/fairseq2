# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import final

import torch
from torch import Generator, Tensor
from typing_extensions import override

from fairseq2.device import Device
from fairseq2.error import CorruptDataError
from fairseq2.typing import Stateful


def use_deterministic(value: bool, warn_only: bool = False) -> None:
    """
    Sets whether PyTorch algorithms must use deterministic algorithms.

    If ``warn_only`` is ``True``, operations that do not have a deterministic
    implementation will raise a warning instead of an error.
    """
    torch.backends.cudnn.benchmark = not value

    torch.use_deterministic_algorithms(value, warn_only=warn_only)


@final
class RngBag(Stateful):
    """Holds a collection of random number generators."""

    def __init__(self, *generators: Generator) -> None:
        self._generators = list(generators)

    @staticmethod
    def from_device_defaults(*devices: Device) -> RngBag:
        """
        Creates an :class:`RngBag` instance from the default random number
        generators of the specified devices.
        """
        unique_devices = set()

        generators = []

        for idx, device in enumerate(devices):
            if device in unique_devices or device.type == "meta":
                continue

            unique_devices.add(device)

            if device.type == "cpu":
                generators.append(torch.default_generator)
            elif device.type == "cuda":
                # Ensure that the default CUDA generators are initialized.
                torch.cuda.init()

                idx = device.index
                if idx is None:
                    idx = torch.cuda.current_device()

                generators.append(torch.cuda.default_generators[idx])
            else:
                raise ValueError(
                    f"`devices` must be of type `cpu` or `cuda`, but the device at index {idx} is of type `{device.type}` instead."
                )

        return RngBag(*generators)

    def seed(self) -> None:
        """Sets the seed of the random number generators to a random number."""
        if not self._generators:
            return

        self._generators[0].seed()

        random_seed = self._generators[0].initial_seed()

        for g in self._generators[1:]:
            g.manual_seed(random_seed)

    def manual_seed(self, seed: int) -> None:
        """Sets the seed of the random number generators to the specified value."""
        if seed < 0 or seed >= 1 << 32:
            raise ValueError(
                f"`seed` must be greater than or equal to 0 and less than 2^32, but is {seed} instead."
            )

        for g in self._generators:
            g.manual_seed(seed)

    @contextmanager
    def temporary_manual_seed(self, seed: int) -> Iterator[None]:
        """
        Temporarily changes the seed of the random number generators to the
        specified value.
        """
        original_states = [g.get_state() for g in self._generators]

        self.manual_seed(seed)

        try:
            yield
        finally:
            for g, s in zip(self._generators, original_states):
                g.set_state(s)

    @override
    def state_dict(self) -> dict[str, object]:
        return {"generators": [g.get_state() for g in self._generators]}

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        state_dict = dict(state_dict)

        try:
            gen_tensors = state_dict.pop("generators")
        except KeyError:
            raise CorruptDataError(
                "`state_dict` does not contain a key named 'generators'."
            ) from None

        if not isinstance(gen_tensors, list):
            raise CorruptDataError(
                f"`state_dict['generators']` is expected to be of type `{list}`, but is of type `{type(gen_tensors)}` instead."
            )

        if len(gen_tensors) != len(self._generators):
            raise CorruptDataError(
                f"Number of generators in `state_dict['generators']` is expected to match the number of generators in the bag ({len(self._generators)}), but is {len(gen_tensors)} instead."
            )

        for idx, gen_tensor in enumerate(gen_tensors):
            if not isinstance(tensor, Tensor):
                raise CorruptDataError(
                    f"Generator states in `state_dict['generators']` are expected to be of type `{Tensor}`, but the state at index {idx} is of type `{type(gen_tensor)}` instead."
                )

            self._generators[idx].set_state(gen_tensor.clone())

        CorruptDataError.raise_if_state_dict_not_empty(state_dict)
