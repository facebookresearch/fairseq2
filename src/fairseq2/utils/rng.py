# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import (
    Any,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    final,
)

import torch
from torch import Generator, Tensor

from fairseq2.typing import Device


def use_deterministic(value: bool, warn_only: bool = False) -> None:
    """Set whether PyTorch algorithms must use deterministic algorithms.

    :param value:
        If ``True``, uses deterministic algorithms.
    :param warn_only:
        If ``True``, operations that do not have a deterministic implementation
        will raise a warning instead of an error.
    """
    torch.backends.cudnn.benchmark = not value

    torch.use_deterministic_algorithms(value, warn_only=warn_only)


@final
class RngBag:
    """Holds a collection of random number generators."""

    _generators: List[Generator]

    def __init__(self, *generators: Generator) -> None:
        """
        :param generators:
            The generators to hold.
        """
        self._generators = list(generators)

    @staticmethod
    def from_device_defaults(*devices: Device) -> RngBag:
        """Create an :class:`RngBag` from the random number generators of ``devices``."""
        unique_devices = set()

        generators = []

        for device in devices:
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
                    f"`devices` must be of type `cpu` or `cuda`, but at least one device is of type `{device.type}` instead."
                )

        return RngBag(*generators)

    def add_generator(self, generator: Generator) -> None:
        """Add ``generator`` to the bag."""
        self._generators.append(generator)

    def seed(self) -> None:
        """Set the seed of the random number generators to a random number."""
        if not self._generators:
            return

        self._generators[0].seed()

        random_seed = self._generators[0].initial_seed()

        for g in self._generators[1:]:
            g.manual_seed(random_seed)

    def manual_seed(self, seed: int) -> None:
        """Set the seed of the random number generators."""
        if seed < 0 or seed >= 1 << 32:
            raise ValueError(
                f"`seed` must be greater than or equal to 0 and less than 2^32, but is {seed} instead."
            )

        for g in self._generators:
            g.manual_seed(seed)

    @contextmanager
    def temporary_manual_seed(self, seed: int) -> Iterator[None]:
        """Temporarily change the seed of the random number generators."""
        original_states = [g.get_state() for g in self._generators]

        self.manual_seed(seed)

        try:
            yield
        finally:
            for g, s in zip(self._generators, original_states):
                g.set_state(s)

    def state_dict(self) -> Dict[str, Any]:
        return {"generators": [g.get_state() for g in self._generators]}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        try:
            states = state_dict["generators"]
        except KeyError:
            raise ValueError("`state_dict` must contain an item named `generators`.")

        if not isinstance(states, list):
            raise TypeError(
                f"The `generators` item of `state_dict` must be of type `{list}`, but is of type `{type(states)}` instead."
            )

        if len(states) != len(self._generators):
            raise ValueError(
                f"The number of generators in `state_dict` must match the number of generators in the bag ({len(self._generators)}), but is {len(states)} instead."
            )

        for idx, state in enumerate(states):
            if not isinstance(state, Tensor):
                raise TypeError(
                    f"The generator states in `state_dict` must be of type `{Tensor}`, but the element at index {idx} is of type `{type(state)}` instead."
                )

            self._generators[idx].set_state(state.clone())


def temporary_manual_seed(
    devices: Iterable[Device], seed: Optional[int]
) -> ContextManager[None]:
    """Temporarily change the seed of the random number generators of ``devices``.

    :param devices:
        The devices whose random number generators will be updated.
    :param seed:
        The seed to set. If ``None``, becomes a no-op.
    """
    if seed is None:
        return nullcontext()

    rng_bag = RngBag.from_device_defaults(*devices)

    return rng_bag.temporary_manual_seed(seed)


# compat
RNGBag = RngBag
