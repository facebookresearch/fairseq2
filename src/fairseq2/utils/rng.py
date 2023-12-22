# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch
from torch import Generator, Tensor

from fairseq2.typing import CPU, Device


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


class RNGBag:
    """Holds a collection of random number generators."""

    generators: List[Generator]

    def __init__(self, *generators: Generator) -> None:
        """
        :param generators:
            The generators to hold.
        """
        self.generators = list(generators)

    @staticmethod
    def from_device_defaults(*devices: Device) -> RNGBag:
        """Create an :class:`RNGBag` instance holding the default random number
        generators of ``devices``."""
        unique_devices = set()

        generators = []

        for device in devices:
            if device in unique_devices:
                raise ValueError(f"`devices` already contains the device '{device}'.")

            unique_devices.add(device)

            if device == CPU:
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
                    f"`RNGBag` supports only CPU and CUDA devices, but a {device.type.upper()} device is specified."
                )

        return RNGBag(*generators)

    def seed(self) -> None:
        """Set the seed of the random number generators to a random number."""
        if not self.generators:
            return

        self.generators[0].seed()

        random_seed = self.generators[0].initial_seed()

        for g in self.generators[1:]:
            g.manual_seed(random_seed)

    def manual_seed(self, seed: int) -> None:
        """Set the seed of the random number generators."""
        if seed < 0 or seed >= 1 << 32:
            raise ValueError(
                f"`seed` must be greater than or equal to 0 and less than 2^32, but is {seed} instead."
            )

        for g in self.generators:
            g.manual_seed(seed)

    def state_dict(self) -> Dict[str, Any]:
        return {"generators": [g.get_state() for g in self.generators]}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        try:
            states = state_dict["generators"]
        except KeyError:
            raise ValueError("`state_dict` must contain an item named `generators`.")

        if not isinstance(states, list):
            raise ValueError(
                f"The `generators` item of `state_dict` must be of type `{list}`, but is of type `{type(states)}` instead."
            )

        if len(states) != len(self.generators):
            raise ValueError(
                f"The number of generators in `state_dict` must match the number of generators in the bag ({len(self.generators)}), but is {len(states)} instead."
            )

        for idx, state in enumerate(states):
            if not isinstance(state, Tensor):
                raise ValueError(
                    f"The generator states in `state_dict` must be of type `{Tensor}`, but the element at index {idx} is of type `{type(state)}` instead."
                )

            self.generators[idx].set_state(state.clone())
