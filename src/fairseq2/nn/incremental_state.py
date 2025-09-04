# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, final

from torch import Tensor
from torch.nn import Module


class IncrementalState(ABC):
    """
    Holds the state of a module during incremental decoding.

    Incremental decoding is a special mode at inference time where the module
    only receives an input corresponding to the previous output and must produce
    the next output incrementally. Thus the module must cache any long-term
    state that is needed about the sequence.
    """

    @abstractmethod
    def reorder(self, new_order: Tensor) -> None:
        """
        Rearranges the state according to a new batch order.

        This will be called when the order of the batch has changed. A typical
        use case is beam search, where the batch order changes between steps
        based on the selection of beams.

        :param new_order:
            The new order of the batch. It is frequently used with
            :func:`torch.index_select` to rearrange the state tensors. *Shape:*
            :math:`(N)`, where :math:`N` is the batch size.
        """

    @abstractmethod
    def size_bytes(self) -> int:
        """Returns the size of the state in bytes."""

    @abstractmethod
    def capacity_bytes(self) -> int:
        """Returns the reserved capacity of the state in bytes."""


T = TypeVar("T", bound=IncrementalState)


@final
class IncrementalStateBag:
    """Holds the module states during incremental decoding."""

    def __init__(
        self, max_num_steps: int, *, capacity_increment: int | None = 16
    ) -> None:
        """
        :param max_num_steps: The maximum number of steps to take.
        :param capacity_increment: The sequence length capacity of state tensors
            will be incremented by multiples of this value. If ``None``, state
            tensors will be preallocated with a capacity of ``max_num_steps``.
        """
        if capacity_increment is not None and capacity_increment < 1:
            raise ValueError(
                f"`capacity_increment` must be greater than or equal to 1, but is {capacity_increment} instead."
            )

        self._step_nr = 0
        self._max_num_steps = max_num_steps
        self._capacity_increment = capacity_increment
        self._module_states: dict[Module, IncrementalState] = {}

    def increment_step_nr(self, value: int = 1) -> None:
        """
        Increments the step number.

        This method should be called after every decoding step. It is used by
        modules to keep track of the position in the sequence.

        :param value: The value by which to increment the step number.
        """
        step_nr = self._step_nr + value

        if step_nr >= self._max_num_steps:
            raise ValueError(
                f"The current step number ({self._step_nr}) with `value` increment ({value}) must be less than or equal to the maximum number of steps ({self.max_num_steps}), but is {self._step_nr + value} instead."
            )

        self._step_nr = step_nr

    def maybe_get_state(self, m: Module, kls: type[T]) -> T | None:
        """
        Gets the state of ``m`` if present in the bag.

        :param m: The module.
        :param kls: The expected ``type`` of the state. If the type of the state
            in the bag does not match ``kls``, ``None`` will be returned.

        :returns: The state of the module.
        """
        state = self._module_states.get(m, None)
        if isinstance(state, kls):
            return state
        else:
            return None

    def set_state(self, m: Module, state: IncrementalState) -> None:
        """
        Sets the state of ``m``.

        :param m: The module.
        :param state: The state to store.
        """
        self._module_states[m] = state

    def reorder(self, new_order: Tensor) -> None:
        """
        Reorders the module states.

        See :meth:`IncrementalState.reorder` for more information.
        """
        for state in self._module_states.values():
            state.reorder(new_order)

    @property
    def step_nr(self) -> int:
        """The current step number."""
        return self._step_nr

    @property
    def max_num_steps(self) -> int:
        """The maximum number of steps."""
        return self._max_num_steps

    @property
    def capacity_increment(self) -> int | None:
        """
        The sequence length capacity of state tensors will be incremented by
        multiples of this value.
        """
        return self._capacity_increment

    def size_bytes(self) -> int:
        """Returns the size of the state bag in bytes."""
        return sum(s.size_bytes() for s in self._module_states.values())

    def capacity_bytes(self) -> int:
        """Returns the reserved capacity of the state bag in bytes."""
        return sum(s.capacity_bytes() for s in self._module_states.values())
