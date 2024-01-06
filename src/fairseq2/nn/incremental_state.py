# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, TypeVar

from torch import Tensor
from torch.nn import Module


class IncrementalState(ABC):
    """Holds the state of a module during incremental decoding.

    Incremental decoding is a special mode at inference time where the module
    only receives an input corresponding to the previous output and must produce
    the next output incrementally. Thus the module must cache any long-term
    state that is needed about the sequence.
    """

    @abstractmethod
    def reorder(self, new_order: Tensor) -> None:
        """Rearrange the state according to a new batch order.

        This will be called when the order of the batch has changed. A typical
        use case is beam search, where the batch order changes between steps
        based on the selection of beams.

        :param new_order:
            The new order of the batch. It is frequently used with
            :func:`torch.index_select` to rearrange the state tensors. *Shape:*
            :math:`(N)`, where :math:`N` is the batch size.
        """


T = TypeVar("T", bound=IncrementalState)


class IncrementalStateBag:
    """Holds the module states during incremental decoding."""

    step_nr: int
    max_num_steps: int
    capacity_increment: Optional[int]

    _module_states: Dict[Module, IncrementalState]

    def __init__(
        self, max_num_steps: int, *, capacity_increment: Optional[int] = 16
    ) -> None:
        """
        :param max_num_steps:
            The expected maximum number of steps to take.
        :param capacity_increment:
            The sequence length capacity of state tensors will be incremented by
            multiples of this value. If ``None``, state tensors will be
            preallocated with a capacity of ``max_num_steps``.
        """
        if capacity_increment is not None and capacity_increment < 1:
            raise ValueError(
                f"`capacity_increment` must be greater than or equal to 1, but is {capacity_increment} instead."
            )

        self.step_nr = 0
        self.max_num_steps = max_num_steps
        self.capacity_increment = capacity_increment

        self._module_states = {}

    def increment_step_nr(self, value: int = 1) -> None:
        """Increment the step number.

        This method should be called after every decoding step. It is used by
        modules to keep track of the position in the sequence.

        :param value:
            The value by which to increment the step number.
        """
        step_nr = self.step_nr + value

        if step_nr >= self.max_num_steps:
            raise ValueError(
                f"The current step number ({self.step_nr}) with `value` increment ({value}) must be less than or equal to the maximum number of steps ({self.max_num_steps}), but is {self.step_nr + value} instead."
            )

        self.step_nr = step_nr

    def get_state(self, m: Module, kls: Type[T]) -> Optional[T]:
        """Get the state of ``m`` if present in the bag.

        :param m:
            The module.
        :param kls:
            The expected ``type`` of the state. If the type of the state in the
            bag does not match ``kls``, ``None`` will be returned.

        :returns:
            The state of the module.
        """
        state = self._module_states.get(m, None)
        if isinstance(state, kls):
            return state
        else:
            return None

    def set_state(self, m: Module, state: IncrementalState) -> None:
        """Set the state of ``m``.

        :param m:
            The module.
        :param state:
            The state to store.
        """
        self._module_states[m] = state

    def reorder(self, new_order: Tensor) -> None:
        """Reorder the module states.

        See :meth:`IncrementalState.reorder` for more information.
        """
        for state in self._module_states.values():
            state.reorder(new_order)
