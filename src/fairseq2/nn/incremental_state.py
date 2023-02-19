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
    """Holds the state of a module during an incremental evaluation.

    Incremental evaluation is a special mode where the module only receives an
    input corresponding to the previous output and must produce the next output
    incrementally. Thus the module must cache any long-term state that is needed
    about the sequence.
    """

    @abstractmethod
    def reorder(self, new_order: Tensor) -> None:
        """Rearrange the incremental state according to a new batch order.

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
    """Holds the module states during an incremental evaluation."""

    _step: int
    _module_states: Dict[Module, IncrementalState]

    def __init__(self) -> None:
        self._step = 0

        self._module_states = {}

    @property
    def step(self) -> int:
        """Return the current step in the sequence."""
        return self._step

    def increment_step(self, delta: int = 1) -> None:
        """Increment the step.

        This method should be called after every incremental evaluation (e.g.
        beam search). It is used by modules to keep track of the position in
        the sequence.

        :param delta:
            The value by which to increment the step.
        """
        self._step += delta

    def get_state(self, m: Module, kls: Type[T]) -> Optional[T]:
        """Get the incremental state of ``m``, or ``None`` if ``m`` is not
        present in the bag.

        :param m:
            The module.
        :param kls:
            The expected ``type`` of the incremental state. If the type of the
            incremental state in the bag does not match ``kls``, ``None`` will
            be returned.

        :returns:
            The incremental state of the module.
        """
        state = self._module_states.get(m, None)
        if isinstance(state, kls):
            return state
        else:
            return None

    def set_state(self, m: Module, state: IncrementalState) -> None:
        """Set the incremental state of ``m``.

        :param m:
            The module.
        :param state:
            The incremental state to store.
        """
        self._module_states[m] = state

    def reorder(self, new_order: Tensor) -> None:
        """Reorder all incremental states in the bag.

        See :meth:`IncrementalState.reorder` for more information.
        """
        for state in self._module_states.values():
            state.reorder(new_order)
