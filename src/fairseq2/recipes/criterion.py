# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Mapping, Optional, Tuple, TypeVar

from torch import Tensor
from torch.nn import Module

from fairseq2.metrics import MetricBag
from fairseq2.typing import override
from fairseq2.utils.state import StatefulObjectBag

BatchT = TypeVar("BatchT")

BatchT_contra = TypeVar("BatchT_contra", contravariant=True)


class Criterion(ABC, Generic[BatchT_contra]):
    """Computes loss in a training recipe."""

    @abstractmethod
    def set_step(self, step_nr: int) -> None:
        """Set the training step.

        Typically used to modify the model (e.g. freeze or unfreeze) or metrics
        during training.

        :param step_nr:
            The number of the current training step.
        """

    @abstractmethod
    def compute_loss(self, batch: BatchT_contra) -> Tuple[Tensor, int]:
        """Compute the loss of ``batch``.

        :returns:
            The loss and the number of targets used to compute the loss.
        """

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        ...

    @property
    @abstractmethod
    def model(self) -> Module:
        """The underlying model used to compute loss."""

    @property
    @abstractmethod
    def train_metric_bag(self) -> MetricBag:
        """The metrics used for training."""

    @property
    @abstractmethod
    def valid_metric_bag(self) -> MetricBag:
        """The metrics used for validation."""

    @property
    @abstractmethod
    def throughput_metric_name(self) -> str:
        """The name of the metric to use for throughput calculation."""

    @property
    @abstractmethod
    def score_metric_name(self) -> Optional[str]:
        """The name of the metric to use for checkpoint score calculation."""


class AbstractCriterion(StatefulObjectBag, Criterion[BatchT]):
    """Provides a skeletal implementation of :class:`Criterion`."""

    def __init__(self, model: Module) -> None:
        super().__init__()

        self._model = model

    @override
    def set_step(self, step_nr: int) -> None:
        pass

    @property
    @override
    def model(self) -> Module:
        return self._model

    @property
    @override
    def throughput_metric_name(self) -> str:
        return "num_elements"

    @property
    @override
    def score_metric_name(self) -> Optional[str]:
        return None
