# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import importlib
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
    final,
)

import torch
from torcheval.metrics import Metric
from typing_extensions import Self

from fairseq2.typing import Device, override

if TYPE_CHECKING:
    import numpy


@final
class HFMetric(Metric[torch.Tensor]):
    """
    A wrapper of HuggingFace `evaluate.Metric` that is compatible with
    fairseq2 MetricBag API (which uses `torcheval.metrics.Metric`)
    """

    def __init__(self, metric_name, device: Optional[Device] = None) -> None:
        try:
            evaluate = importlib.import_module("evaluate")
        except ImportError as exc:
            raise ImportError(
                "HFMetric requires the library `evaluate`, for instance via `pip install evaluate`"
            ) from exc
        super().__init__(device=device)
        self._metric = evaluate.load(metric_name)
        self._metric_name = metric_name
        self._add_state(
            metric_name, torch.zeros([]), device=device, dtype=torch.float32
        )

    @override
    @torch.inference_mode()
    def update(
        self,
        predictions: Optional[Union[List[Any], torch.Tensor, numpy.ndarray]] = None,
        references: Optional[Union[List[Any], torch.Tensor, numpy.ndarray]] = None,
        **kwargs,
    ) -> Self:
        self._metric.add_batch(predictions=predictions, references=references, **kwargs)

    @override
    @torch.inference_mode()
    def compute(self) -> torch.Tensor:
        """
        Compute the metric.

        The real metric result is in rank-0 device. For all other ranks, it will be zero
        """
        result = self._metric.compute()
        if result is not None:  # rank 0
            assert (
                self._metric_name in result
            ), f"Invalid result format: {result}. Expect key `{self._metric_name}`"
            result_metric = torch.FloatTensor([result[self._metric_name]])
            self.__setattr__(self._metric_name, result_metric)
        return result_metric

    @override
    @torch.inference_mode()
    def merge_state(self, metrics: Iterable[HFMetric]) -> Self:
        raise NotImplementedError(
            "Calling `merge_state() is forbidden in HFMetric. If you run HFMetric inside"
            "a MetricBag, set the `auto_sync` in the bag to True"
        )

    @override
    @torch.inference_mode()
    def reset(self) -> Self:
        self.__setattr__(self._metric_name, torch.zeros([]))

        # Reset the HF locks
        self._metric._finalize()
        if hasattr(self._metric, "filelock") and self.filelock is not None:
            self._metric.filelock.release()
        if (
            hasattr(self._metric, "rendez_vous_lock")
            and self.rendez_vous_lock is not None
        ):
            self._metric.rendez_vous_lock.release()
        self._metric.writer = None
        self._metric.data = None
