# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Union, final

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

    def __init__(self, metric_name: str, device: Optional[Device] = None, **kwargs) -> None:  # type: ignore
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
            metric_name, torch.zeros((), device=device, dtype=torch.float32)
        )
        self.kwargs = kwargs

    @override
    @torch.inference_mode()
    def update(  # type: ignore
        self,
        predictions: Optional[Union[List[Any], torch.Tensor, numpy.ndarray]] = None,  # type: ignore
        references: Optional[Union[List[Any], torch.Tensor, numpy.ndarray]] = None,  # type: ignore
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
        result = self._metric.compute(**self.kwargs)
        if result is not None:  # rank 0
            assert (
                self._metric_name in result
            ), f"Invalid result format: {result}. Expect key `{self._metric_name}`"
            result_metric = torch.tensor(
                result[self._metric_name], device=self.device, dtype=torch.float32
            )
            self.__setattr__(self._metric_name, result_metric)
        else:
            result_metric = torch.zeros((), device=self.device, dtype=torch.float32)
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
    def reset(self) -> Self:  # type: ignore
        self.__setattr__(
            self._metric_name, torch.zeros((), device=self.device, dtype=torch.float32)
        )

        # Reset the HF locks
        self._metric._finalize()  # type: ignore
        if hasattr(self._metric, "filelock") and self.filelock is not None:  # type: ignore
            self._metric.filelock.release()  # type: ignore
        if (
            hasattr(self._metric, "rendez_vous_lock")
            and self.rendez_vous_lock is not None  # type: ignore
        ):
            self._metric.rendez_vous_lock.release()  # type: ignore
        self._metric.writer = None
        self._metric.data = None
        return self
