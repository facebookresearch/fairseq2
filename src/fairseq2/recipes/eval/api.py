# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from numpy.random import RandomState

from fairseq2.data.text.text_tokenizer import TextTokenDecoder
from fairseq2.generation.generator import SequenceGenerator


# TODO: Replace Any wih non-text decoder such as Diffusion-based models
Decoder = Union[TextTokenDecoder, Any]

Example = Dict[str, Any]
ExampleFn = Callable[[Example], Example]
MetricFn = Callable[[Example], Dict[str, float]]


@dataclass
class TaskConfig: ...


@dataclass
class AverageMetric:
    """
    Average metric with confidence interval.

    avg is the mean of a list of values
    count is the length of this list
    square is the mean of the squares of the values
    avg_ci_fn is a function applied to the bounds of the confidence interval
    """

    avg: float
    count: int
    square: float
    avg_ci_fn: Optional[Callable] = None

    @property
    def value(self):
        return self.avg_ci_fn(self.avg) if self.avg_ci_fn else self.avg

    def update(self, value: float, count: int, square: Optional[float] = None) -> None:
        self.avg = (self.avg * self.count + value * count) / (self.count + count)
        if square is None:
            assert count == 1
            square = value**2
        self.square = (self.square * self.count + square * count) / (self.count + count)
        self.count += count

    def compute_ci(
        self, confidence_level: float = 0.95
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        returns bounds of confidence interval: ci_lb ('lower_bound') and ci_ub ('upper_bound').
        Confidence interval is computed with error margins:
        z * s / sqrt(n), where:
        - P(-z <= X <= z) = confidence_level and X follows a t student low with self.count - 1 parameters.
        - s is the unbiased std estimate: (1/(n-1) sum((xi - mean(xi) ** 2))) ** 0.5

        example: first 100 integers as metric_values and confidence_level = 0.95:
        >>> avg_m = AverageMetric(0, 0, 0)
        >>> for i in range(100):
        >>>     avg_m.update(value=i, count=1)
        >>> avg_m.compute_ci() # mean is 49.5, std is 29.0115, self.count = 100, z = 1.98
        >>> (43.743, 55.257)
        """
        from scipy.stats import t

        if self.count < 2:
            return None, None

        std = (self.count / (self.count - 1) * (self.square - (self.avg) ** 2)) ** 0.5
        scale = std / (self.count**0.5)
        lb, ub = t.interval(confidence_level, self.count - 1, loc=self.avg, scale=scale)
        if self.avg_ci_fn:
            lb, ub = self.avg_ci_fn(lb), self.avg_ci_fn(ub)
        return (lb, ub)


@dataclass
class TaskResult:
    metrics: Dict[str, AverageMetric]
    raw_results: Optional[List[Dict[str, Any]]] = None


class Task(abc.ABC):
    @abc.abstractmethod
    def run(
        self,
        generator: SequenceGenerator,
        decoder: Optional[Decoder] = None,
        random_state: Optional[RandomState] = None,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TaskResult:
        """Run the task for a given predictor"""
        ...
