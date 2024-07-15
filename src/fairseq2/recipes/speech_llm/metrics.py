from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torcheval.metrics import Throughput

from fairseq2.gang import Gang
from fairseq2.generation import Seq2SeqGeneratorOutput, SequenceGeneratorOutput
from fairseq2.metrics import MetricBag
from fairseq2.metrics.aggregation import Max, Mean, Sum
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch


class PPLEvalMetricBag(MetricBag):
    """Holds the metrics of a sequence model training or evaluation task."""
    _nll: Mean
    _ppl: Mean
    _batch_size: Mean
    _elements_per_batch: Mean
    _num_examples: Sum
    _num_elements: Sum
    _num_target_elements: Sum
    _total_num_examples: Sum
    _total_num_elements: Sum
    _total_num_target_elements: Sum

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang over which to sync metrics.
        """
        super().__init__(gang)

        d = gang.device

        self.register_metric("_ppl", Sum(device=d), persistent=False)
        self.register_metric("_nll", Mean(device=d), persistent=False)

        self.register_metric("_batch_size", Mean(device=d), persistent=False)

        self.register_metric("_elements_per_batch", Mean(device=d), persistent=False)

        self.register_metric("_num_examples", Sum(device=d), persistent=False)

        self.register_metric("_num_elements", Sum(device=d), persistent=False)

        self.register_metric("_num_target_elements", Sum(device=d), persistent=False)

        self._total_num_examples = Sum(device=d)

        self._total_num_elements = Sum(device=d)

        self._total_num_target_elements = Sum(device=d)


    @torch.inference_mode()
    def update_nll(self, nll: float, num_target_elements: int) -> None:
        self._ppl.update(nll)
        self._nll.update(nll / num_target_elements, weight=num_target_elements)
        self._total_num_target_elements.update(num_target_elements)
        self._num_target_elements.update(num_target_elements)



    @torch.inference_mode()
    def update_batch_metrics(self, batch: SequenceBatch) -> None:
        """Update the batch metrics.

        :param batch:
            The batch processed by the model.
        """
        batch_size = batch.batch_size
        num_elements = batch.num_elements()

        self._batch_size.update(batch_size * self._gang.size)

        self._elements_per_batch.update(num_elements * self._gang.size)

        self._num_examples.update(batch_size)

        self._num_elements.update(num_elements)

        self._total_num_examples.update(batch_size)

        self._total_num_elements.update(num_elements)

        


class MMLUEvalMetricBag(MetricBag):
    """Holds the metrics of a sequence model training or evaluation task."""
    _nll: Mean
    _acc: Mean

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang over which to sync metrics.
        """
        super().__init__(gang)

        d = gang.device

        self.register_metric("_acc", Mean(device=d), persistent=False)
        self.register_metric("_nll", Mean(device=d), persistent=False)


    @torch.inference_mode()
    def update_nll(self, acc: float, nll: float) -> None:
        self._acc.update(acc, weight=1)
        self._nll.update(nll, weight=1)



 