# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torcheval.metrics import Mean, Sum

from fairseq2.gang import FakeGang
from fairseq2.metrics import MetricBag
from fairseq2.metrics.hf import HFMetric
from tests.common import device


class TestMetricBag:
    def test_register_works(self) -> None:
        bag = MetricBag(gang=FakeGang(device=device))

        # Implicit
        bag.test1 = Sum(device=device)

        # Explicit
        bag.register_metric("test2", Mean(device=torch.device("cpu")))

        assert hasattr(bag, "test1")
        assert hasattr(bag, "test2")

        assert isinstance(bag.test1, Sum)
        assert isinstance(bag.test2, Mean)

        assert bag.test1.device == device
        assert bag.test2.device == device

    def test_getattr_raises_error_when_metric_is_missing(self) -> None:
        bag = MetricBag(gang=FakeGang(device=device))

        with pytest.raises(
            AttributeError, match=r"^`MetricBag` object has no attribute 'foo'\."
        ):
            bag.foo

    def test_state_dict_works(self) -> None:
        bag = MetricBag(gang=FakeGang(device=device))

        # Imlicit
        bag.test1 = Sum(device=device)

        # Explicit
        bag.register_metric("test2", Mean(device=device))
        bag.register_metric("test3", Mean(device=device), persistent=False)

        bag.test4 = Sum(device=device)

        del bag.test1

        state_dict = bag.state_dict()

        assert len(state_dict) == 2

        assert set(state_dict.keys()) == {"test2", "test4"}

    def test_load_state_dict_raises_error_when_state_dict_is_corrupt(self) -> None:
        state_dict = {"foo": 0}

        bag = MetricBag(gang=FakeGang(device=device))

        bag.test1 = Sum()
        bag.test2 = Sum()

        with pytest.raises(
            ValueError,
            match=r"^`state_dict` must contain metrics \['test1', 'test2'\], but contains \['foo'\] instead\.$",
        ):
            bag.load_state_dict(state_dict)

    def test_hf_metric(self) -> None:
        bag = MetricBag(gang=FakeGang(device=device))
        bag.hf_accuracy = HFMetric("accuracy")

        references = [[0, 1, 2], [0, 1], [2], [3]]
        predictions = [[0, 1, 1], [2, 1], [0], [3]]

        bag.begin_updates()
        for p, r in zip(predictions, references):
            bag.hf_accuracy.update(predictions=p, references=r)
        
        # Make sure auto_sync is set for HFMetric
        with pytest.raises(
            NotImplementedError, match=r"^Calling `merge_state() is forbidden in HFMetric"
        ):
            bag.sync_and_compute_metrics()
        
        bag.rollback_updates()
        
        # Add another metrics and properly set auto_sync flag
        bag.em = HFMetric("exact_match")
        bag.auto_sync = True

        bag.begin_updates()
        for p, r in zip(predictions, references):
            bag.hf_accuracy.update(predictions=p, references=r)
            bag.em.update(predictions=p, references=r)
        bag.commit_updates()
        
        result = bag.sync_and_compute_metrics()
        assert "accuracy" in result
        assert "exact_match" in result
