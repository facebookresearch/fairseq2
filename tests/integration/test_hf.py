# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.gang import FakeGang
from fairseq2.metrics import MetricBag
from fairseq2.recipes.hf.metrics import HFMetric
from tests.common import device


class TestMetric:
    def test_hf_metric(self) -> None:
        bag = MetricBag(gang=FakeGang(device=device))
        bag.hf_accuracy = HFMetric("accuracy")

        # All compute arguments are registered in the beginning
        bag.f1 = HFMetric("f1", average="macro")

        references = [[0, 1, 2], [0, 1], [2], [3]]
        predictions = [[0, 1, 1], [2, 1], [0], [3]]

        bag.begin_updates()
        for p, r in zip(predictions, references):
            bag.hf_accuracy.update(predictions=p, references=r)
            bag.f1.update(predictions=p, references=r)
        bag.commit_updates()

        bag.auto_sync = True
        result = bag.sync_and_compute_metrics()
        assert result
        assert (
            "hf_accuracy" in result
            and pytest.approx(result["hf_accuracy"].item(), 0.0001) == 0.5714
        )
        assert "f1" in result and pytest.approx(result["f1"].item(), 0.001) == 0.575
