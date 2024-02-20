# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.data import DataPipeline


class TestConstantOp:
    def test_op_works(self) -> None:
        pipeline = DataPipeline.constant("foo").take(10).and_return()

        for _ in range(2):
            list(pipeline) == ["foo"] * 10

            pipeline.reset()
