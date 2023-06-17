# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torch.nn import Linear

from fairseq2.nn import ModuleList


class TestModuleList:
    def test_iter_returns_no_modules_when_drop_p_is_one(self) -> None:
        modules = [Linear(10, 10), Linear(10, 10), Linear(10, 10), Linear(10, 10)]

        m = ModuleList(modules, drop_p=1.0)

        with pytest.raises(StopIteration):
            next(m.drop_iter())

    def test_iter_returns_all_modules_when_drop_p_is_zero(self) -> None:
        modules = [Linear(10, 10), Linear(10, 10), Linear(10, 10), Linear(10, 10)]

        m = ModuleList(modules)

        count = 0

        for m1, m2 in zip(m.drop_iter(), modules):
            assert m1 is m2

            count += 1

        assert count == len(modules)

    def test_iter_returns_all_modules_in_eval(self) -> None:
        modules = [Linear(10, 10), Linear(10, 10), Linear(10, 10), Linear(10, 10)]

        m = ModuleList(modules, drop_p=1.0)

        m.eval()

        count = 0

        for m1, m2 in zip(m.drop_iter(), modules):
            assert m1 is m2

            count += 1

        assert count == len(modules)
