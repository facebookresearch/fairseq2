# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.gang import create_fake_gangs, maybe_get_current_gangs
from tests.common import device


def test_maybe_get_current_gangs_works() -> None:
    gangs1 = create_fake_gangs(device)
    gangs2 = create_fake_gangs(device)

    assert maybe_get_current_gangs() is None

    with gangs1:
        assert maybe_get_current_gangs() is gangs1

        with gangs2:
            assert maybe_get_current_gangs() is gangs2

        assert maybe_get_current_gangs() is gangs1

    assert maybe_get_current_gangs() is None
