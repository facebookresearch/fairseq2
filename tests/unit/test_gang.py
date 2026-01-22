# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.gang import (
    FakeGang,
    create_fake_gangs,
    get_current_gangs,
    get_default_gangs,
    set_default_gangs,
    set_gangs,
)
from tests.common import device


def test_get_current_gangs_works() -> None:
    fake_gangs1 = create_fake_gangs(device)
    fake_gangs2 = create_fake_gangs(device)
    fake_gangs3 = create_fake_gangs(device)

    default_gangs1 = get_current_gangs()
    default_gangs2 = get_current_gangs()

    assert default_gangs1 is default_gangs2

    assert isinstance(default_gangs1.root, FakeGang)

    assert default_gangs1.root.rank == 0
    assert default_gangs1.root.size == 1

    set_default_gangs(fake_gangs1)

    with device, fake_gangs2:
        assert get_current_gangs() is fake_gangs2
        assert get_default_gangs() is fake_gangs1

        with set_gangs(fake_gangs3):
            assert get_current_gangs() is fake_gangs3
            assert get_default_gangs() is fake_gangs1

        assert get_current_gangs() is fake_gangs2
        assert get_default_gangs() is fake_gangs1

    assert get_current_gangs(device) is fake_gangs1
    assert get_default_gangs(device) is fake_gangs1
