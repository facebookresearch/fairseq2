# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from fairseq2.gang import (
    FakeGang,
    Gangs,
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


def _gangs_with_ranks(*, dp: int = 0, tp: int = 0, pp: int = 0) -> Gangs:
    # The root gang's coordinator is rank 0; build each parallel gang as a
    # `FakeGang` at the requested rank, with `size` just large enough to make
    # that rank valid.
    root = FakeGang(device=device, rank=0, size=1)

    def gang(rank: int) -> FakeGang:
        return FakeGang(device=device, rank=rank, size=rank + 1)

    dp_gang = gang(dp)

    return Gangs(
        root=root,
        dp=dp_gang,
        rdp=dp_gang,
        sdp=dp_gang,
        tp=gang(tp),
        pp=gang(pp),
    )


def test_gangs_accept_coordinator_that_is_rank_0_in_every_parallel_gang() -> None:
    # `root.rank == 0` and dp/tp/pp all rank 0 is a valid coordinator; must not
    # raise.
    _gangs_with_ranks(dp=0, tp=0, pp=0)


@pytest.mark.parametrize(
    "dp,tp,pp",
    [
        (1, 0, 0),  # data parallel rank non-zero
        (0, 1, 0),  # tensor parallel rank non-zero
        (0, 0, 1),  # pipeline parallel rank non-zero
        (0, 1, 1),  # tensor and pipeline parallel ranks non-zero
    ],
)
def test_gangs_reject_coordinator_with_a_non_zero_parallel_rank(
    dp: int, tp: int, pp: int
) -> None:
    # A `root.rank == 0` process that is not rank 0 in *every* parallel gang is
    # an invalid coordinator and must be rejected. The `(0, 1, 0)` and
    # `(0, 0, 1)` cases in particular are the ones an `and`/`or` precedence bug
    # in the check would let slip through.
    with pytest.raises(ValueError, match="must be rank 0 in all parallel gangs"):
        _gangs_with_ranks(dp=dp, tp=tp, pp=pp)
