# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.cli.metrics import Bleu
from tests.common import assert_close


def test_bleu() -> None:
    bleu = Bleu()
    bleu.update("a b c d", ["a b d"])

    # hyp_len, ref_len, *correct, *total
    assert_close(bleu.bleu_counts, [4, 3, 3, 1, 0, 0, 4, 3, 2, 1])

    assert round(bleu.compute().item()) == 35
