# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.data import create_bucket_sizes


def test_create_bucket_sizes() -> None:
    bucket_sizes = create_bucket_sizes(
        max_num_elements=16, max_seq_len=8, min_seq_len=2
    )

    assert bucket_sizes == [(8, 2), (5, 3), (4, 4), (3, 5), (2, 8)]
