# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator

import pytest

from fairseq2.data.data_pipeline import read_iterator
from fairseq2.data.utils import read_pickle_wrapped_iterator


def example_generator() -> Iterator[int]:
    for i in range(10):
        yield i


class TestReadAndPickleWrapIterator:
    def test_read_and_pickle_wrap_iterator_works(self) -> None:
        with pytest.raises(TypeError):
            read_iterator(
                example_generator(),
                reset_fn=lambda x: example_generator(),
                infinite=False,
            ).and_return()

        pipeline = read_pickle_wrapped_iterator(example_generator).and_return()

        it = iter(pipeline)

        assert next(it) == 0
        assert next(it) == 1

        state = pipeline.state_dict()

        assert next(it) == 2
        assert next(it) == 3
        assert next(it) == 4

        pipeline.load_state_dict(state)

        assert next(it) == 2
        assert next(it) == 3
        assert next(it) == 4

        pipeline.reset()

        for _ in range(2):
            assert list(pipeline) == [*range(10)]
            pipeline.reset()
