# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import DataPipelineError, get_last_failed_example, read_sequence


class TestDataPipeline:
    def test_next_sets_last_failed_example(self) -> None:
        def fn(d: int) -> bool:
            if d == 3:
                raise ValueError("foo")

            return True

        pipeline = read_sequence([3, 4]).filter(fn).and_return()

        it = iter(pipeline)

        with pytest.raises(DataPipelineError):
            next(it)

        assert not pipeline.is_broken

        example = get_last_failed_example()

        assert isinstance(example, int)

        assert example == 3

        # We expect the error state to be cleared after a succesful operation.
        next(it)

        example = get_last_failed_example()

        assert example is None

    def test_next_works_when_error_is_recoverable(self) -> None:
        def fn(d: int) -> bool:
            if d == 2 or d == 4:
                # Errors caused by the filter callable are always treated as
                # recoverable.
                raise ValueError("foo")

            return True

        seq = [1, 2, 3, 4, 5]

        pipeline = read_sequence(seq).filter(fn).and_return()

        output = []

        it = iter(pipeline)

        while True:
            try:
                output.append(next(it))
            except DataPipelineError:
                assert not pipeline.is_broken
            except StopIteration:
                break

        assert output == [1, 3, 5]

    def test_next_does_not_raise_error_when_num_errors_is_less_than_max_num_warnings(
        self,
    ) -> None:
        def fn(d: int) -> bool:
            if d == 3 or d == 5:
                raise ValueError("foo")

            return True

        seq = list(range(1, 9))

        pipeline = read_sequence(seq).filter(fn).and_return(max_num_warnings=3)

        assert list(pipeline) == [1, 2, 4, 6, 7, 8]

        # TODO: assert log warning

    @pytest.mark.parametrize("max_num_warnings", [0, 1, 2])
    def test_next_raises_error_when_num_errors_exceed_max_num_warnings(
        self, max_num_warnings: int
    ) -> None:
        run_count = 0

        def fn(d: int) -> bool:
            nonlocal run_count

            run_count += 1

            if d < 4:
                raise ValueError("foo")

            return True

        seq = [1, 2, 3, 4, 5]

        pipeline = read_sequence(seq).filter(fn).and_return(max_num_warnings)

        with pytest.raises(DataPipelineError):
            for _ in pipeline:
                pass

        assert run_count == max_num_warnings + 1

    def test_load_state_dict_raises_error_when_tape_is_corrupt(self) -> None:
        seq = [1, 2, 3, 4, 5]

        pipeline = read_sequence(seq).and_return()

        next(iter(pipeline))

        state_dict = pipeline.state_dict()

        # Deliberately corrupt the underlying tape.
        state_dict["position"].append("foo")

        for s in [{}, {"position": "foo"}, state_dict]:
            with pytest.raises(
                ValueError,
                match=r"^`state_dict` must contain a valid data pipeline state, but cannot be parsed as such\.$",
            ):
                pipeline.load_state_dict(s)  # type: ignore[arg-type]
