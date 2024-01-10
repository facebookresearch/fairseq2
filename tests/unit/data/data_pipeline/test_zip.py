# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import DataPipeline, DataPipelineError, read_sequence
from fairseq2.data.text import read_text


class TestZipOp:
    def test_op_works(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6, 7, 8]).and_return()
        pipeline3 = read_sequence([0, 2, 4, 6]).and_return()

        pipeline = DataPipeline.zip([pipeline1, pipeline2, pipeline3]).and_return()

        for _ in range(2):
            assert list(pipeline) == [[1, 5, 0], [2, 6, 2], [3, 7, 4], [4, 8, 6]]

            pipeline.reset()

    def test_op_works_when_a_single_pipeline_is_specified(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()

        pipeline = DataPipeline.zip([pipeline1]).and_return()

        for _ in range(2):
            assert list(pipeline) == [[1], [2], [3], [4]]

            pipeline.reset()

    def test_op_works_when_no_pipeline_is_specified(self) -> None:
        pipeline = DataPipeline.zip([]).and_return()

        for _ in range(2):
            with pytest.raises(StopIteration):
                next(iter(pipeline))

            pipeline.reset()

    def test_op_works_when_infinite_pipeline_is_specified(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = DataPipeline.constant(0).and_return()
        pipeline3 = read_sequence([5, 6, 7, 8]).and_return()

        pipeline = DataPipeline.zip([pipeline1, pipeline2, pipeline3]).and_return()

        for _ in range(2):
            assert list(pipeline) == [[1, 0, 5], [2, 0, 6], [3, 0, 7], [4, 0, 8]]

            pipeline.reset()

    def test_op_works_when_names_are_specified(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6, 7, 8]).and_return()
        pipeline3 = read_sequence([0, 2, 4, 6]).and_return()

        pipeline = DataPipeline.zip(
            [pipeline1, pipeline2, pipeline3], names=["p1", "p2", "p3"]
        ).and_return()

        for _ in range(2):
            assert list(pipeline) == [
                {"p1": 1, "p2": 5, "p3": 0},
                {"p1": 2, "p2": 6, "p3": 2},
                {"p1": 3, "p2": 7, "p3": 4},
                {"p1": 4, "p2": 8, "p3": 6},
            ]

            pipeline.reset()

    def test_op_works_when_zip_to_shortest_is_true(self) -> None:
        pipeline1 = read_sequence([1, 2, 3]).and_return()
        pipeline2 = read_sequence([5, 6, 7, 8]).and_return()
        pipeline3 = read_sequence([3, 4, 5, 6]).and_return()
        pipeline4 = DataPipeline.count(1).and_return()

        pipeline = DataPipeline.zip(
            [pipeline1, pipeline2, pipeline3, pipeline4], zip_to_shortest=True
        ).and_return()

        for _ in range(2):
            assert list(pipeline) == [[1, 5, 3, 1], [2, 6, 4, 2], [3, 7, 5, 3]]

            pipeline.reset()

    def test_op_works_when_flatten_is_true_and_inputs_are_lists(self) -> None:
        pipeline1 = read_sequence([1, 2, 3]).and_return()
        pipeline2 = read_sequence([[1, 2], [3, 4], [5, 6]]).and_return()
        pipeline3 = read_sequence([[4], [5], [6]]).and_return()

        pipeline = DataPipeline.zip(
            [pipeline1, pipeline2, pipeline3], flatten=True
        ).and_return()

        for _ in range(2):
            assert list(pipeline) == [[1, 1, 2, 4], [2, 3, 4, 5], [3, 5, 6, 6]]

            pipeline.reset()

    def test_op_works_when_flatten_is_true_and_inputs_are_dicts(self) -> None:
        pipeline1 = read_sequence([{"foo1": 1}, {"foo1": 2}, {"foo1": 3}]).and_return()
        pipeline2 = read_sequence([{"foo2": 4, "foo3": 5}, {"foo2": 6, "foo3": 7}, {"foo2": 8, "foo3": 9}]).and_return()  # fmt: skip
        pipeline3 = read_sequence([{"foo4": 2}, {"foo4": 3}, {"foo4": 4}]).and_return()

        pipeline = DataPipeline.zip(
            [pipeline1, pipeline2, pipeline3], flatten=True
        ).and_return()

        for _ in range(2):
            assert list(pipeline) == [
                {"foo1": 1, "foo2": 4, "foo3": 5, "foo4": 2},
                {"foo1": 2, "foo2": 6, "foo3": 7, "foo4": 3},
                {"foo1": 3, "foo2": 8, "foo3": 9, "foo4": 4},
            ]

            pipeline.reset()

    def test_op_raises_error_when_pipelines_have_different_lengths(self) -> None:
        pipeline1 = read_sequence([1, 2, 3]).and_return()
        pipeline2 = read_sequence([5, 6, 7, 8]).and_return()
        pipeline3 = read_sequence([5, 6, 7, 8]).and_return()

        pipeline = DataPipeline.zip([pipeline1, pipeline2, pipeline3]).and_return()

        with pytest.raises(
            DataPipelineError,
            match=r"^The zipped data pipelines must all have the same number of examples, but the data pipelines at the indices \[1, 2\] have more examples than the others\.$",
        ):
            for d in pipeline:
                pass

    def test_op_raises_error_when_the_number_of_pipelines_and_names_do_not_match(
        self,
    ) -> None:
        pipeline1 = read_sequence([]).and_return()
        pipeline2 = read_sequence([]).and_return()

        with pytest.raises(
            ValueError,
            match=r"^The number of `pipelines` and the number of `names` must be equal, but are 2 and 3 instead\.$",
        ):
            DataPipeline.zip([pipeline1, pipeline2], ["p1", "p2", "p3"])

    def test_op_raises_error_when_one_of_the_pipelines_is_broken(self) -> None:
        # Force a non-recoverable error.
        pipeline1 = read_text(pathname=" &^#").and_return()
        pipeline2 = read_text(pathname=" &^#").and_return()

        # Break the first pipeline.
        try:
            next(iter(pipeline1))
        except DataPipelineError:
            assert pipeline1.is_broken

        with pytest.raises(
            ValueError,
            match=r"^At least one of the specified data pipelines is broken and cannot be zipped\.$",
        ):
            DataPipeline.zip([pipeline1, pipeline2]).and_return()

    def test_op_raises_error_when_both_names_and_flatten_are_specified(self) -> None:
        pipeline1 = read_sequence([1]).and_return()
        pipeline2 = read_sequence([1]).and_return()

        with pytest.raises(
            ValueError,
            match=r"^`names` and `flatten` are mutually exclusive and cannot be specified at the same time\.$",
        ):
            DataPipeline.zip([pipeline1, pipeline2], names=["foo"], flatten=True)

    def test_op_raises_error_when_flatten_is_true_and_input_has_both_list_and_dict(
        self,
    ) -> None:
        pipeline1 = read_sequence([1]).and_return()
        pipeline2 = read_sequence([{"foo1": 1}]).and_return()

        pipeline = DataPipeline.zip([pipeline1, pipeline2], flatten=True).and_return()

        with pytest.raises(
            DataPipelineError,
            match=r"^The zipped data pipelines must all return only dicts, or only non-dicts when `flatten` is set\.$",
        ):
            next(iter(pipeline))

        pipeline1 = read_sequence([{"foo1": 1}]).and_return()
        pipeline2 = read_sequence([1]).and_return()

        pipeline = DataPipeline.zip([pipeline1, pipeline2], flatten=True).and_return()

        with pytest.raises(
            DataPipelineError,
            match=r"^The zipped data pipelines must all return only dicts, or only non-dicts when `flatten` is set\.$",
        ):
            next(iter(pipeline))

    def test_op_raises_error_when_flatten_is_true_and_dict_keys_are_not_unique(
        self,
    ) -> None:
        pipeline1 = read_sequence([{"foo1": 1}]).and_return()
        pipeline2 = read_sequence([{"foo2": 1}]).and_return()
        pipeline3 = read_sequence([{"foo1": 1}]).and_return()

        pipeline = DataPipeline.zip(
            [pipeline1, pipeline2, pipeline3], flatten=True
        ).and_return()

        with pytest.raises(
            DataPipelineError,
            match=r"^The zipped data pipelines must all return unique keys when `flatten` is set, but the key 'foo1' is not unique\.$",
        ):
            next(iter(pipeline))

    def test_op_saves_and_restores_its_state(self) -> None:
        pipeline1 = read_sequence([1, 2, 3, 4]).and_return()
        pipeline2 = read_sequence([5, 6, 7, 8]).and_return()
        pipeline3 = read_sequence([0, 2, 4, 6]).and_return()

        pipeline = DataPipeline.zip([pipeline1, pipeline2, pipeline3]).and_return()

        d = None

        it = iter(pipeline)

        # Move to the second example.
        for _ in range(2):
            d = next(it)

        assert d == [2, 6, 2]

        state_dict = pipeline.state_dict()

        # Read one more example before we roll back.
        d = next(it)

        assert d == [3, 7, 4]

        # Expected to roll back to the second example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(2):
            d = next(it)

        assert d == [4, 8, 6]

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))
