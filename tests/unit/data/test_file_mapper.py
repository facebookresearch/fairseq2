# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Final, Optional

import pytest

from fairseq2.data import FileMapper, FileMapperOutput

TEST_BIN_PATH: Final = Path(__file__).parent.joinpath("text", "test.spm")


class TestFileMapper:
    def test_call_works(self) -> None:
        mapper = FileMapper()

        for _ in range(2):
            output = mapper(TEST_BIN_PATH)

            assert output["path"] == TEST_BIN_PATH

            self.assert_file(output, TEST_BIN_PATH)

    def test_call_works_when_offset_is_specified(self) -> None:
        mapper = FileMapper()

        pathname = f"{TEST_BIN_PATH}:100"

        for _ in range(2):
            output = mapper(pathname)

            assert output["path"] == pathname

            self.assert_file(output, TEST_BIN_PATH, offset=100)

    def test_call_works_when_offset_and_size_are_specified(self) -> None:
        mapper = FileMapper()

        pathname = f"{TEST_BIN_PATH}:100:200"

        for _ in range(2):
            output = mapper(pathname)

            assert output["path"] == pathname

            self.assert_file(output, TEST_BIN_PATH, offset=100, size=200)

    def test_call_works_when_root_directory_is_specified(self) -> None:
        root_dir = TEST_BIN_PATH.parent.parent

        mapper = FileMapper(root_dir)

        pathname = TEST_BIN_PATH.relative_to(root_dir)

        output = mapper(pathname)

        assert output["path"] == pathname

        self.assert_file(output, TEST_BIN_PATH)

    @staticmethod
    def assert_file(
        output: FileMapperOutput,
        pathname: Path,
        offset: int = 0,
        size: Optional[int] = None,
    ) -> None:
        data = output["data"]

        if size is None:
            size = pathname.stat().st_size - offset

        assert len(data) == size

        with pathname.open(mode="rb") as fp:
            content = fp.read()

        assert content[offset : offset + size] == memoryview(data)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "value,type_name", [(None, "pyobj"), (123, "int"), (1.2, "float")]
    )
    def test_call_raises_error_when_input_is_not_string(
        self, value: Any, type_name: str
    ) -> None:
        mapper = FileMapper()

        with pytest.raises(
            ValueError,
            match=rf"^The input data must be of type `string`, but is of type `{type_name}` instead\.$",
        ):
            mapper(value)

    @pytest.mark.parametrize(
        "pathname",
        [
            "",
            "  ",
            ":",
            "::",
            "  : :",
            "foo:",
            "foo::",
            ":12:",
            ":12:34",
            "foo:12:34:56",
        ],
    )
    def test_call_raises_error_when_pathname_is_invalid(self, pathname: str) -> None:
        mapper = FileMapper()

        with pytest.raises(
            ValueError,
            match=rf"^The input string must be a pathname with optional offset and size specifiers, but is '{pathname}' instead\.$",
        ):
            mapper(pathname)

    @pytest.mark.parametrize("offset", ["ab", "12a", "a12", "12 34"])
    def test_call_raises_error_when_offset_is_invalid(self, offset: str) -> None:
        mapper = FileMapper()

        pathname = f"foo:{offset}:34"

        with pytest.raises(
            ValueError,
            match=rf"^The offset specifier of '{pathname}' must be an integer, but is '{offset}' instead\.$",
        ):
            mapper(pathname)

    def test_call_raises_error_when_offset_is_out_of_range(self) -> None:
        mapper = FileMapper()

        offset = 9999999999999999999999999

        pathname = f"foo:{offset}:34"

        with pytest.raises(
            ValueError,
            match=rf"^The offset specifier of '{pathname}' must be a machine-representable integer, but is '{offset}' instead, which is out of range\.$",
        ):
            mapper(pathname)

    @pytest.mark.parametrize("size", ["ab", "12a", "a12", "12 34"])
    def test_call_raises_error_when_size_is_invalid(self, size: str) -> None:
        mapper = FileMapper()

        pathname = f"foo:12:{size}"

        with pytest.raises(
            ValueError,
            match=rf"^The size specifier of '{pathname}' must be an integer, but is '{size}' instead\.$",
        ):
            mapper(pathname)

    def test_call_raises_error_when_size_is_out_of_range(self) -> None:
        mapper = FileMapper()

        size = 9999999999999999999999999

        pathname = f"foo:12:{size}"

        with pytest.raises(
            ValueError,
            match=rf"^The size specifier of '{pathname}' must be a machine-representable integer, but is '{size}' instead, which is out of range\.$",
        ):
            mapper(pathname)

    def test_call_raises_error_when_offset_is_larger_than_file_size(self) -> None:
        mapper = FileMapper()

        file_size = TEST_BIN_PATH.stat().st_size

        pathname = f"{TEST_BIN_PATH}:{file_size + 1}"

        with pytest.raises(
            ValueError,
            match=rf"^The specified offset within '{pathname}' must be less than or equal to the file size \({file_size:,} bytes\), but is {file_size + 1:,} instead\.$",
        ):
            mapper(pathname)

    def test_call_raises_error_when_offset_plus_size_is_larger_than_file_size(
        self,
    ) -> None:
        mapper = FileMapper()

        file_size = TEST_BIN_PATH.stat().st_size

        pathname = f"{TEST_BIN_PATH}:{file_size - 10}:11"

        with pytest.raises(
            ValueError,
            match=rf"^The end of the specified region within '{pathname}' must be less than or equal to the file size \({file_size:,} bytes\), but is {file_size + 1:,} instead\.$",
        ):
            mapper(pathname)
