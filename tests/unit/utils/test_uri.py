# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from fairseq2.error import NotSupportedError
from fairseq2.utils.uri import Uri, UriFormatError


class TestUri:
    def test_parse_works(self) -> None:
        uri = Uri.parse("https://foo.com/foo2/foo3")

        assert uri.scheme == "https"
        assert uri.netloc == "foo.com"
        assert uri.path == "/foo2/foo3"
        assert uri.params == ""
        assert uri.query == ""

    def test_parse_works_when_params_specified(self) -> None:
        uri = Uri.parse("https://foo.com/foo2/foo3;a = 1; bc =2; de= 4?x=3&y=4")

        assert uri.scheme == "https"
        assert uri.netloc == "foo.com"
        assert uri.path == "/foo2/foo3"
        assert uri.params == "a = 1; bc =2; de= 4"
        assert uri.query == "x=3&y=4"

    def test_parse_raises_error_when_input_has_no_scheme(self) -> None:
        with pytest.raises(
            UriFormatError, match=r"^/foo1/foo2 does not have a URI scheme\.$"
        ):
            Uri.parse("/foo1/foo2")

    def test_to_path_works(self) -> None:
        uri = Uri.parse("file:///root/sub1/sub2")

        path = uri.to_path()

        assert str(path) == "/root/sub1/sub2"

    def test_to_path_raises_error_when_scheme_is_not_file(self) -> None:
        uri = Uri.parse("https://foo.com")

        with pytest.raises(
            NotSupportedError, match=r"^`to_path\(\)` is only supported for URIs with file scheme\.$"  # fmt: skip
        ):
            uri.to_path()

    def test_str_works(self) -> None:
        s = "https://foo.com/foo2/foo3;a = 1; bc =2; de= 4?x=3&y=4"

        uri = Uri.parse(s)

        assert str(uri) == s
