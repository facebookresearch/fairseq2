# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from typing import Optional, get_type_hints

import pytest

from fairseq2.runtime.dependency import _strip_optional


class CustomSimpleClass:
    """A custom simple class for testing."""
    pass


class TestClass:
    """Test class with various type hints for testing _strip_optional
    in the same conditions as they will be used in `_create_auto_wired_instance`."""

    def method_with_int(self, param: int) -> None:
        pass
    
    def method_with_custom_class(self, param: CustomSimpleClass) -> None:
        pass
    
    def method_with_optional_custom_class(self, param: Optional[CustomSimpleClass]) -> None:
        pass
    
    def method_with_union_custom_class_none(self, param: CustomSimpleClass | None) -> None:
        pass
    
    def method_with_union_three_types(self, param: int | bool | None) -> None:
        pass
    
    def method_with_union_two_types_no_none(self, param: int | str) -> None:
        pass


class TestStripOptional:
    """Test cases for the _strip_optional function."""
    
    @pytest.mark.parametrize("method_name,expected_optional,expected_type", [
        ("method_with_int", False, int),
        ("method_with_custom_class", False, CustomSimpleClass),
        ("method_with_optional_custom_class", True, CustomSimpleClass),
        ("method_with_union_custom_class_none", True, CustomSimpleClass),
    ])
    def test_plain_and_optional_types(self, method_name, expected_optional, expected_type):
        """Test with plain non-optional types and Optional types."""
        method = getattr(TestClass, method_name)
        type_hints = get_type_hints(method)
        param_type = type_hints["param"]

        is_optional, stripped_type = _strip_optional(param_type)
        
        assert is_optional is expected_optional
        assert stripped_type is expected_type
    
    @pytest.mark.parametrize("method_name", [
        "method_with_union_three_types",
        "method_with_union_two_types_no_none",
    ])
    def test_union_types_not_optional(self, method_name):
        """Test with unions that should not be considered optional."""
        method = getattr(TestClass, method_name)
        type_hints = get_type_hints(method)
        param_type = type_hints['param']
        
        is_optional, stripped_type = _strip_optional(param_type)
        
        # Should return False because either no None or multiple non-None types
        assert is_optional is False
        assert stripped_type == param_type  # original type
