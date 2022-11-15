# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.data.interop import IDict, IList, IString
from fairseq2.data.variant import IVariant

IVariant = IVariant
"""Holds data of different types that can be zero-copy marshalled between Python
and native code."""

__all__ = ["IDict", "IList", "IString", "IVariant"]
