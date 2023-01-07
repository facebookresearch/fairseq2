# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["StringLike"]

from typing import Union

from typing_extensions import TypeAlias

from fairseq2.data.string import String

StringLike: TypeAlias = Union[str, String]
