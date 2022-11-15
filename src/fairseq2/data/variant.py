# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

from torch import Tensor
from typing_extensions import TypeAlias

from fairseq2.data.interop import IDict, IList, IString

IVariant: TypeAlias = Union[None, bool, int, float, str, IString, Tensor, IList, IDict]
