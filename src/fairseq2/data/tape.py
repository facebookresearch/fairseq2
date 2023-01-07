# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["Tape"]

from fairseq2._C.data.tape import Tape

Tape.__module__ = __name__
