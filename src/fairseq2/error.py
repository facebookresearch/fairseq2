# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


class InternalError(Exception):
    pass


class ContractError(Exception):
    pass


class InvalidOperationError(Exception):
    pass


class AlreadyExistsError(Exception):
    pass


class NotSupportedError(Exception):
    pass
