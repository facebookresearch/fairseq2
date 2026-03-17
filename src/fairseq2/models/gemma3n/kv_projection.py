# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KV projection sharing types for Gemma3n."""

from enum import Enum


class KVProjectionType(Enum):
    """Types of attention layers for KV projection sharing."""

    LOCAL = "local"
    """Local attention layers (sliding window)."""

    GLOBAL = "global"
    """Global attention layers (full causal)."""


class KVProjectionRole(Enum):
    """Role of a layer in KV projection sharing."""

    SOURCE = "source"
    """Layer computes and stores K/V projections for consumers."""

    CONSUMER = "consumer"
    """Layer retrieves pre-computed K/V projections from source."""

    NONE = "none"
    """Layer does not participate in KV projection sharing."""
