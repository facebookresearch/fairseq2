"""KV projection sharing types for Gemma3n."""

from enum import Enum
from typing import Final


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
