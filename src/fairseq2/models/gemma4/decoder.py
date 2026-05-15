# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Re-export :class:`Gemma4Decoder` from the factory module.

The canonical definition lives in :mod:`~fairseq2.models.gemma4.factory` because
the decoder is tightly coupled to factory construction and KV sharing logic.
This module provides a convenient top-level import path.
"""

from fairseq2.models.gemma4.factory import Gemma4Decoder as Gemma4Decoder

__all__ = ["Gemma4Decoder"]
