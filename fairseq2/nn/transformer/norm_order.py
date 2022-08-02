# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class TransformerNormOrder(Enum):
    """Specifies the Layer Normalization order of a Transformer model."""

    POST = 0
    """Apply Layer Normalization after each layer's residual connection as
    described in :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`."""
    PRE = 1
    """Apply Layer Normalization at the beginning of each layer as described
    in :cite:t:`DBLP:journals/corr/abs-2002-04745`."""
    PRE_WITH_NORMFORMER = 2
    """Apply Layer Normalization as described in
    :cite:t:`DBLP:journals/corr/abs-2110-09456`."""
