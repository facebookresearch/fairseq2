# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.nn._embedding import Embedding as Embedding
from fairseq2.nn._embedding import ShardedEmbedding as ShardedEmbedding
from fairseq2.nn._embedding import StandardEmbedding as StandardEmbedding
from fairseq2.nn._embedding import VocabShardedEmbedding as VocabShardedEmbedding
from fairseq2.nn._embedding import init_scaled_embedding as init_scaled_embedding
from fairseq2.nn._incremental_state import IncrementalState as IncrementalState
from fairseq2.nn._incremental_state import IncrementalStateBag as IncrementalStateBag
from fairseq2.nn._normalization import LayerNorm as LayerNorm
from fairseq2.nn._normalization import RMSNorm as RMSNorm
from fairseq2.nn._normalization import StandardLayerNorm as StandardLayerNorm
from fairseq2.nn._position_encoder import (
    InterpolatedPositionEncoder as InterpolatedPositionEncoder,
)
from fairseq2.nn._position_encoder import (
    LearnedPositionEncoder as LearnedPositionEncoder,
)
from fairseq2.nn._position_encoder import PositionEncoder as PositionEncoder
from fairseq2.nn._position_encoder import RotaryEncoder as RotaryEncoder
from fairseq2.nn._position_encoder import ReferenceRotaryEncoder as ReferenceRotaryEncoder
from fairseq2.nn._position_encoder import (
    Sinusoidal2dPositionEncoder as Sinusoidal2dPositionEncoder,
)
from fairseq2.nn._position_encoder import (
    Sinusoidal3dPositionEncoder as Sinusoidal3dPositionEncoder,
)
from fairseq2.nn._position_encoder import (
    SinusoidalNdPositionEncoder as SinusoidalNdPositionEncoder,
)
from fairseq2.nn._position_encoder import (
    SinusoidalPositionEncoder as SinusoidalPositionEncoder,
)
from fairseq2.nn._projection import ColumnShardedLinear as ColumnShardedLinear
from fairseq2.nn._projection import IdentityProjection as IdentityProjection
from fairseq2.nn._projection import Linear as Linear
from fairseq2.nn._projection import Projection as Projection
from fairseq2.nn._projection import RowShardedLinear as RowShardedLinear
from fairseq2.nn._projection import TiedProjection as TiedProjection
from fairseq2.nn._projection import init_bert_projection as init_bert_projection
