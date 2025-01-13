# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.optim.adamw import ADAMW_OPTIMIZER as ADAMW_OPTIMIZER
from fairseq2.optim.adamw import AdamW as AdamW
from fairseq2.optim.adamw import AdamWConfig as AdamWConfig
from fairseq2.optim.adamw import AdamWHandler as AdamWHandler
from fairseq2.optim.dynamic_loss_scaler import DynamicLossScaler as DynamicLossScaler
from fairseq2.optim.dynamic_loss_scaler import LossScaleResult as LossScaleResult
from fairseq2.optim.handler import OptimizerHandler as OptimizerHandler
from fairseq2.optim.handler import OptimizerNotFoundError as OptimizerNotFoundError
from fairseq2.optim.optimizer import AbstractOptimizer as AbstractOptimizer
from fairseq2.optim.optimizer import ParameterCollection as ParameterCollection
from fairseq2.optim.static import create_optimizer as create_optimizer
