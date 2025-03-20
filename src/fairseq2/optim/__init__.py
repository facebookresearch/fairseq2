# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.optim._adamw import ADAMW_OPTIMIZER as ADAMW_OPTIMIZER
from fairseq2.optim._adamw import AdamWConfig as AdamWConfig
from fairseq2.optim._adamw import AdamWHandler as AdamWHandler
from fairseq2.optim._dynamic_loss_scaler import DynamicLossScaler as DynamicLossScaler
from fairseq2.optim._dynamic_loss_scaler import LossScaleResult as LossScaleResult
from fairseq2.optim._error import UnknownOptimizerError as UnknownOptimizerError
from fairseq2.optim._handler import OptimizerHandler as OptimizerHandler
from fairseq2.optim._optimizer import OptimizerBase as OptimizerBase
from fairseq2.optim._optimizer import ParameterCollection as ParameterCollection
