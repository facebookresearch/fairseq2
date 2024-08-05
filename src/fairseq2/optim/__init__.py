# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.optim.adamw import AdamW as AdamW
from fairseq2.optim.dynamic_loss_scaler import DynamicLossScaler as DynamicLossScaler
from fairseq2.optim.dynamic_loss_scaler import LossScaleResult as LossScaleResult
from fairseq2.optim.factory import AdamWConfig as AdamWConfig
from fairseq2.optim.factory import create_adamw_optimizer as create_adamw_optimizer
from fairseq2.optim.factory import create_optimizer as create_optimizer
from fairseq2.optim.factory import optimizer_factories as optimizer_factories
from fairseq2.optim.factory import optimizer_factory as optimizer_factory
from fairseq2.optim.optimizer import AbstractOptimizer as AbstractOptimizer
from fairseq2.optim.optimizer import ParameterCollection as ParameterCollection
