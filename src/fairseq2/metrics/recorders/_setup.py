# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.metrics.recorders._jsonl import register_jsonl_metric_recorder
from fairseq2.metrics.recorders._log import register_log_metric_recorder
from fairseq2.metrics.recorders._tensorboard import register_tensorboard_recorder
from fairseq2.metrics.recorders._wandb import register_wandb_recorder


def register_metric_recorders(context: RuntimeContext) -> None:
    register_log_metric_recorder(context)
    register_jsonl_metric_recorder(context)
    register_tensorboard_recorder(context)
    register_wandb_recorder(context)
