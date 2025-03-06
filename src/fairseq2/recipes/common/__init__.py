# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipes.common._asset import (
    register_extra_asset_paths as register_extra_asset_paths,
)
from fairseq2.recipes.common._checkpoint import (
    create_checkpoint_manager as create_checkpoint_manager,
)
from fairseq2.recipes.common._dataset import load_dataset as load_dataset
from fairseq2.recipes.common._device import (
    create_device_stat_tracker as create_device_stat_tracker,
)
from fairseq2.recipes.common._distributed import broadcast_model as broadcast_model
from fairseq2.recipes.common._distributed import (
    setup_data_parallel_model as setup_data_parallel_model,
)
from fairseq2.recipes.common._error import (
    DatasetPathNotFoundError as DatasetPathNotFoundError,
)
from fairseq2.recipes.common._error import (
    HybridShardingNotSupportedError as HybridShardingNotSupportedError,
)
from fairseq2.recipes.common._error import (
    InvalidCheckpointPathError as InvalidCheckpointPathError,
)
from fairseq2.recipes.common._error import (
    ModelCompilationNotSupportedError as ModelCompilationNotSupportedError,
)
from fairseq2.recipes.common._error import (
    ModelParallelismNotSupportedError as ModelParallelismNotSupportedError,
)
from fairseq2.recipes.common._error import (
    ModelPathNotFoundError as ModelPathNotFoundError,
)
from fairseq2.recipes.common._evaluator import create_evaluator as create_evaluator
from fairseq2.recipes.common._gang import setup_gangs as setup_gangs
from fairseq2.recipes.common._generation import (
    create_seq2seq_generator as create_seq2seq_generator,
)
from fairseq2.recipes.common._generation import (
    create_seq_generator as create_seq_generator,
)
from fairseq2.recipes.common._generator import create_generator as create_generator
from fairseq2.recipes.common._metrics import (
    create_metric_recorder as create_metric_recorder,
)
from fairseq2.recipes.common._model import load_base_model as load_base_model
from fairseq2.recipes.common._model import prepare_model as prepare_model
from fairseq2.recipes.common._model import setup_model as setup_model
from fairseq2.recipes.common._optim import create_lr_scheduler as create_lr_scheduler
from fairseq2.recipes.common._optim import create_optimizer as create_optimizer
from fairseq2.recipes.common._profilers import create_profiler as create_profiler
from fairseq2.recipes.common._ref_model import (
    load_reference_model as load_reference_model,
)
from fairseq2.recipes.common._ref_model import (
    prepare_reference_model as prepare_reference_model,
)
from fairseq2.recipes.common._ref_model import (
    setup_reference_model as setup_reference_model,
)
from fairseq2.recipes.common._text_tokenizer import (
    load_text_tokenizer as load_text_tokenizer,
)
from fairseq2.recipes.common._trainer import create_trainer as create_trainer
