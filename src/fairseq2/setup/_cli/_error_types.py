# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.chatbots import UnknownChatbotError
from fairseq2.cli import Cli
from fairseq2.data.text.tokenizers import (
    UnknownTextTokenizerError,
    UnknownTextTokenizerFamilyError,
)
from fairseq2.datasets import (
    InvalidDatasetTypeError,
    UnknownDatasetError,
    UnknownDatasetFamilyError,
    UnknownSplitError,
)
from fairseq2.generation import (
    UnknownBeamSearchAlgorithmError,
    UnknownSamplerError,
    UnknownSeq2SeqGeneratorError,
    UnknownSequenceGeneratorError,
)
from fairseq2.metrics import UnknownMetricDescriptorError
from fairseq2.metrics.recorders import UnknownMetricRecorderError
from fairseq2.metrics.text import UnknownBleuTokenizerError
from fairseq2.models import (
    InvalidModelTypeError,
    ModelCheckpointNotFoundError,
    ModelParallelismNotSupportedError,
    ShardedModelLoadError,
    UnknownModelArchitectureError,
    UnknownModelError,
    UnknownModelFamilyError,
)
from fairseq2.optim import UnknownOptimizerError
from fairseq2.optim.lr_scheduler import (
    UnknownLRSchedulerError,
    UnspecifiedNumberOfStepsError,
)
from fairseq2.profilers import UnknownProfilerError
from fairseq2.recipes.common import (
    InvalidCheckpointPathError,
    NotSupportedDistributedFeature,
)
from fairseq2.utils.validation import ValidationError


def _register_user_error_types(cli: Cli) -> None:
    cli.register_user_error_type(InvalidCheckpointPathError)
    cli.register_user_error_type(InvalidDatasetTypeError)
    cli.register_user_error_type(InvalidModelTypeError)
    cli.register_user_error_type(ModelCheckpointNotFoundError)
    cli.register_user_error_type(ModelParallelismNotSupportedError)
    cli.register_user_error_type(NotSupportedDistributedFeature)
    cli.register_user_error_type(ShardedModelLoadError)
    cli.register_user_error_type(UnknownBeamSearchAlgorithmError)
    cli.register_user_error_type(UnknownBleuTokenizerError)
    cli.register_user_error_type(UnknownChatbotError)
    cli.register_user_error_type(UnknownDatasetError)
    cli.register_user_error_type(UnknownDatasetFamilyError)
    cli.register_user_error_type(UnknownLRSchedulerError)
    cli.register_user_error_type(UnknownMetricDescriptorError)
    cli.register_user_error_type(UnknownMetricRecorderError)
    cli.register_user_error_type(UnknownModelArchitectureError)
    cli.register_user_error_type(UnknownModelError)
    cli.register_user_error_type(UnknownModelFamilyError)
    cli.register_user_error_type(UnknownOptimizerError)
    cli.register_user_error_type(UnknownProfilerError)
    cli.register_user_error_type(UnknownSamplerError)
    cli.register_user_error_type(UnknownSeq2SeqGeneratorError)
    cli.register_user_error_type(UnknownSequenceGeneratorError)
    cli.register_user_error_type(UnknownSplitError)
    cli.register_user_error_type(UnknownTextTokenizerError)
    cli.register_user_error_type(UnknownTextTokenizerFamilyError)
    cli.register_user_error_type(UnspecifiedNumberOfStepsError)
    cli.register_user_error_type(ValidationError)
