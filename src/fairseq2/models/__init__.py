# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.handler import (
    ActivationCheckpointApplier as ActivationCheckpointApplier,
)
from fairseq2.models.handler import FSDPApplier as FSDPApplier
from fairseq2.models.handler import HuggingFaceSaver as HuggingFaceSaver
from fairseq2.models.handler import ModelFactory as ModelFactory
from fairseq2.models.handler import ModelFamilyHandler as ModelFamilyHandler
from fairseq2.models.handler import ShardSpecsProvider as ShardSpecsProvider
from fairseq2.models.handler import (
    StandardModelFamilyHandler as StandardModelFamilyHandler,
)
from fairseq2.models.handler import StateDictConverter as StateDictConverter
from fairseq2.models.handler import TorchCompiler as TorchCompiler
from fairseq2.models.handler import register_model_family as register_model_family
from fairseq2.models.hub import GlobalModelLoader as GlobalModelLoader
from fairseq2.models.hub import ModelFamilyNotKnownError as ModelFamilyNotKnownError
from fairseq2.models.hub import ModelHub as ModelHub
from fairseq2.models.hub import ModelHubAccessor as ModelHubAccessor
from fairseq2.models.hub import ModelNotKnownError as ModelNotKnownError
from fairseq2.models.hub import load_model as load_model
