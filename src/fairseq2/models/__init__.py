# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.models.architecture_registry import (
    ModelArchitectureRegistry as ModelArchitectureRegistry,
)
from fairseq2.models.architecture_registry import (
    ModelConfigFactory as ModelConfigFactory,
)
from fairseq2.models.config_loader import ModelConfigLoader as ModelConfigLoader
from fairseq2.models.config_loader import (
    StandardModelConfigLoader as StandardModelConfigLoader,
)
from fairseq2.models.loader import CheckpointConverter as CheckpointConverter
from fairseq2.models.loader import DelegatingModelLoader as DelegatingModelLoader
from fairseq2.models.loader import ModelFactory as ModelFactory
from fairseq2.models.loader import ModelLoader as ModelLoader
from fairseq2.models.loader import StandardModelLoader as StandardModelLoader
from fairseq2.models.loader import load_model as load_model
from fairseq2.models.setup import setup_model as setup_model
