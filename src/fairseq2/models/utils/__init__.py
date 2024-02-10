# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.data.text import BasicTextTokenizerLoader, GenericTextTokenizerLoader
from fairseq2.models.utils.arch_registry import (
    ArchitectureRegistry as ArchitectureRegistry,
)
from fairseq2.models.utils.generic_loaders import ConfigLoader as ConfigLoader
from fairseq2.models.utils.generic_loaders import ModelLoader as ModelLoader

# For backwards-compatibility with v0.2
TokenizerLoader = BasicTextTokenizerLoader
TokenizerLoaderBase = GenericTextTokenizerLoader
