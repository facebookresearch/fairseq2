# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.setup._asset import (
    register_package_metadata_provider as register_package_metadata_provider,
)
from fairseq2.setup._datasets import DatasetRegistrar as DatasetRegistrar
from fairseq2.setup._error import SetupError as SetupError
from fairseq2.setup._models import ModelRegistrar as ModelRegistrar
from fairseq2.setup._root import setup_fairseq2 as setup_fairseq2
from fairseq2.setup._root import setup_library as setup_library
from fairseq2.setup._text_tokenizers import (
    TextTokenizerRegistrar as TextTokenizerRegistrar,
)
