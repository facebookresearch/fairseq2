# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.setup.assets import (
    register_package_metadata_provider as register_package_metadata_provider,
)
from fairseq2.setup.datasets import register_dataset as register_dataset
from fairseq2.setup.root import setup_fairseq2 as setup_fairseq2
from fairseq2.setup.root import setup_runtime_context as setup_runtime_context
from fairseq2.setup.text_tokenizers import (
    register_text_tokenizer as register_text_tokenizer,
)
