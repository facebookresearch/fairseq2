# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

__version__ = "0.4.0dev0"

import fairseq2n  # Report any fairseq2n initialization error eagerly.

# isort: split

import fairseq2.datasets
import fairseq2.models

# isort: split

from fairseq2.setup import setup_fairseq2 as setup_fairseq2

setup_extensions = setup_fairseq2  # compat
