# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.setup._assets import (
    register_package_metadata_provider as register_package_metadata_provider,
)
from fairseq2.setup._cli import setup_cli as setup_cli
from fairseq2.setup._error import SetupError as SetupError
from fairseq2.setup._lib import setup_fairseq2 as setup_fairseq2
from fairseq2.setup._lib import setup_library as setup_library
