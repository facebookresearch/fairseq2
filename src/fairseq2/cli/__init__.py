# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util

if (spec := importlib.util.find_spec("torchtnt")) is None:
    raise RuntimeError(
        "`torchtnt` cannot be found. Run `pip install 'fairseq2[cli]'` to install the required dependencies."
    )

from fairseq2.cli.distributed import DDP as DDP
from fairseq2.cli.module_loader import Xp as Xp
from fairseq2.cli.module_loader import XpScript as XpScript
from fairseq2.cli.module_loader import fairseq2_hub as fairseq2_hub
