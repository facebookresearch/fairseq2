# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.opt.config import OPT_FAMILY as OPT_FAMILY
from fairseq2.models.opt.config import OPTConfig as OPTConfig
from fairseq2.models.opt.config import register_opt_configs as register_opt_configs
from fairseq2.models.opt.factory import OPTFactory as OPTFactory
from fairseq2.models.opt.factory import create_opt_model as create_opt_model
from fairseq2.models.opt.hub import get_opt_model_hub as get_opt_model_hub
from fairseq2.models.opt.interop import convert_opt_state_dict as convert_opt_state_dict
