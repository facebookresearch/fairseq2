# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from fairseq2.models.opt.config import OPTConfig
from fairseq2.models.opt.factory import OPTFactory
from fairseq2.nn import BatchLayout


class TestOptFactory:
    def test_opt_factory(self) -> None:
        """Sanity check for factory + forward on a small model."""
        config = OPTConfig()  # by default opt-125m
        config.num_layers = 2
        config.vocab_size = 258
        config.model_dim = 24
        config.ffn_inner_dim = 48

        factory = OPTFactory(config)

        model = factory.create_model()

        model.eval()

        _ = model.forward(
            seqs=torch.randint(0, config.vocab_size, (2, 10)),
            seqs_layout=BatchLayout((2, 10), [10, 10]),
        )
