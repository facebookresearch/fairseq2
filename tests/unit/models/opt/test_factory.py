# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from fairseq2.models.opt.config import OPTConfig
from fairseq2.models.opt.factory import create_opt_model
from fairseq2.nn import BatchLayout


class TestOptFactory:
    def test_opt_factory(self) -> None:
        """Sanity check for factory + forward on a small model."""
        config = OPTConfig(num_layers=2, vocab_size=258, model_dim=24, ffn_inner_dim=48)

        model = create_opt_model(config)

        model.eval()

        seqs = torch.randint(0, config.vocab_size, (2, 10))

        _ = model.forward(seqs=seqs, seqs_layout=BatchLayout.of(seqs))
