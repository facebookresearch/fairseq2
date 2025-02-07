# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.models import AbstractModelHandler
from fairseq2.models.jepa._config import JEPA_MODEL_FAMILY, JepaConfig
from fairseq2.models.jepa._factory import JepaFactory
from fairseq2.models.jepa._model import JepaModel
from fairseq2.models.utils.checkpoint import convert_model_state_dict


class JepaModelHandler(AbstractModelHandler):
    @property
    @override
    def family(self) -> str:
        return JEPA_MODEL_FAMILY

    @property
    @override
    def kls(self) -> type[Module]:
        return JepaModel

    @override
    def _create_model(self, config: object) -> Module:
        config = cast(JepaConfig, config)

        return JepaFactory(config).create_model()

    @override
    def _convert_checkpoint(
        self, checkpoint: dict[str, object], config: object
    ) -> dict[str, object]:
        return convert_jepa_checkpoint(checkpoint)


def convert_jepa_checkpoint(checkpoint: dict[str, object]) -> dict[str, object]:
    encoder_checkpoint = checkpoint.get("target_encoder")
    if encoder_checkpoint is None:
        encoder_checkpoint = checkpoint.get("encoder")
        if encoder_checkpoint is None:
            raise ValueError(
                "`checkpoint` does contain neither a 'target_encoder' nor an 'encoder' key."
            )

    if not isinstance(encoder_checkpoint, dict):
        raise TypeError(
            f"The encoder state in `checkpoint` must be of type `dict`, but is of type `{type(encoder_checkpoint)}` instead."
        )

    return _convert_jepa_encoder_checkpoint(encoder_checkpoint)


def _convert_jepa_encoder_checkpoint(
    checkpoint: dict[str, object],
) -> dict[str, object]:
    try:
        del checkpoint["module.backbone.pos_embed"]
    except KeyError:
        pass

    new_checkpoint: dict[str, object] = {}

    for name, param in checkpoint.items():
        if not isinstance(param, Tensor):
            raise TypeError(
                f"`checkpoint['encoder'][{name}]` must be of type `{Tensor}`, but is of type `{type(param)}` instead."
            )

        if name.endswith("qkv.weight"):
            q_proj, k_proj, v_proj = torch.chunk(param, 3, dim=0)

            new_checkpoint[name[:-10] + "q_proj.weight"] = q_proj
            new_checkpoint[name[:-10] + "k_proj.weight"] = k_proj
            new_checkpoint[name[:-10] + "v_proj.weight"] = v_proj

            continue

        if name.endswith("qkv.bias"):
            q_bias, k_bias, v_bias = torch.chunk(param, 3, dim=0)

            new_checkpoint[name[:-8] + "q_proj.bias"] = q_bias
            new_checkpoint[name[:-8] + "k_proj.bias"] = k_bias
            new_checkpoint[name[:-8] + "v_proj.bias"] = v_bias

            continue

        new_checkpoint[name] = param

    key_map = {
        # fmt: off
        r"^module\.backbone\.blocks\.([0-9]+)\.attn\.q_proj\.":   r"encoder.layers.\1.self_attn.q_proj.",
        r"^module\.backbone\.blocks\.([0-9]+)\.attn\.k_proj\.":   r"encoder.layers.\1.self_attn.k_proj.",
        r"^module\.backbone\.blocks\.([0-9]+)\.attn\.v_proj\.":   r"encoder.layers.\1.self_attn.v_proj.",
        r"^module\.backbone\.blocks\.([0-9]+)\.attn\.proj\.":     r"encoder.layers.\1.self_attn.output_proj.",
        r"^module\.backbone\.blocks\.([0-9]+)\.norm1\.":          r"encoder.layers.\1.self_attn_layer_norm.",
        r"^module\.backbone\.blocks\.([0-9]+)\.mlp\.fc1\.":       r"encoder.layers.\1.ffn.inner_proj.",
        r"^module\.backbone\.blocks\.([0-9]+)\.mlp\.fc2\.":       r"encoder.layers.\1.ffn.output_proj.",
        r"^module\.backbone\.blocks\.([0-9]+)\.norm2\.":          r"encoder.layers.\1.ffn_layer_norm.",
        r"^module\.backbone\.norm\.":                             r"encoder.layer_norm.",
        r"^module\.backbone\.patch_embed\.proj\.":                r"encoder_frontend.feature_extractor.conv.",
        # fmt: on
    }

    checkpoint = convert_model_state_dict(new_checkpoint, key_map)

    return {"model": checkpoint}
