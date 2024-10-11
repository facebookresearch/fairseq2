# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from mypy_extensions import DefaultNamedArg
from torch.nn import Module

from fairseq2.factory_registry import ConfigBoundFactoryRegistry
from fairseq2.typing import CPU, DataClass, DataType, Device

model_factories = ConfigBoundFactoryRegistry[
    [DefaultNamedArg(Device, "device"), DefaultNamedArg(DataType, "dtype")], Module
]()


def create_model(
    family: str,
    arch: str | None = None,
    unstructured_config: object = None,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> tuple[Module, DataClass]:
    """Create a model of type registered with ``family``.

    :param family:
        The family of the model.
    :param arch:
        The architecture of the model.
    :param unstructured_config:
        The (partial) configuration of the model. Any ``EMPTY`` field will be
        filled with the corresponding value from the configuration of ``arch``.

    :returns:
        - The model.
        - The effective configuration of the model.
    """
    factory = model_factories.get(family, unstructured_config, arch, set_empty=True)

    model = factory(device=device or CPU, dtype=dtype or torch.float32)

    return model, factory.config
