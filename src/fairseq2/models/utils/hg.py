# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import huggingface_hub
from torch import Tensor

try:
    import transformers  # type: ignore[import-not-found]
except ImportError:
    _has_transformers = False
else:
    _has_transformers = True


def save_hg_checkpoint(
    save_dir: Path,
    hg_checkpoint: dict[str, object],
    hg_config_class: str,
    hg_config: Mapping[str, object],
    hg_architecture: str | Sequence[str],
) -> None:
    if not _has_transformers:
        raise RuntimeError(
            "Hugging Face Transformers is not found in your Python environment. Use `pip install transformers`."
        )

    from transformers import PretrainedConfig  # type: ignore[attr-defined]

    try:
        hg_config_kls = getattr(transformers, hg_config_class)
    except AttributeError:
        raise TypeError(f"`transformers.{hg_config_class}` is not a type.") from None

    if not issubclass(hg_config_kls, PretrainedConfig):
        raise TypeError(
            f"`transformers.{hg_config_class}` is expected to be a subclass of `{PretrainedConfig}`."
        )

    try:
        hg_config_ = hg_config_kls()
    except TypeError as ex:
        raise ValueError(
            f"An instance of `transformers.{hg_config_class}` cannot be constructed. See the nested exception for details."
        ) from ex

    for key, value in hg_config.items():
        if not hasattr(hg_config_, key):
            raise ValueError(
                f"`transformers.{hg_config_class}` does not have an attributed named '{key}'."
            )

        setattr(hg_config_, key, value)

    if isinstance(hg_architecture, str):
        hg_architecture = [hg_architecture]

    setattr(hg_config_, "architectures", hg_architecture)

    hg_config_.save_pretrained(save_dir)

    tensors = {}

    for key, value in hg_checkpoint.items():
        if not isinstance(value, Tensor):
            raise TypeError(
                f"All values in `hg_checkpoint` must be of type `{Tensor}`, but the value of key '{key}' is of type `{type(value)}` instead."
            )

        tensors[key] = value

    huggingface_hub.save_torch_state_dict(tensors, save_dir)
