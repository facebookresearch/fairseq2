# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides helper functions to add support for new optimizers and
learning rate schedulers in recipes.
"""

from __future__ import annotations

from collections.abc import Sequence

from fairseq2.utils.validation import ValidationError


def maybe_raise_param_group_length_error(
    field: str, value: Sequence[object], num_param_groups: int
) -> None:
    """
    A helper function that raises :class:`~fairseq2.utils.validation.ValidationError`
    if the length of a learning rate scheduler configuration field (``len(value)``)
    does not match the number of optimizer parameter groups (``num_param_groups``).

    :param field: The name of the configuration field that holds ``value``.
    :param value: The value whose length to check.
    :param num_param_groups: The number of optimizer parameter groups.

    :raises ~fairseq2.utils.validation.ValidationError: if ``len(value)`` does
        not match ``num_param_groups``.

    .. code-block:: python
        :caption: A basic use of ``maybe_raise_param_group_length_error``

        from torch.optim import Optimizer

        from fairseq2.recipe.config import MyleLRConfig
        from fairseq2.recipe.optim import maybe_raise_param_group_length_error

        def get_start_lr(config: MyleLRConfig, optimizer: Optimizer) -> list[float]:
            num_param_groups = len(optimizer.param_groups)

            start_lr: float | list[float] = config.start_lr

            if isinstance(start_lr, float):
                return [start_lr] * num_param_groups

            maybe_raise_param_group_length_error("start_lr", start_lr, num_param_groups)

            return start_lr
    """
    if len(value) != num_param_groups:
        raise ValidationError(
            f"The length of `{field}` must match the number of optimizer parameter groups ({num_param_groups}), but is {len(value)} instead.", field="lr_scheduler.config"  # fmt: skip
        )
