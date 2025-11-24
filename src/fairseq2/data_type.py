# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides functions for managing PyTorch data types.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any, ContextManager, TypeAlias, cast, final

import torch
from torch import get_default_dtype
from torch.overrides import TorchFunctionMode

from fairseq2.utils.threading import _tls
from fairseq2.utils.warn import _warn_deprecated

DataType: TypeAlias = torch.dtype


# Holds the stack of current thread-local data types.
_tls.dtype_contexts = []


@contextmanager
def current_dtype(dtype: DataType) -> Iterator[None]:
    """
    Changes the floating-point data type of the calling thread to the specified
    type.

    This function acts as a context manager, ensuring that within its scope, any
    operation that constructs tensors uses the specified data type - unless an
    explicit ``dtype`` argument is provided.

    .. code:: python

        import torch

        from fairseq2.data_type import current_dtype

        with current_dtype(torch.bfloat16):
            t = torch.ones((4,4))

            assert t.dtype == torch.bfloat16

            with current_dtype(torch.float16):
                t = torch.ones((4, 4))

                assert t.dtype == torch.float16

        t = torch.ones((4, 4))

        assert t.dtype == torch.float32
    """
    contexts = _tls.dtype_contexts

    if contexts:
        contexts[-1].enabled = False

    try:
        constructors = _tensor_constructors()

        context = _DataTypeContext(dtype, constructors)

        contexts.append(context)

        try:
            with context:
                yield
        finally:
            contexts.pop()
    finally:
        if contexts:
            contexts[-1].enabled = True


def default_dtype(dtype: DataType) -> ContextManager[None]:
    _warn_deprecated(
        "`default_dtype()` is deprecated and will be removed in v0.14. Use `current_dtype()` instead."
    )

    return current_dtype(dtype)


def get_current_dtype() -> DataType:
    """Returns the current floating point data type of the calling thread."""
    contexts = _tls.dtype_contexts
    if contexts:
        return cast(DataType, contexts[-1].dtype)

    return get_default_dtype()


@final
class _DataTypeContext(TorchFunctionMode):
    def __init__(self, dtype: DataType, constructors: set[Any]) -> None:
        self.dtype = dtype
        self.constructors = constructors
        self.enabled = True

    def __torch_function__(  # type: ignore[override]
        self, func: Any, types: Any, args: Any, kwargs: Any = None
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if self.enabled and func in self.constructors:
            dtype = kwargs.get("dtype", None)
            if dtype is None:
                kwargs["dtype"] = self.dtype

        return func(*args, **kwargs)


@cache
def _tensor_constructors() -> set[Any]:
    # Taken from torch/utils/_device.py.
    return {
        torch.empty,
        torch.empty_permuted,
        torch.empty_strided,
        torch.empty_quantized,
        torch.ones,
        torch.arange,
        torch.bartlett_window,
        torch.blackman_window,
        torch.eye,
        torch.fft.fftfreq,
        torch.fft.rfftfreq,
        torch.full,
        torch.hamming_window,
        torch.hann_window,
        torch.kaiser_window,
        torch.linspace,
        torch.logspace,
        torch.nested.nested_tensor,
        torch.rand,
        torch.randn,
        torch.randint,
        torch.randperm,
        torch.range,
        torch.sparse_coo_tensor,
        torch.sparse_compressed_tensor,
        torch.sparse_csr_tensor,
        torch.sparse_csc_tensor,
        torch.sparse_bsr_tensor,
        torch.sparse_bsc_tensor,
        torch.tril_indices,
        torch.triu_indices,
        torch.zeros,
        torch.asarray,
        torch.tensor,
        torch.as_tensor,
        torch.scalar_tensor,
    }
