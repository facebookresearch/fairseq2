# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides functions for managing PyTorch data types.
"""

from __future__ import annotations

from collections.abc import Iterator, Set
from contextlib import contextmanager
from typing import Any, ContextManager, TypeAlias, final

import torch
from torch import get_default_dtype
from torch.overrides import TorchFunctionMode

from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.utils.threading import ThreadLocalStorage
from fairseq2.utils.warn import _warn_deprecated

DataType: TypeAlias = torch.dtype


@contextmanager
def set_dtype(dtype: DataType) -> Iterator[None]:
    """
    Changes the floating-point data type of the calling thread to the specified
    type.

    This function acts as a context manager, ensuring that within its scope, any
    operation that constructs tensors uses the specified data type - unless an
    explicit ``dtype`` argument is provided.

    .. code:: python

        import torch

        from fairseq2.data_type import set_dtype

        with set_dtype(torch.bfloat16):
            t = torch.ones((4,4))

            assert t.dtype == torch.bfloat16

            with set_dtype(torch.float16):
                t = torch.ones((4, 4))

                assert t.dtype == torch.float16

        t = torch.ones((4, 4))

        assert t.dtype == torch.float32
    """
    resolver = get_dependency_resolver()

    with resolver.resolve(_DataTypeModeStack).push_mode(dtype) as mode:
        with mode:
            yield


def default_dtype(dtype: DataType) -> ContextManager[None]:
    _warn_deprecated(
        "`default_dtype()` is deprecated and will be removed in v0.14. Use `set_dtype()` instead."
    )

    return set_dtype(dtype)


def get_current_dtype() -> DataType:
    """Returns the current floating point data type of the calling thread."""
    resolver = get_dependency_resolver()

    mode = resolver.resolve(_DataTypeModeStack).maybe_get_top_mode()
    if mode is not None:
        return mode.dtype

    return get_default_dtype()


@final
class _DataTypeModeStack:
    def __init__(self, constructors: Set[Any], tls: ThreadLocalStorage) -> None:
        self._constructors = constructors
        self._tls = tls

    @contextmanager
    def push_mode(self, dtype: DataType) -> Iterator[_DataTypeMode]:
        mode = _DataTypeMode(dtype, self._constructors)

        modes = self._get_thread_dtype_modes()

        modes.append(mode)

        if len(modes) > 1:
            modes[-2].enabled = False

        try:
            yield mode
        finally:
            if len(modes) > 1:
                modes[-2].enabled = True

            modes.pop()

    def maybe_get_top_mode(self) -> _DataTypeMode | None:
        modes = self._get_thread_dtype_modes()
        if modes:
            return modes[-1]

        return None

    def _get_thread_dtype_modes(self) -> list[_DataTypeMode]:
        return self._tls.get("dtype_modes", list)


@final
class _DataTypeMode(TorchFunctionMode):
    def __init__(self, dtype: DataType, constructors: Set[Any]) -> None:
        self._constructors = constructors

        self.dtype = dtype
        self.enabled = True

    def __torch_function__(  # type: ignore[override]
        self, func: Any, types: Any, args: Any, kwargs: Any = None
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if self.enabled and func in self._constructors:
            dtype = kwargs.get("dtype", None)
            if dtype is None:
                kwargs["dtype"] = self.dtype

        return func(*args, **kwargs)


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
