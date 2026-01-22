# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides functions for managing PyTorch data types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Set
from contextlib import contextmanager
from typing import Any, TypeAlias, final

import torch
from torch import get_default_dtype
from torch.overrides import TorchFunctionMode
from typing_extensions import override

from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.typing import ContextManager
from fairseq2.utils.threading import ThreadLocalStorage
from fairseq2.utils.warn import _warn_deprecated

DataType: TypeAlias = torch.dtype


def set_dtype(dtype: DataType) -> ContextManager[None]:
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

    return resolver.resolve(DataTypeContext).set_dtype(dtype)


def default_dtype(dtype: DataType) -> ContextManager[None]:
    _warn_deprecated(
        "`default_dtype()` is deprecated and will be removed in v0.14. Use `set_dtype()` instead."
    )

    return set_dtype(dtype)


def get_current_dtype() -> DataType:
    """
    Returns the current floating-point data type of the calling thread.

    .. warning::

        This function might impose a slight performance cost. Avoid calling it
        in hot code paths.
    """
    resolver = get_dependency_resolver()

    return resolver.resolve(DataTypeContext).get_current_dtype()


class DataTypeContext(ABC):
    """
    Provides methods to get and set the current floating-point data type of the
    calling thread.

    This interface can be used as an alternative to the corresponding standalone
    functions in object-oriented code.
    """

    @abstractmethod
    def get_current_dtype(self) -> DataType:
        """See :func:`get_current_dtype`."""

    @abstractmethod
    def set_dtype(self, dtype: DataType) -> ContextManager[None]:
        """See :func:`set_dtype`."""


@final
class _StandardDataTypeContext(DataTypeContext):
    def __init__(self, mode_stack: _DataTypeModeStack) -> None:
        self._mode_stack = mode_stack

    @override
    def get_current_dtype(self) -> DataType:
        mode = self._mode_stack.maybe_get_top_mode()
        if mode is not None:
            return mode.dtype

        return get_default_dtype()

    @override
    @contextmanager
    def set_dtype(self, dtype: DataType) -> Iterator[None]:
        mode = self._mode_stack.set_mode(dtype)

        try:
            with mode:
                yield
        finally:
            self._mode_stack.pop_mode()


@final
class _DataTypeModeStack:
    def __init__(self, constructors: Set[Any], tls: ThreadLocalStorage) -> None:
        self._constructors = constructors
        self._tls = tls

    def set_mode(self, dtype: DataType) -> _DataTypeMode:
        mode = _DataTypeMode(dtype, self._constructors)

        modes = self._get_dtype_mode_stack()

        modes.append(mode)

        if len(modes) > 1:
            modes[-2].enabled = False

        return mode

    def pop_mode(self) -> None:
        modes = self._get_dtype_mode_stack()

        if len(modes) > 1:
            modes[-2].enabled = True

        modes.pop()

    def maybe_get_top_mode(self) -> _DataTypeMode | None:
        modes = self._get_dtype_mode_stack()
        if modes:
            return modes[-1]

        return None

    def _get_dtype_mode_stack(self) -> list[_DataTypeMode]:
        return self._tls.get("dtype_mode_stack", list)


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
