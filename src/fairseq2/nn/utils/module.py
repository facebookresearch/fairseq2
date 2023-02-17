import threading
from contextlib import contextmanager
from typing import Iterator, Optional, cast

from fairseq2.typing import DataType, Device

_local = threading.local()

_local.device = None
_local.dtype = None


@contextmanager
def module_init_context(
    device: Optional[Device] = None, dtype: Optional[DataType] = None
) -> Iterator[None]:
    original_device, original_dtype = _local.device, _local.dtype

    _local.device, _local.dtype = device, dtype

    yield

    _local.device, _local.dtype = original_device, original_dtype


def device() -> Optional[Device]:
    return cast(Optional[Device], _local.device)


def dtype() -> Optional[DataType]:
    return cast(Optional[DataType], _local.dtype)
