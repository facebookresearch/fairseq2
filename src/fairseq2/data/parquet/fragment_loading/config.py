# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass
from typing import Optional

import pyarrow as pa
import pyarrow.compute as pc


class StringOnlyMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance._check_string_fields()
        return instance


@dataclass
class NamedColumns(metaclass=StringOnlyMeta):
    """
    Base class for defining a list of columns to load from a dataset.

    Example:
    ```
    @dataclass
    class AudioColumns(NamedColumns):
        audio: str = "audio_wav"
        sr: str = "sample_rate"
        label: Optional[str] = None
        extra_columns: List[str] = field(default_factory=lambda: ["quality", "artist"])
    ```

    Using this class as parameter will mean:
    - the columns ["audio_wav", "sample_rate", "quality", "artist"] will be loaded from the dataset
    - after loading the renaming ("audio_wav" -> "audio", "sample_rate" -> "sr") will be applied.
    - `extra_columns` will be loaded as is (no renaming)
    - note that `None` values will be ignored (e.g. "label" will not be loaded or renamed)

    """

    def _check_string_fields(self):
        for field_name, field_value in self.__dict__.items():
            if field_name in ["extra_columns"] and (
                not isinstance(field_value, list)
                or not all(isinstance(col, str) for col in field_value)
            ):
                raise TypeError(
                    f"Field '{field_name}' must be a list of strings, got {type(field_value).__name__} instead."
                )
            elif field_value is not None and not isinstance(field_value, str):
                raise TypeError(
                    f"Field '{field_name}' must be of type str, got {type(field_value).__name__} instead."
                )


@dataclass
class FragmentLoadingConfig:
    """
    This config describes the loading of fragments from the a parquet dataset.
    """

    columns: Optional[NamedColumns] = None
    """The list of columns to load.
    This should be used to indicate which columns to load and how to rename them.
    Renaming is useful to get uniform data schema when working from different datasets.

    Note that if `columns` is None, all columns will be loaded.
    """

    add_fragment_traces: bool = True
    """
    If ``True``, adds a column to the loaded fragment with the file path, row group number, and sample position index,
    so that any sample can be uniquely identified into the global dataset.
    This is useful for debugging or to keep track of the loaded fragments.
    """

    # basic filtering

    drop_null: bool = True
    """If ``True``, drops rows containing any null value."""

    min_batch_size: int = 1
    """Drops tables whose length `<=min_batch_size`. Applied after `drop_null` and `filters`."""

    filters: Optional[str] = None
    """
    Python string representing `pyarrow.dataset.Expression` that will be used to filter the loaded data in memory.
    To get real filter object, `eval(filters)` will be applied first.
    Note that `pa` and `pc` are available in the scope of `eval` call meaning `pyarrow` and `pyarrow.compute` respectively.

    The filters are applied before any column renaming or transormation.
    """

    # performance related params
    use_threads: bool = False
    """Whether pyarrow should use its internal threads to read the Parquet file.
    Since we rely on the external parallelism, this param is tuned off by
    default."""

    pyarrow_cpu_count: Optional[int] = 20

    nb_prefetch: float = 0.0
    """The number loaded fragments to prefetch."""

    num_parallel_fragments: float = 1
    """The number of fragments to load in parallel.
    Typical, memory vs speed tradeoff.
    """

    cache_dir: Optional[str] = None
    """
    Experimental feature! Use with caution !
    The directory to cache the loaded fragments.
    If provided, loaded pa.Table will be memory mapped into under random name into `cache_dir`.
    All references to pa.Table are released, corresponding files will be deleted.
    Allows to reduce the memory footprint with a small performance penalty.

    If None, the fragments will not be cached.
    """
