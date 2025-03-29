# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TableBucketingConfig:
    """
    This config describes the bucketing of the pa.Table elements (aka loaded fragments) into batches
    It consists of two parts:
    - first concat together several Table into a single big batch
    - then split the big batch into smaller batches with different strategies

    Currently, all `target_*` parameters are mutually exclusive.

    With all default values, no bucketing is applied and Table pipeline will remain the same.
    """

    # Table concat strategies
    min_fragment_number: int = 1
    """
    Load fragments at least `min_fragment_number` fragments which
    will be next concatenated together to form a single batch.
    """
    max_fragment_number: Optional[int] = None
    """
    Technical limit to avoid loading too many fragments to be concatenated together.
    """

    target_table_size: Optional[int] = None
    """
    Continue to load fragments until the target batch size is reached.
    Multiple loaded fragment tables will be concatenated together to form a single batch.
    """

    target_table_memory: Optional[int] = None
    """
    Target table memory expressed in `Mb`.
    Continue to load fragments until the target memory size is reached.
    Multiple loaded fragment tables will be concatenated together to form a single batch.
    """

    target_total_length: Optional[int] = None
    """
    Continue to load fragments until the target total length is reached.
    In that case, `length_columns` should be provided.

    For good padding and randomization, it is recommended to use target_total_length >> total_batch_length
    """

    combine_chunks: bool = True
    """
    If True, after several Tables are concatenated, the resulting table will be combined into a single chunk.
    This'll be typically faster for donwstream usage but will increase the memory footprint.
    """

    # batch splitting strategies

    batch_size: Optional[int] = None
    """
    Load fragment will be split into batches of max size = `batch_size` (keeping a potential smaller remainder)
    before being yielded.

    If `order_by_length` is False, then this operation does not present any performance or memory overhead,
    it generates a slice view on concatenated data (pa.Table).
    Otherwise, it will use `table.take(indexes)` which is more expensive.
    """

    order_by_length: bool = False
    """
    Whether to create the batches with homogeneous tokens length
    for more efficient padding.
    """

    length_reducer: str = "max"
    """
    The reducer to use to aggregate the length of multiple columns before bucketing.
    Currently, it can be "max" and "sum".
    Note that `max_tokens` reference the max length of the aggregated length.
    """

    total_batch_length: Optional[int] = None
    """Used with the ``order_by_length`` option to control the total length
     of sequences return in each batch (including padding).

    Row whose length is greater than `total_batch_length` will be dropped.
    """

    length_columns: Optional[List[str | None]] = None
    """
    List of columns to use for ordering by length.
    It should be provided if and only if `order_by_length` is True.
    None columns will be ignored.
    """
    drop_long_seq: bool = True
    """
    If True, drop rows whose length is greater than `total_batch_length`.
    """

    shuffle: bool = False

    seed: Optional[int] = 321

    # performance tuning
    cache: bool = False
    """
    Experimental feature! Use with caution !

    If `cache` is True, concatenated pa.Table will be memory mapped into `cache_dir` under a random name .
    After all references to pa.Table are released, the corresponding file will be deleted.
    Allows to reduce the memory footprint with a small performance penalty.
    This can a be a good tradeoff for large remote datasets.
    """

    cache_dir: Optional[str] = None
    """
    The directory to cache the loaded fragments.
    if None, a tmp dir will be created.
    """

    num_parallel_calls: int = 1
    nb_prefetch: int = 0
