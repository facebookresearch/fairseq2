# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional

import pyarrow as pa
import pytest

from fairseq2.data.parquet.configs import (
    DataLoadingConfig,
    EvaluationDataLoadingConfig,
    ParquetBasicDataloaderConfig,
    ParquetBatchFormat,
    ParquetDatasetConfig,
    ParquetDatasetLimitOptions,
    ValidationDataLoadingConfig,
)


def test_parquet_dataset_limit_options_default() -> None:
    """Check default initialization of ParquetDatasetLimitOptions."""
    limit_options = ParquetDatasetLimitOptions()
    assert limit_options.columns is None
    assert limit_options.fraction_of_files is None
    assert limit_options.nb_files is None
    assert limit_options.nb_fragments is None
    assert limit_options.nb_rows is None
    assert limit_options.limit_nb_tokens is None
    assert limit_options.token_columns is None


@pytest.skip("TODO: check if this is still correct")
def test_parquet_dataset_config_valid() -> None:
    """Check successful init when all required args are valid."""
    cfg = ParquetDatasetConfig(parquet_path="s3://bucket/dataset.parquet")
    assert cfg.parquet_path == "s3://bucket/dataset.parquet"
    assert cfg.split_to_row_groups is True  # default
    assert cfg.nb_parallel_fragments == None  # default


def test_parquet_dataset_config_raises_on_empty_path() -> None:
    """Ensure ValueError is raised when parquet_path is empty."""
    with pytest.raises(ValueError, match="requires non-empty path got"):
        ParquetDatasetConfig(parquet_path="")


def test_parquet_dataset_config_filters_conversion() -> None:
    """If filters is a list of old-style tuples, ensure __post_init__ converts them."""
    filters_list = [("col_name", "=", "foo")]
    cfg = ParquetDatasetConfig(parquet_path="file.parquet", filters=filters_list)
    # Verify it's converted to a pyarrow expression
    assert isinstance(cfg.filters, pa.dataset.Expression)


def test_data_loading_config_valid_batch_size() -> None:
    """Check DataLoadingConfig is valid when only batch_size is provided."""
    cfg = DataLoadingConfig(batch_size=32, max_tokens=None)
    assert cfg.batch_size == 32
    assert cfg.max_tokens is None


def test_data_loading_config_valid_max_tokens() -> None:
    """Check DataLoadingConfig is valid when only max_tokens is provided."""
    cfg = DataLoadingConfig(batch_size=None, max_tokens=1000, order_by_length=True)
    assert cfg.batch_size is None
    assert cfg.max_tokens == 1000


@pytest.mark.parametrize("batch_size,max_tokens", [(None, None), (32, 1000)])
def test_data_loading_config_invalid_batch_vs_tokens(batch_size, max_tokens) -> None:
    """
    Test that DataLoadingConfig fails when both or neither batch_size/max_tokens
    are specified.
    """
    with pytest.raises(
        ValueError, match="need to provide either `batch_size` either `max_tokens`"
    ):
        DataLoadingConfig(batch_size=batch_size, max_tokens=max_tokens)


def test_data_loading_config_requires_order_by_length_for_max_tokens() -> None:
    """Test that DataLoadingConfig raises if max_tokens is used but order_by_length is None."""
    with pytest.raises(ValueError, match="`order_by_length` should be given"):
        DataLoadingConfig(batch_size=None, max_tokens=100, order_by_length=None)


def test_data_loading_config_even_sharding_requires_in_memory() -> None:
    """Test that setting even_sharding=True without sharding_in_memory=True raises ValueError."""
    with pytest.raises(
        ValueError, match="`even_sharding` requires `sharding_in_memory=True`"
    ):
        DataLoadingConfig(
            batch_size=32, max_tokens=None, even_sharding=True, sharding_in_memory=False
        )


def test_validation_data_loading_config_defaults() -> None:
    """Test ValidationDataLoadingConfig default overrides."""
    val_cfg = ValidationDataLoadingConfig()
    assert val_cfg.multiple_dataset_chaining == "concat"
    assert val_cfg.nb_epochs == 1
    assert val_cfg.shuffle is False
    assert val_cfg.batch_size == 10
    assert val_cfg.max_tokens is None
    assert val_cfg.even_sharding is False
    assert val_cfg.sharding_in_memory is False


def test_evaluation_data_loading_config_defaults() -> None:
    """Test EvaluationDataLoadingConfig default overrides."""
    eval_cfg = EvaluationDataLoadingConfig()
    assert eval_cfg.multiple_dataset_chaining == "concat"
    assert eval_cfg.nb_epochs == 1
    assert eval_cfg.min_batch_size == 1
    assert eval_cfg.shuffle is False
    assert eval_cfg.batch_size == 10
    assert eval_cfg.even_sharding is False
    assert eval_cfg.sharding_in_memory is True
    assert eval_cfg.max_samples is None


def test_parquet_dataloader_config_default() -> None:
    """Test minimal usage of ParquetDataloaderConfig."""
    cfg = ParquetBasicDataloaderConfig(batch_size=16, max_tokens=None)
    assert cfg.output_format == ParquetBatchFormat.pyarrow
