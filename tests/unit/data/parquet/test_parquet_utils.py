from fairseq2.data.parquet.utils import (
    get_parquet_dataset_metadata,
    get_row_group_level_metadata,
    pa,
    pq,
    pyarrow_table_to_torch_dict,
)


def test_get_row_group_level_metadata(multi_partition_file_dataset):
    dataset = pq.ParquetDataset(str(multi_partition_file_dataset))
    df = get_row_group_level_metadata(dataset)
    assert len(df) == 10
    assert sorted(df.columns.tolist()) == sorted(
        [
            "parquet_file_path",
            "row_group_id",
            "num_rows",
            "total_byte_size",
            "cat",
            "name",
            "score",
        ]
    )
    assert df["num_rows"].tolist() == [103, 103, 85, 110, 103, 95, 96, 117, 92, 96]

    expected_name0_stats = {
        "compression": "SNAPPY",
        "data_page_offset": 497,
        "dictionary_page_offset": 4,
        "encodings": ("PLAIN", "RLE", "RLE_DICTIONARY"),
        "file_offset": 0,
        "file_path": "",
        "has_dictionary_page": True,
        "is_stats_set": True,
        "num_values": 103,
        "path_in_schema": "name",
        "physical_type": "BYTE_ARRAY",
        "statistics": {
            "distinct_count": None,
            "has_min_max": True,
            "max": "name_99",
            "min": "name_106",
            "null_count": 0,
            "num_values": 103,
            "physical_type": "BYTE_ARRAY",
        },
        "total_compressed_size": 639,
        "total_uncompressed_size": 1385,
    }
    assert df["name"].iloc[0] == expected_name0_stats


def test_get_parquet_dataset_metadata(multi_partition_file_dataset):
    dataset = pq.ParquetDataset(str(multi_partition_file_dataset))
    df = get_parquet_dataset_metadata(dataset, full=True)
    assert len(df) == 10
    assert sorted(df.columns.tolist()) == sorted(
        [
            "parquet_file_path",
            "row_groups",
            "num_rows",
            "serialized_size",
            "num_row_groups",
            "num_columns",
            "created_by",
            "format_version",
            "cat",
        ]
    )

    df = get_parquet_dataset_metadata(dataset, full=False)
    assert len(df) == 10
    assert sorted(df.columns.tolist()) == sorted(
        [
            "parquet_file_path",
            "num_rows",
            "serialized_size",
            "num_row_groups",
            "num_columns",
            "cat",
        ]
    )


def test_nested_text_conversion():
    nested_input = pa.array([["abc", "efg"], ["xyz"]])
    tt = pa.Table.from_pydict({"nested_text": nested_input})
    converted = pyarrow_table_to_torch_dict(tt)
    # we want to keep this type unchanged
    assert isinstance(converted["nested_text"], pa.Array)
