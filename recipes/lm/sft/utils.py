from dataclasses import dataclass, field
from collections.abc import Set
from typing import Any, Final, final, TypeAlias


class DatasetLoadError(Exception):
    dataset_name: str

    def __init__(self, dataset_name: str, message: str) -> None:
        super().__init__(message)

        self.dataset_name = dataset_name


class UnknownSplitError(ValueError):
    dataset_name: str
    split: str
    available_splits: Set[str]

    def __init__(
        self, dataset_name: str, split: str, available_splits: Set[str]
    ) -> None:
        s = ", ".join(sorted(available_splits))

        super().__init__(
            f"'{split}' is not a known split of the '{dataset_name}' dataset. The following splits are available: {s}"
        )

        self.dataset_name = dataset_name
        self.split = split
        self.available_splits = available_splits


@dataclass
class StaticBatching:
    """Specifies batching where each batch has the same number of examples."""

    batch_size: int
    """The number of examples in each batch."""


@dataclass
class LengthBatching:
    """Specifies batching where each batch has a maximum number of elements."""

    max_num_elements: int
    """The maximum number of elements (e.g. tokens) in each batch."""


Batching: TypeAlias = StaticBatching | LengthBatching
