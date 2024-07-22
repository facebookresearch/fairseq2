from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from fairseq2.typing import DataType


@dataclass
class PreferenceCriterionConfig:  # TODO: should this be abstract?
    """Holds the variant-configurations of a language model preference-finetuning task."""

    # Reference Model
    reference_model: Union[str, Path]
    """The name or path to the asset card of the reference model to use."""

    reference_dtype: DataType
    """The data type of the reference model."""

    reference_tensor_parallel_size: int
    """The size of tensor parallelism for the reference model."""
