from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VocabularyInfo:
    size: int
    """The size of the vocabulary."""

    unk_idx: Optional[int]
    """The index of the symbol that represents an unknown element."""

    bos_idx: Optional[int]
    """The index of the symbol that represents the beginning of a sequence."""

    eos_idx: Optional[int]
    """The index of the symbol that represents the end of a sequence."""

    pad_idx: Optional[int]
    """The index of the symbol that is used to pad a sequence."""
