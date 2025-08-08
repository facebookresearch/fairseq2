from fairseq2.datasets import LengthBatching, SequenceBatch, SyncMode
from fairseq2.recipes.config import (
    CommonSection,
    DatasetSection,
    EvaluatorSection,
    GangSection,
    ReferenceModelSection,
)


@dataclass(kw_only=True)
class Wav2Vec2EvalDatasetSection(DatasetSection):
    name: str | None = "librispeech_960h"

    family: str = WAV2VEC2_DATASET

    path: Path | None = None

    split: str = "valid"

    min_audio_len: int = 32_000
    """The minimum audio sequence length."""

    max_audio_len: int = 250_000
    """The maximum audio sequence length."""

    max_num_elements: int = 1_500_000
    """The maximum number of elements per batch."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""
