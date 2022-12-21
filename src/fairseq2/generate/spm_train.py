import dataclasses
import logging
from pathlib import Path

import sentencepiece

log = logging.getLogger(__name__)


@dataclasses.dataclass()
class TrainSpmConfig:
    vocab_size: int = 2**16 - 300  # 2 ** 16 - some room for special tokens
    training_lines: int = 5_000_000
    seed_sentencepiece_size: int = 5_000_000
    character_coverage: float = 0.999995
    model_type: str = "unigram"
    shuffle_input_sentence: bool = True
    num_threads: int = 4


def train(
    cfg: TrainSpmConfig,
    text_file: Path,
    output: Path,
) -> None:
    spm_output = Path(str(output) + ".model")
    assert not output.exists(), f"Sentencepiece training would override {output}"
    assert (
        not spm_output.exists()
    ), f"Sentencepiece training would override {spm_output}"
    log.info(f"Training sentencepiece model on: {text_file}, output model: {output}")
    sentencepiece.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=str(output),
        vocab_size=int(cfg.vocab_size),
        character_coverage=cfg.character_coverage,
        model_type=cfg.model_type,
        input_sentence_size=cfg.training_lines,
        seed_sentencepiece_size=min(cfg.training_lines, cfg.seed_sentencepiece_size),
        shuffle_input_sentence=cfg.shuffle_input_sentence,
        num_threads=cfg.num_threads,
    )

    spm_output.rename(output)
    log.info("sentencepiece training completed.")
