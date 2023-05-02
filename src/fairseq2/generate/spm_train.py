import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import sentencepiece

log = logging.getLogger(__name__)


@dataclass
class TrainSpmConfig:
    vocab_size: int = 2**16
    training_lines: int = 5_000_000
    seed_sentencepiece_size: int = 5_000_000
    character_coverage: float = 0.999995
    model_type: str = "unigram"
    shuffle_input_sentence: bool = True
    num_threads: int = 4
    pad_idx: int = 0
    unk_idx: int = 1
    bos_idx: int = 2
    eos_idx: int = 3
    control_tokens: Optional[List[str]] = None


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

    if cfg.control_tokens:
        user_defined_symbols = ",".join(cfg.control_tokens)
    else:
        user_defined_symbols = None

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
        pad_id=cfg.pad_idx,
        unk_id=cfg.unk_idx,
        bos_id=cfg.bos_idx,
        eos_id=cfg.eos_idx,
        user_defined_symbols=user_defined_symbols,
    )

    spm_output.rename(output)
    log.info("sentencepiece training completed.")


def train_from_stream(cfg: TrainSpmConfig, stream: Iterable[str], output: Path) -> None:
    try:
        tmp_path = Path(tempfile.mkstemp(suffix=".txt")[1])
        training_lines = cfg.training_lines
        with tmp_path.open("wt", encoding="utf-8") as tmp:
            for i, line in enumerate(stream):
                print(line, file=tmp)
                if i >= training_lines:
                    break
        train(cfg, tmp_path, output)
    finally:
        tmp_path.unlink()
