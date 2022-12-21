import os
import sys
from pathlib import Path
from typing import Any, List

import torch

import fairseq2.callbacks
import fairseq2.generate
from fairseq2.typing import Device


def inference(
    task: Any,
    beam_size: int,
    max_len: int,
    unk_penalty: float,
    batch_size: int,
    src_bos: str,
    tgt_bos: str,
    device: Device,
) -> None:
    tty = os.isatty(sys.stdin.fileno())
    if tty:
        batch_size = 1
    strategy = fairseq2.generate.BeamSearchStrategy(
        token_meta=task.tokenizer,
        beam_size=beam_size,
        max_len=max_len,
        unk_penalty=unk_penalty,
    )

    task.model.eval()

    def gen(batch: List[str]) -> List[str]:
        if not batch:
            return batch
        return strategy.generate_str(
            task.model,
            task.tokenizer,
            batch,
            src_bos=src_bos,
            tgt_bos=tgt_bos,
            device=device,
        )

    batch = []
    if tty:
        print("> ", end="", flush=True)
    for line in sys.stdin:
        batch.append(line.strip())
        if len(batch) < batch_size:
            continue
        for translation in gen(batch):
            print(translation)
        if tty:
            print("> ", end="", flush=True)
        batch.clear()

    for translation in gen(batch):
        print(translation)


def main(
    snapshot: Path,
    beam_size: int = 2,
    max_len: int = 128,
    unk_penalty: float = 1.0,
    batch_size: int = 16,
    src_bos: str = "",
    tgt_bos: str = "",
    device: Device = Device("cuda:0"),
) -> None:
    task = torch.hub.load(
        str(snapshot.parent), "model", snapshot.name, source="local", device=device
    )
    inference(
        task,
        beam_size=beam_size,
        max_len=max_len,
        unk_penalty=unk_penalty,
        batch_size=batch_size,
        src_bos=src_bos,
        tgt_bos=tgt_bos,
        device=device,
    )


if __name__ == "__main__":
    import func_argparse

    func_argparse.single_main(main)
