import pickle
from pathlib import Path
from typing import Iterable, List

import torch

from tests.testlib import build_test_spm_tokenizer


def longs(x: List[Iterable[int]]) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.long)


def test_spm_decodes_special_token() -> None:
    spm = build_test_spm_tokenizer()
    ref_sample = spm.decode_batch(longs([[3, 4, 5]]))
    pad_sample = spm.decode_batch(longs([[3, 4, 5, spm.PAD, spm.PAD, spm.PAD]]))
    assert pad_sample == ref_sample

    eos_sample = spm.decode_batch(longs([[3, 4, 5, spm.EOS]]))
    assert eos_sample == ref_sample


def test_spm_can_pickle(tmp_path: Path) -> None:
    spm = build_test_spm_tokenizer()
    ref_sample = spm.decode_batch(longs([range(50)]))

    pkl_file = tmp_path / "spm.pkl"
    pkl_file.write_bytes(pickle.dumps(spm))
    new_spm = pickle.loads(pkl_file.read_bytes())

    new_sample = new_spm.decode_batch(longs([range(50)]))
    assert new_sample == ref_sample


def test_spm_decodes_in_batch() -> None:
    spm = build_test_spm_tokenizer()
    ref_sample = spm.decode_batch(longs([[3, 4, 5]]))

    batch_sample = spm.decode_batch(
        longs(
            [
                [3, 4, 5, spm.PAD, spm.PAD, spm.PAD],
                [3, 4, 5, spm.EOS, spm.PAD, spm.PAD],
                [3, 4, 5, spm.EOS, spm.EOS, spm.PAD],
            ]
        )
    )

    assert batch_sample == ref_sample * 3
