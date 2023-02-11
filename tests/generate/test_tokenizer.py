import functools
import pickle
from pathlib import Path
from typing import Iterable, List

import pytest
import torch

from fairseq2.generate import tokenizer
from tests.common import assert_equal

DATA = Path(__file__).parents[1] / "data"
SPM_PATH = DATA / "eng_Latn.1000.spm"


@functools.lru_cache()
def build_test_spm_tokenizer(pad_shift_hack: bool = False) -> tokenizer.SpmTokenizer:
    """Build a small testing SpmTokenizer from a local model.

    :return: an SpmTokenizer.
    """
    return tokenizer.SpmTokenizer.from_file(SPM_PATH, _pad_shift_hack=pad_shift_hack)


def longs(x: List[Iterable[int]]) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.long)


def test_spm_decodes_special_token() -> None:
    spm = build_test_spm_tokenizer()
    ref_sample = spm.decode_batch(longs([[3, 4, 5]]))
    pad_sample = spm.decode_batch(longs([[3, 4, 5, spm.PAD, spm.PAD, spm.PAD]]))
    assert pad_sample == ref_sample

    eos_sample = spm.decode_batch(longs([[3, 4, 5, spm.EOS]]))
    assert eos_sample == ref_sample


@pytest.mark.parametrize("pad_shift_hack", [False, True])
def test_spm_can_pickle(tmp_path: Path, pad_shift_hack: bool) -> None:
    spm = build_test_spm_tokenizer(pad_shift_hack=pad_shift_hack)
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


def test_spm_encode() -> None:
    spm = build_test_spm_tokenizer()
    sample = spm.encode_batch(["Hello world !"])
    assert_equal(sample, [[spm.BOS, 131, 29, 130, 113, 51, 417, 5, 67, spm.EOS]])
    assert spm.decode_batch(sample)[0] == "Hello world !"

    sample = spm.encode_batch(["Hello world !"], bos=42)
    assert_equal(sample, [[42, 131, 29, 130, 113, 51, 417, 5, 67, spm.EOS]])


def test_pad_shift_hack() -> None:
    base_spm = build_test_spm_tokenizer(pad_shift_hack=False)
    base_sample = base_spm.encode_batch(["Hello world !"])

    shift_spm = build_test_spm_tokenizer(pad_shift_hack=True)
    shift_sample = shift_spm.encode_batch(["Hello world !"])
    # There is no PAD, so you can simply shift to get the original tokens.
    assert_equal(base_sample, shift_sample - 1)
    assert shift_spm.decode_batch(shift_sample)[0] == "Hello world !"
