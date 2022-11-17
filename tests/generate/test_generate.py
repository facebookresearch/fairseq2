import torch

from fairseq2.generate.search import _get_last_time_axis, _stretch_to_beams
from tests.common import assert_close


def test_stretch_to_beams() -> None:
    t = torch.tensor(
        [
            [[3, 4], [5, 6]],
            [[13, 14], [15, 16]],
        ]
    )
    assert_close(
        _stretch_to_beams(t, 2),
        [
            [[3, 4], [5, 6]],
            [[3, 4], [5, 6]],
            [[13, 14], [15, 16]],
            [[13, 14], [15, 16]],
        ],
    )


def test_get_last_time_axis() -> None:
    assert_close(
        _get_last_time_axis(
            torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), batch_first=True
        ),
        [[3, 4], [7, 8]],
    )

    assert_close(
        _get_last_time_axis(
            torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), batch_first=False
        ),
        [[5, 6], [7, 8]],
    )
