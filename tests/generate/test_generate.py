import torch

from fairseq2.generate.search import _get_last_time_axis, _stretch_to_beams
from tests.common import TestCase


class TestStretchToBeams(TestCase):
    def test_stretch_to_beams(self) -> None:
        t = torch.tensor(
            [
                [[3, 4], [5, 6]],
                [[13, 14], [15, 16]],
            ]
        )
        self.assertAllClose(
            _stretch_to_beams(t, 2),
            [
                [[3, 4], [5, 6]],
                [[3, 4], [5, 6]],
                [[13, 14], [15, 16]],
                [[13, 14], [15, 16]],
            ],
        )


class TestGetLastTimeAxis(TestCase):
    def test(self) -> None:
        self.assertAllClose(
            _get_last_time_axis(
                torch.tensor(
                    [
                        [[1, 2], [3, 4]],
                        [[5, 6], [7, 8]],
                    ]
                ),
                batch_first=True,
            ),
            [
                [3, 4],
                [7, 8],
            ],
        )

        self.assertAllClose(
            _get_last_time_axis(
                torch.tensor(
                    [
                        [[1, 2], [3, 4]],
                        [[5, 6], [7, 8]],
                    ]
                ),
                batch_first=False,
            ),
            [
                [5, 6],
                [7, 8],
            ],
        )
