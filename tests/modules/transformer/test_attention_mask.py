import torch

from fairseq2.nn.transformer import CausalAttentionMaskGenerator

from ...common import TestCase


class TestCausalAttentionMaskGenerator(TestCase):
    def test_call_generates_correct_mask(self) -> None:
        g = CausalAttentionMaskGenerator()

        mask = g(torch.ones((4, 6)))

        self.assertEqual(mask.shape, (4, 4))

        inf = float("-inf")

        expected_mask = torch.tensor(
            [
                [0.0, inf, inf, inf],
                [0.0, 0.0, inf, inf],
                [0.0, 0.0, 0.0, inf],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertAllClose(mask, expected_mask)

    def test_call_generates_correct_mask_if_batch_first(self) -> None:
        g = CausalAttentionMaskGenerator()

        mask = g(torch.ones((6, 4)), batch_first=True)

        self.assertEqual(mask.shape, (4, 4))

        inf = float("-inf")

        expected_mask = torch.tensor(
            [
                [0.0, inf, inf, inf],
                [0.0, 0.0, inf, inf],
                [0.0, 0.0, 0.0, inf],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertAllClose(mask, expected_mask)

    def test_call_generates_correct_mask_if_no_batch(self) -> None:
        g = CausalAttentionMaskGenerator()

        mask = g(torch.ones(4))

        self.assertEqual(mask.shape, (4, 4))

        inf = float("-inf")

        expected_mask = torch.tensor(
            [
                [0.0, inf, inf, inf],
                [0.0, 0.0, inf, inf],
                [0.0, 0.0, 0.0, inf],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertAllClose(mask, expected_mask)

    def test_call_returns_same_mask_if_seq_len_is_equal_or_less(self) -> None:
        g = CausalAttentionMaskGenerator()

        mask1 = g(torch.ones(4))
        mask2 = g(torch.ones(4))
        mask3 = g(torch.ones(3))

        self.assertEqual(mask1.data_ptr(), mask2.data_ptr())
        self.assertEqual(mask1.data_ptr(), mask3.data_ptr())

        self.assertEqual(mask1.shape, (4, 4))
        self.assertEqual(mask2.shape, (4, 4))
        self.assertEqual(mask3.shape, (3, 3))

    def test_call_returns_new_mask_if_seq_len_is_greater(self) -> None:
        g = CausalAttentionMaskGenerator()

        mask1 = g(torch.ones(4))
        mask2 = g(torch.ones(5))
        mask3 = g(torch.ones(8))

        self.assertNotEqual(mask1.data_ptr(), mask2.data_ptr())
        self.assertNotEqual(mask1.data_ptr(), mask3.data_ptr())

        self.assertEqual(mask1.shape, (4, 4))
        self.assertEqual(mask2.shape, (5, 5))
        self.assertEqual(mask3.shape, (8, 8))
