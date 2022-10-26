import unittest

import hamcrest
import torch

from fairseq2.generate import search
from tests import tensor_matchers as tm
from tests.testlib import build_test_spm_tokenizer


class TestLogProb(unittest.TestCase):
    def test_log_prob(self) -> None:
        tm.assert_tensor_equals(
            search.log_prob(
                torch.tensor(
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, 1.0, torch.nan],
                        [0.0, 1.0, torch.inf],
                        [0.0, 1.0, -torch.inf],
                    ]
                ),
                temperature=0.0,
                pad=0,
            ),
            torch.tensor(
                [
                    [-torch.inf, -1.5514448, -0.5514447],
                    [-torch.inf, -torch.inf, -torch.inf],
                    [-torch.inf, -torch.inf, -torch.inf],
                    [-torch.inf, -0.31326166, -torch.inf],
                ]
            ),
            close=True,
        )

    def test_log_pad(self) -> None:
        tm.assert_tensor_equals(
            search.log_prob(
                torch.tensor(
                    [
                        [0.0, 1.0, 1.0],
                    ]
                ),
                temperature=0.0,
                pad=2,
            ),
            torch.tensor(
                [
                    [-1.861995, -0.861995, -torch.inf],
                ]
            ),
            close=True,
        )


class TestUnkPenalty(unittest.TestCase):
    def test(self) -> None:
        tm.assert_tensor_equals(
            search.unk_penalty(
                torch.tensor(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                    ]
                ),
                penalty=0.5,
                unk=1,
            ),
            torch.tensor(
                [
                    [1.0, 1.5, 3.0],
                    [4.0, 4.5, 6.0],
                ]
            ),
        )


class TestPreventEos(unittest.TestCase):
    def test(self) -> None:
        tm.assert_tensor_equals(
            search.prevent_eos(
                torch.tensor(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                    ]
                ),
                eos=1,
            ),
            torch.tensor(
                [
                    [1.0, -torch.inf, 3.0],
                    [4.0, -torch.inf, 6.0],
                ]
            ),
        )


class TestForceEos(unittest.TestCase):
    def test(self) -> None:
        tm.assert_tensor_equals(
            search.force_eos(
                torch.tensor(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                    ]
                ),
                eos=1,
            ),
            torch.tensor(
                [
                    [-torch.inf, 2.0, -torch.inf],
                    [-torch.inf, 5.0, -torch.inf],
                ]
            ),
        )


class TestBeamSearch(unittest.TestCase):
    def test_choose_beams(self) -> None:
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            min_len=10,
            max_len=20,
        )

        # (bsz=2, beam_size=2, vocab_size=5)
        ps = torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.5, 4.5, 3.5, 2.5, 1.5],
                ],
                [
                    [11.0, 12.0, 13.0, 14.0, 17.0],
                    [14.5, 13.5, 12.5, 11.5, 9.0],
                ],
            ]
        )

        sel = bs.chose_beams(ps)

        tm.assert_tensor_equals(
            sel.values,
            [
                [5.5, 4.5],
                [17.0, 14.5],
            ],
        )
        tm.assert_tensor_equals(
            sel.indices,
            [
                [5, 6],
                [4, 5],
            ],
        )

    def test_log_prob_below_min(self) -> None:
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            min_len=10,
            max_len=20,
        )

        # pull the token for 'A' (which won't be a control token).
        a_idx = bs.tokenizer.encode_batch(["A"])[0, 1].item()

        t = torch.rand((2, bs.tokenizer.vocab_size()))

        lprobs = bs.log_prob(t)

        raw = search.log_prob(t, temperature=0.1, pad=bs.tokenizer.PAD)

        tm.assert_match(
            lprobs[:, bs.tokenizer.UNK].tolist(),
            (raw[:, bs.tokenizer.UNK] - bs.unk_penalty).tolist(),
        )

        tm.assert_tensor_equals(
            lprobs[:, bs.tokenizer.PAD],
            [-torch.inf, -torch.inf],
        )

        # Since we aren't forcing EOS, other tokens should not have -inf
        assert bs._step < bs.max_len, (bs._step, bs.max_len)
        tm.assert_match(
            lprobs[:, a_idx].tolist(),
            hamcrest.only_contains(
                hamcrest.not_(-torch.inf),
            ),
        )

        # Since we've not yet reached min_len, EOS should have -inf.
        assert bs._step < bs.min_len, (bs._step, bs.min_len)
        tm.assert_tensor_equals(
            lprobs[:, bs.tokenizer.EOS],
            [-torch.inf, -torch.inf],
        )

    def test_log_prob_running(self) -> None:
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            min_len=10,
            max_len=20,
        )

        # min_len < bs._step < max_len
        bs._step = 15

        # pull the token for 'A' (which won't be a control token).
        a_idx = bs.tokenizer.encode_batch(["A"])[0, 1].item()

        t = torch.rand((2, bs.tokenizer.vocab_size()))

        lprobs = bs.log_prob(t)

        raw = search.log_prob(t, temperature=0.1, pad=bs.tokenizer.PAD)

        tm.assert_match(
            lprobs[:, bs.tokenizer.UNK].tolist(),
            (raw[:, bs.tokenizer.UNK] - bs.unk_penalty).tolist(),
        )

        tm.assert_tensor_equals(
            lprobs[:, bs.tokenizer.PAD],
            [-torch.inf, -torch.inf],
        )

        # Since we aren't forcing EOS, other tokens should not have -inf
        assert bs._step < bs.max_len, (bs._step, bs.max_len)
        tm.assert_match(
            lprobs[:, a_idx].tolist(),
            hamcrest.only_contains(
                hamcrest.not_(-torch.inf),
            ),
        )

        # Since we aren't preventing EOS, EOS should not have -inf
        assert bs._step > bs.min_len, (bs._step, bs.min_len)
        tm.assert_match(
            lprobs[:, bs.tokenizer.EOS].tolist(),
            hamcrest.only_contains(
                hamcrest.not_(-torch.inf),
            ),
        )

    def test_log_prob_above_max(self) -> None:
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            min_len=10,
            max_len=20,
        )

        # pull the token for 'A' (which won't be a control token).
        a_idx = bs.tokenizer.encode_batch(["A"])[0, 1].item()

        # force max_len trigger.
        bs._step = 20

        t = torch.rand((2, bs.tokenizer.vocab_size()))

        lprobs = bs.log_prob(t)

        tm.assert_tensor_equals(
            lprobs[:, bs.tokenizer.PAD],
            [-torch.inf, -torch.inf],
        )

        # Since we are forcing EOS, other tokens should have -inf
        assert bs._step >= bs.max_len, (bs._step, bs.max_len)
        tm.assert_match(
            lprobs[:, a_idx].tolist(),
            [-torch.inf, -torch.inf],
        )
        # And EOS should not have -inf
        tm.assert_match(
            lprobs[:, bs.tokenizer.EOS].tolist(),
            hamcrest.only_contains(
                hamcrest.not_(-torch.inf),
            ),
        )
