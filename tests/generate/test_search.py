import torch

from fairseq2.generate import Tokenizer, search
from tests.common import TestCase
from tests.testlib import build_test_spm_tokenizer


class FakeTokenizer(Tokenizer):
    _non_symbol_vocab_size: int

    def __init__(
        self,
        *,
        non_symbol_vocab_size: int,
    ):
        self._non_symbol_vocab_size = non_symbol_vocab_size

    def vocab_size(self) -> int:
        return self._non_symbol_vocab_size + 4


class TestLogProb(TestCase):
    def test_log_prob(self) -> None:
        self.assertAllClose(
            search.log_prob(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, torch.nan],
                        [0.0, 0.0, 1.0, torch.inf],
                        [0.0, 0.0, 1.0, -torch.inf],
                    ]
                ),
                temperature=0.0,
                pad=0,
                bos=1,
            ),
            torch.tensor(
                [
                    [-torch.inf, -torch.inf, -1.743668, -0.743668],
                    [-torch.inf, -torch.inf, -torch.inf, -torch.inf],
                    [-torch.inf, -torch.inf, -torch.inf, -torch.inf],
                    [-torch.inf, -torch.inf, -0.551445, -torch.inf],
                ]
            ),
        )


class TestUnkPenalty(TestCase):
    def test(self) -> None:
        self.assertAllClose(
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


class TestPreventEos(TestCase):
    def test(self) -> None:
        self.assertAllClose(
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


class TestForceEos(TestCase):
    def test(self) -> None:
        self.assertAllClose(
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


class TestBeamSearch(TestCase):
    def test_prepare_state_noprefix(self) -> None:
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            max_len=100,
            beam_size=3,
        )

        src_tokens = torch.tensor(
            [
                [1, 2, 3, 4],
                [7, 8, 9, 10],
            ],
            dtype=torch.int64,
        )

        # min(100, 2 * 4 + 10) -> 18
        exp_max_len = 18

        # (bsz:2, beam_size:3, (exp_max_len:18 + 2))
        # (2, 3, 20)
        expected_tokens = torch.full((2, 3, 20), bs.tokenizer.PAD)
        expected_tokens[:, :, 0] = bs.tokenizer.BOS

        state = bs.prepare_state(src_tokens)

        self.assertEqual(state.step, 0)
        self.assertEqual(state.max_len, exp_max_len)

        self.assertAllClose(state.tokens, expected_tokens)

        self.assertAllClose(state.scores, torch.zeros((2, 3, 20)))

        self.assertAllClose(
            state.finished_mask,
            # (bsz, beam_size)
            torch.tensor([[False, False, False], [False, False, False]]),
        )

    def test_prepare_state_noprefix_maxlen(self) -> None:
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            max_len=10,
            beam_size=1,
        )

        src_tokens = torch.tensor(
            [
                [1, 2, 3, 4],
                [7, 8, 9, 10],
            ],
            dtype=torch.int64,
        )

        # min(10, 2 * 4 + 10) -> 10
        exp_max_len = 10

        # (bsz:2, beam_size:1, (exp_max_len:10 + 2))
        # (2, 12)
        expected_tokens = torch.full((2, 1, 12), bs.tokenizer.PAD)
        expected_tokens[:, :, 0] = bs.tokenizer.BOS

        state = bs.prepare_state(src_tokens)

        self.assertEqual(state.step, 0)
        self.assertEqual(state.max_len, exp_max_len)

        self.assertAllClose(state.tokens, expected_tokens)

    def test_prepare_state_prefix_single(self) -> None:
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            max_len=10,
            beam_size=2,
        )

        src_tokens = torch.tensor(
            [
                [1, 2, 3, 4],
                [7, 8, 9, 10],
            ],
            dtype=torch.int64,
        )

        prefix_tokens = torch.tensor(
            [99, 17],
            dtype=torch.int64,
        )

        # min(100, 2 * 4 + 10) -> 18
        exp_max_len = 10

        P = bs.tokenizer.PAD
        expected_tokens = torch.tensor(
            [
                [
                    [99, 17, P, P, P, P, P, P, P, P, P, P],
                    [99, 17, P, P, P, P, P, P, P, P, P, P],
                ],
                [
                    [99, 17, P, P, P, P, P, P, P, P, P, P],
                    [99, 17, P, P, P, P, P, P, P, P, P, P],
                ],
            ]
        )

        state = bs.prepare_state(
            src_tokens=src_tokens,
            prefix_tokens=prefix_tokens,
        )

        self.assertEqual(state.step, 1)
        self.assertEqual(state.max_len, exp_max_len)

        self.assertAllClose(state.tokens, expected_tokens)

        self.assertAllClose(state.scores, torch.zeros((2, 2, 10 + 2)))

        self.assertAllClose(
            state.finished_mask,
            # (bsz, beam_size)
            torch.tensor([[False, False], [False, False]]),
        )

    def test_prepare_state_prefix_batched(self) -> None:
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            max_len=10,
            beam_size=2,
        )

        src_tokens = torch.tensor(
            [
                [1, 2, 3, 4],
                [7, 8, 9, 10],
            ],
            dtype=torch.int64,
        )

        prefix_tokens = torch.tensor(
            [
                [99, 17],
                [88, 18],
            ],
            dtype=torch.int64,
        )

        # min(100, 2 * 4 + 10) -> 18
        exp_max_len = 10

        P = bs.tokenizer.PAD
        expected_tokens = torch.tensor(
            [
                [
                    [99, 17, P, P, P, P, P, P, P, P, P, P],
                    [99, 17, P, P, P, P, P, P, P, P, P, P],
                ],
                [
                    [88, 18, P, P, P, P, P, P, P, P, P, P],
                    [88, 18, P, P, P, P, P, P, P, P, P, P],
                ],
            ]
        )

        state = bs.prepare_state(
            src_tokens=src_tokens,
            prefix_tokens=prefix_tokens,
        )

        self.assertEqual(state.step, 1)
        self.assertEqual(state.max_len, exp_max_len)

        self.assertAllClose(state.tokens, expected_tokens)

        self.assertAllClose(state.scores, torch.zeros((2, 2, 10 + 2)))

        self.assertAllClose(
            state.finished_mask,
            # (bsz, beam_size)
            torch.tensor([[False, False], [False, False]]),
        )

    def test_step_done(self) -> None:
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            max_len=10,
            beam_size=1,
        )

        src_tokens = torch.tensor(
            [
                [1, 2, 3, 4],
                [7, 8, 9, 10],
            ],
            dtype=torch.int64,
        )

        state = bs.prepare_state(src_tokens)

        self.assertEqual(state.step, 0)
        self.assertEqual(state.batch_size, 2)
        self.assertEqual(state.beam_size, 1)

        dec_out = torch.rand((state.flat_size, bs.tokenizer.vocab_size()))

        state.done = True

        self.assertRaises(AssertionError, lambda: bs.step(dec_out, state))

    def test_step_bad_dec_shape(self) -> None:
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            max_len=10,
            beam_size=2,
        )

        src_tokens = torch.tensor(
            [
                [1, 2, 3, 4],
                [7, 8, 9, 10],
            ],
            dtype=torch.int64,
        )

        state = bs.prepare_state(src_tokens)

        self.assertEqual(state.step, 0)
        self.assertEqual(state.batch_size, 2)
        self.assertEqual(state.beam_size, 2)

        dec_out = torch.rand((state.flat_size * 2, bs.tokenizer.vocab_size() + 1))

        self.assertRaises(AssertionError, lambda: bs.step(dec_out, state))

    def test_step_one(self) -> None:
        tokenizer = FakeTokenizer(non_symbol_vocab_size=4)
        # we care about what these are, because we're going to hard-code some dec_out.
        self.assertEqual(tokenizer.UNK, 0)
        self.assertEqual(tokenizer.BOS, 1)
        self.assertEqual(tokenizer.EOS, 2)
        self.assertEqual(tokenizer.PAD, 3)

        beam_size = 2

        bs = search.BeamSearch(
            tokenizer=tokenizer,
            max_len=10,
            beam_size=beam_size,
        )

        batch_size = 2
        src_len = 4

        src_tokens = torch.zeros(
            size=(batch_size, src_len),
            dtype=torch.int64,
        )

        state = bs.prepare_state(src_tokens)

        self.assertEqual(state.step, 0)
        self.assertEqual(state.batch_size, 2)
        self.assertEqual(state.beam_size, 2)

        self.assertAllClose(
            state.tokens[:, :, 0],
            torch.tensor(
                [
                    [bs.tokenizer.BOS, bs.tokenizer.BOS],
                    [bs.tokenizer.BOS, bs.tokenizer.BOS],
                ]
            ),
        )
        self.assertAllClose(
            state.scores[:, :, 0],
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
        )

        dec_out_beam = torch.tensor(
            [
                # batch
                [
                    # beam
                    [
                        # [ UNK, BOS, EOS, PAD, ... ]
                        [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
                    ],
                ],
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                    ],
                ],
            ]
        )
        # (bsz * beam_size, vocab)
        dec_out = dec_out_beam.view(-1, bs.tokenizer.vocab_size())

        bs.step(dec_out, state)

        self.assertEqual(state.step, 1)

        dec_out_log_prob = bs.log_prob(
            dec_out,
            step=state.step,
            max_len=state.max_len,
        )

        dec_out_log_prob_beam = dec_out_log_prob.view(
            batch_size,
            beam_size,
            bs.tokenizer.vocab_size(),
        )

        self.assertAllClose(
            state.tokens[:, :, 1],
            torch.tensor(
                [
                    [5, 4],
                    [4, 6],
                ]
            ),
        )
        self.assertAllClose(
            state.scores[:, :, 1],
            torch.tensor(
                [
                    [
                        dec_out_log_prob_beam[0, 1, 5],
                        dec_out_log_prob_beam[0, 0, 4],
                    ],
                    [
                        dec_out_log_prob_beam[1, 1, 4],
                        dec_out_log_prob_beam[1, 0, 6],
                    ],
                ]
            ),
        )

    def test_step_continue(self) -> None:
        tokenizer = FakeTokenizer(non_symbol_vocab_size=4)
        # we care about what these are, because we're going to hard-code some dec_out.
        self.assertEqual(tokenizer.UNK, 0)
        self.assertEqual(tokenizer.BOS, 1)
        self.assertEqual(tokenizer.EOS, 2)
        self.assertEqual(tokenizer.PAD, 3)

        beam_size = 2

        bs = search.BeamSearch(
            tokenizer=tokenizer,
            max_len=10,
            beam_size=beam_size,
        )

        batch_size = 2
        src_len = 4

        src_tokens = torch.zeros(
            size=(batch_size, src_len),
            dtype=torch.int64,
        )

        state = bs.prepare_state(src_tokens)

        self.assertEqual(state.step, 0)
        self.assertEqual(state.batch_size, 2)
        self.assertEqual(state.beam_size, 2)

        self.assertAllClose(
            state.tokens[:, :, 0],
            torch.tensor(
                [
                    [bs.tokenizer.BOS, bs.tokenizer.BOS],
                    [bs.tokenizer.BOS, bs.tokenizer.BOS],
                ]
            ),
        )
        self.assertAllClose(
            state.scores[:, :, 0],
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
        )

        state.step = 1
        state.scores[:, :, 1] = torch.tensor(
            [
                [
                    [0.5, 0.0],
                    [0.0, 0.05],
                ]
            ]
        )
        state.tokens[:, :, 1] = torch.tensor(
            [
                [
                    # > 3
                    [4, 5],
                    [6, 7],
                ]
            ]
        )

        dec_out_beam = torch.tensor(
            [
                # batch
                [
                    # beam
                    # [ UNK, BOS, EOS, PAD, ... ]
                    [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0],
                    [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                ],
            ]
        )
        # (bsz * beam_size, vocab)
        dec_out = dec_out_beam.view(-1, bs.tokenizer.vocab_size())

        bs.step(dec_out, state)

        self.assertEqual(state.step, 2)

        dec_out_log_prob = bs.log_prob(
            dec_out,
            step=state.step,
            max_len=state.max_len,
        )

        dec_out_log_prob_beam = dec_out_log_prob.view(
            batch_size,
            beam_size,
            bs.tokenizer.vocab_size(),
        )

        self.assertAllClose(
            state.finished_mask,
            torch.tensor(
                [
                    [False, False],
                    [False, False],
                ]
            ),
        )

        # in selecting beams, we restructure history:
        self.assertAllClose(
            state.tokens[:, :, state.step - 1],
            torch.tensor(
                [
                    [5, 4],
                    [7, 6],
                ]
            ),
        )
        self.assertAllClose(
            state.tokens[:, :, state.step],
            torch.tensor(
                [
                    [7, 4],
                    [4, 7],
                ]
            ),
        )
        self.assertAllClose(
            state.scores[:, :, state.step],
            torch.tensor(
                [
                    [
                        state.scores[0, 0, state.step - 1]
                        + dec_out_log_prob_beam[0, 1, 7],
                        state.scores[0, 1, state.step - 1]
                        + dec_out_log_prob_beam[0, 0, 4],
                    ],
                    [
                        state.scores[1, 0, state.step - 1]
                        + dec_out_log_prob_beam[1, 1, 4],
                        state.scores[1, 1, state.step - 1]
                        + dec_out_log_prob_beam[1, 0, 7],
                    ],
                ]
            ),
        )

    def test_step_finished(self) -> None:
        tokenizer = FakeTokenizer(non_symbol_vocab_size=4)
        # we care about what these are, because we're going to hard-code some dec_out.
        self.assertEqual(tokenizer.UNK, 0)
        self.assertEqual(tokenizer.BOS, 1)
        self.assertEqual(tokenizer.EOS, 2)
        self.assertEqual(tokenizer.PAD, 3)

        beam_size = 2

        bs = search.BeamSearch(
            tokenizer=tokenizer,
            # force min_len == 0
            min_len=1,
            max_len=10,
            beam_size=beam_size,
        )

        batch_size = 2
        src_len = 4

        src_tokens = torch.zeros(
            size=(batch_size, src_len),
            dtype=torch.int64,
        )

        state = bs.prepare_state(src_tokens)

        self.assertEqual(state.step, 0)
        self.assertEqual(state.batch_size, 2)
        self.assertEqual(state.beam_size, 2)

        self.assertAllClose(
            state.tokens[:, :, 0],
            torch.tensor(
                [
                    [bs.tokenizer.BOS, bs.tokenizer.BOS],
                    [bs.tokenizer.BOS, bs.tokenizer.BOS],
                ]
            ),
        )
        self.assertAllClose(
            state.scores[:, :, 0],
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
        )

        dec_out_beam = torch.tensor(
            [
                # batch
                [
                    # beam
                    # [ UNK, BOS, EOS, PAD, ... ]
                    [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
                ],
                [
                    # force EOS here
                    [0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                ],
            ]
        )
        # (bsz * beam_size * search_breadth, vocab)
        dec_out = dec_out_beam.view(-1, bs.tokenizer.vocab_size())

        bs.step(dec_out, state)
        self.assertEqual(state.step, 1)

        dec_out_log_prob = bs.log_prob(
            dec_out,
            step=state.step,
            max_len=state.max_len,
        )
        dec_out_log_prob_beam = dec_out_log_prob.view(
            batch_size,
            beam_size,
            bs.tokenizer.vocab_size(),
        )

        self.assertAllClose(
            state.finished_mask,
            torch.tensor(
                [
                    [False, False],
                    [True, False],
                ]
            ),
        )

        self.assertAllClose(
            state.tokens[:, :, state.step],
            torch.tensor(
                [
                    [5, 4],
                    [bs.tokenizer.EOS, 4],
                ]
            ),
        )

        self.assertAllClose(
            state.scores[:, :, state.step],
            torch.tensor(
                [
                    [
                        dec_out_log_prob_beam[0, 1, 5],
                        dec_out_log_prob_beam[0, 0, 4],
                    ],
                    [
                        dec_out_log_prob_beam[1, 0, bs.tokenizer.EOS],
                        dec_out_log_prob_beam[1, 1, 4],
                    ],
                ]
            ),
        )

        dec_out_beam = torch.tensor(
            [
                # batch
                [
                    # beam
                    # [ UNK, BOS, EOS, PAD, ... ]
                    [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
                ],
                [
                    # should be masked by previous EOS
                    [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                ],
            ]
        )
        # (bsz * beam_size * search_breadth, vocab)
        dec_out = dec_out_beam.view(-1, bs.tokenizer.vocab_size())

        bs.step(dec_out, state)

        # finished (but still selected) beams have token PAD
        self.assertAllClose(
            state.tokens[:, :, state.step],
            torch.tensor(
                [
                    [4, 5],
                    [bs.tokenizer.PAD, 4],
                ]
            ),
        )

        # finished (but still selected) beams have score[step] == score[step-1]
        self.assertAllClose(
            state.scores[1, 0, state.step],
            state.scores[1, 0, state.step - 1],
        )

    def test_finalize_notop(self) -> None:
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            max_len=10,
            beam_size=3,
        )

        src_tokens = torch.tensor(
            [
                [1, 2, 3, 4],
                [7, 8, 9, 10],
            ],
            dtype=torch.int64,
        )

        # ((bsz:2 * beam_size:3), (exp_max_len:10 + 2))
        # (6, 12)

        state = bs.prepare_state(src_tokens)

        # beam_size = 1
        self.assertEqual(
            state.tokens.shape,
            torch.Size((2, 3, 12)),
        )
        state.tokens = torch.randint_like(
            state.tokens,
            low=0,
            high=1000,
        )
        state.scores = torch.rand_like(state.scores)

        sr = bs.finalize(state)
        self.assertAllClose(
            sr.tokens.view(2, 3, -1),
            # beam_size = 1
            state.tokens,
        )
        self.assertAllClose(
            sr.scores.view(2, 3, -1),
            # beam_size = 1
            state.scores,
        )

    def test_finalize_top(self) -> None:
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            max_len=10,
            beam_size=3,
        )

        src_tokens = torch.tensor(
            [
                [1, 2, 3, 4],
                [7, 8, 9, 10],
            ],
            dtype=torch.int64,
        )

        # ((bsz:2 * beam_size:3), (exp_max_len:10 + 2))
        # (6, 12)

        state = bs.prepare_state(src_tokens)
        state.step = 5

        # beam_size = 1
        self.assertEqual(
            state.tokens.shape,
            torch.Size((2, 3, 12)),
        )
        state.tokens = torch.randint_like(
            state.tokens,
            low=0,
            high=1000,
        )
        state.scores = torch.rand_like(state.scores)

        state.tokens[:, :, state.step + 1 :] = bs.tokenizer.PAD
        state.scores[:, :, state.step + 1 :] = -torch.inf

        # Force scores at step with a known sort order.
        # top-k [[1, 2], [1, 0]]
        state.scores[:, :, state.step] = torch.tensor(
            [
                [0.1, 0.9, 0.3],
                [0.4, 0.7, 0.2],
            ],
        )

        sr = bs.finalize(state, top=2)

        self.assertAllClose(
            sr.scores,
            torch.stack(
                [
                    torch.stack(
                        [
                            state.scores[0, 1, :],
                            state.scores[0, 2, :],
                        ]
                    ),
                    torch.stack(
                        [
                            state.scores[1, 1, :],
                            state.scores[1, 0, :],
                        ]
                    ),
                ]
            ),
        )
        self.assertAllClose(
            sr.tokens,
            torch.stack(
                [
                    torch.stack(
                        [
                            state.tokens[0, 1, :],
                            state.tokens[0, 2, :],
                        ]
                    ),
                    torch.stack(
                        [
                            state.tokens[1, 1, :],
                            state.tokens[1, 0, :],
                        ]
                    ),
                ]
            ),
        )

    def test_choose_beams(self) -> None:
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            min_len=10,
            max_len=20,
            beam_size=2,
        )

        # (bsz=2, input_beam_size=2, vocab_size=5)
        ps = torch.tensor(
            [
                [
                    [4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                ],
                [
                    [0.5, 4.5, 0.5, 1.5],
                    [0.5, 0.5, 0.5, 0.75],
                ],
            ]
        )

        sel = bs.choose_beams(2, ps)

        self.assertAllClose(
            sel.scores,
            torch.tensor(
                [
                    [4.0, 3.0],
                    [4.5, 1.5],
                ]
            ),
        )
        self.assertAllClose(
            sel.tokens,
            torch.tensor(
                [
                    [0, 2],
                    [1, 3],
                ]
            ),
        )
        self.assertAllClose(
            sel.beams,
            torch.tensor(
                [
                    [0, 1],
                    [0, 0],
                ]
            ),
        )

    def test_log_prob_below_min(self) -> None:
        max_len = 20
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            min_len=10,
            max_len=max_len,
        )

        # pull the token for 'A' (which won't be a control token).
        a_idx = bs.tokenizer.encode_batch(["A"])[0, 1].item()

        t = torch.rand((2, bs.tokenizer.vocab_size()))

        step = 1
        lprobs = bs.log_prob(t, step=step, max_len=max_len)

        raw = search.log_prob(t, temperature=0.1, pad=bs.tokenizer.PAD)

        self.assertEqual(
            lprobs[:, bs.tokenizer.UNK].tolist(),
            (raw[:, bs.tokenizer.UNK] - bs.unk_penalty).tolist(),
        )

        self.assertAllClose(
            lprobs[:, bs.tokenizer.PAD],
            torch.tensor([-torch.inf, -torch.inf]),
        )

        # Since we aren't forcing EOS, other tokens should not have -inf
        assert step < bs.max_len, (step, bs.max_len)
        self.assertNotIn(-torch.inf, lprobs[:, a_idx].tolist())

        # Since we've not yet reached min_len, EOS should have -inf.
        assert step < bs.min_len, (step, bs.min_len)
        self.assertAllClose(
            lprobs[:, bs.tokenizer.EOS],
            torch.tensor([-torch.inf, -torch.inf]),
        )

    def test_log_prob_running(self) -> None:
        max_len = 20
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            min_len=10,
            max_len=max_len,
        )

        # min_len < step < max_len
        step = 15

        # pull the token for 'A' (which won't be a control token).
        a_idx = bs.tokenizer.encode_batch(["A"])[0, 1].item()

        t = torch.rand((2, bs.tokenizer.vocab_size()))

        lprobs = bs.log_prob(t, step=step, max_len=max_len)

        raw = search.log_prob(t, temperature=0.1, pad=bs.tokenizer.PAD)

        self.assertEqual(
            lprobs[:, bs.tokenizer.UNK].tolist(),
            (raw[:, bs.tokenizer.UNK] - bs.unk_penalty).tolist(),
        )

        self.assertAllClose(
            lprobs[:, bs.tokenizer.PAD],
            torch.tensor([-torch.inf, -torch.inf]),
        )

        # Since we aren't forcing EOS, other tokens should not have -inf
        assert step < bs.max_len, (step, bs.max_len)
        self.assertNotIn(-torch.inf, lprobs[:, a_idx].tolist())

        # Since we aren't preventing EOS, EOS should not have -inf
        assert step > bs.min_len, (step, bs.min_len)
        self.assertNotIn(-torch.inf, lprobs[:, bs.tokenizer.EOS].tolist())

    def test_log_prob_above_max(self) -> None:
        max_len = 20
        bs = search.BeamSearch(
            tokenizer=build_test_spm_tokenizer(),
            min_len=10,
            max_len=max_len,
        )

        # pull the token for 'A' (which won't be a control token).
        a_idx = bs.tokenizer.encode_batch(["A"])[0, 1].item()

        # force max_len trigger.
        step = 20

        t = torch.rand((2, bs.tokenizer.vocab_size()))

        lprobs = bs.log_prob(t, step=20, max_len=max_len)

        self.assertAllClose(
            lprobs[:, bs.tokenizer.PAD],
            torch.tensor([-torch.inf, -torch.inf]),
        )

        # Since we are forcing EOS, other tokens should have -inf
        assert step >= bs.max_len, (step, bs.max_len)
        self.assertEqual(
            lprobs[:, a_idx].tolist(),
            [-torch.inf, -torch.inf],
        )
        # And EOS should not have -inf
        self.assertNotIn(-torch.inf, lprobs[:, bs.tokenizer.EOS].tolist())
