import torch

from fairseq2.generate import search
from fairseq2.generate.tokenizer import TokenMeta
from tests.common import TestCase


class TestLogProb(TestCase):
    def test_log_prob(self) -> None:
        self.assertAllClose(
            search.dec_out_to_log_prob(
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


class TestForceToken(TestCase):
    def test(self) -> None:
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        search.force_token_(t, token=1)

        self.assertAllClose(
            t,
            torch.tensor(
                [
                    [-torch.inf, 2.0, -torch.inf],
                    [-torch.inf, 5.0, -torch.inf],
                ]
            ),
        )


class TestBeamSearchStrategy(TestCase):
    def test_prepare_state_noprefix(self) -> None:
        token_meta = TokenMeta(vocab_size=8, BOS=0, EOS=1, UNK=2, PAD=3)

        bs = search.BeamSearchStrategy(token_meta=token_meta, max_len=100, beam_size=3)

        src_tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], dtype=torch.int64)

        # min(100, 2 * 4 + 10) -> 18
        exp_max_len = 18

        # (bsz:2, beam_size:3, (exp_max_len:18 + 2))
        # (2, 3, 20)
        expected_tokens = torch.full((2, 3, 20), token_meta.PAD)
        expected_tokens[:, :, 0] = token_meta.BOS

        s = bs.new_search_job(src_tokens)

        self.assertEqual(s.step, 0)
        self.assertEqual(s.max_len, exp_max_len)

        self.assertAllClose(s.tokens, expected_tokens)

        self.assertAllClose(s.scores, torch.zeros((2, 3, 20)))

        self.assertAllClose(
            s.finished_mask,
            # (bsz, beam_size)
            torch.tensor([[False, False, False], [False, False, False]]),
        )

    def test_prepare_state_noprefix_maxlen(self) -> None:
        token_meta = TokenMeta(vocab_size=8, BOS=0, EOS=1, UNK=2, PAD=3)

        bs = search.BeamSearchStrategy(token_meta=token_meta, max_len=10, beam_size=1)

        src_tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], dtype=torch.int64)

        # min(10, 2 * 4 + 10) -> 10
        exp_max_len = 10

        # (bsz:2, beam_size:1, (exp_max_len:10 + 2))
        # (2, 12)
        expected_tokens = torch.full((2, 1, 12), token_meta.PAD)
        expected_tokens[:, :, 0] = token_meta.BOS

        s = bs.new_search_job(src_tokens)

        self.assertEqual(s.step, 0)
        self.assertEqual(s.max_len, exp_max_len)

        self.assertAllClose(s.tokens, expected_tokens)

    def test_prepare_state_prefix_single(self) -> None:
        token_meta = TokenMeta(vocab_size=8, BOS=0, EOS=1, UNK=2, PAD=3)

        bs = search.BeamSearchStrategy(token_meta=token_meta, max_len=10, beam_size=2)

        src_tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], dtype=torch.int64)

        prefix_tokens = torch.tensor([99, 17], dtype=torch.int64)

        # min(100, 2 * 4 + 10) -> 18
        exp_max_len = 10

        P = token_meta.PAD
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

        state = bs.new_search_job(src_tokens=src_tokens, prefix_tokens=prefix_tokens)

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
        token_meta = TokenMeta(vocab_size=8, BOS=0, EOS=1, UNK=2, PAD=3)

        bs = search.BeamSearchStrategy(token_meta=token_meta, max_len=10, beam_size=2)

        src_tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], dtype=torch.int64)

        prefix_tokens = torch.tensor([[99, 17], [88, 18]], dtype=torch.int64)

        # min(100, 2 * 4 + 10) -> 18
        exp_max_len = 10

        P = token_meta.PAD
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

        s = bs.new_search_job(src_tokens=src_tokens, prefix_tokens=prefix_tokens)

        self.assertEqual(s.step, 1)
        self.assertEqual(s.max_len, exp_max_len)

        self.assertAllClose(s.tokens, expected_tokens)

        self.assertAllClose(s.scores, torch.zeros((2, 2, 10 + 2)))

        self.assertAllClose(
            s.finished_mask,
            # (bsz, beam_size)
            torch.tensor([[False, False], [False, False]]),
        )

    def test_step_done(self) -> None:
        token_meta = TokenMeta(vocab_size=8, BOS=0, EOS=1, UNK=2, PAD=3)

        bs = search.BeamSearchStrategy(token_meta=token_meta, max_len=10, beam_size=1)

        src_tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], dtype=torch.int64)

        s = bs.new_search_job(src_tokens)
        self.assertEqual(s.step, 0)
        self.assertEqual(s.batch_size, 2)
        self.assertEqual(s.beam_size, 1)

        dec_out = torch.rand((s.flat_size, token_meta.vocab_size))

        s.done = True

        self.assertRaisesRegex(
            AssertionError, "done == True", lambda: s.update(dec_out)
        )

    def test_step_bad_dec_shape(self) -> None:
        token_meta = TokenMeta(vocab_size=8, UNK=0, BOS=1, EOS=2, PAD=3)
        bs = search.BeamSearchStrategy(token_meta=token_meta, max_len=10, beam_size=2)

        src_tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], dtype=torch.int64)

        s = bs.new_search_job(src_tokens)
        self.assertEqual(s.step, 0)
        self.assertEqual(s.batch_size, 2)
        self.assertEqual(s.beam_size, 2)

        dec_out = torch.rand((s.flat_size * 2, token_meta.vocab_size + 1))

        self.assertRaisesRegex(
            AssertionError,
            "input_beam_size .* must == .* beam_size",
            lambda: s.update(dec_out),
        )

    def test_step_one(self) -> None:
        token_meta = TokenMeta(vocab_size=8, UNK=0, BOS=1, EOS=2, PAD=3)
        beam_size = 2

        bs = search.BeamSearchStrategy(
            token_meta=token_meta, max_len=10, beam_size=beam_size
        )

        batch_size = 2
        src_len = 4

        src_tokens = torch.zeros(size=(batch_size, src_len), dtype=torch.int64)

        s = bs.new_search_job(src_tokens)
        self.assertEqual(s.step, 0)
        self.assertEqual(s.batch_size, 2)
        self.assertEqual(s.beam_size, 2)

        self.assertAllClose(
            s.tokens[:, :, 0],
            [[token_meta.BOS, token_meta.BOS], [token_meta.BOS, token_meta.BOS]],
        )
        self.assertAllClose(s.scores[:, :, 0], [[0.0, 0.0], [0.0, 0.0]])

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
        dec_out = dec_out_beam.view(-1, token_meta.vocab_size)

        s.update(dec_out)
        self.assertEqual(s.step, 1)

        dec_out_log_prob = s._log_prob(dec_out, step=s.step, max_len=s.max_len)

        dec_out_log_prob_beam = dec_out_log_prob.view(
            batch_size, beam_size, token_meta.vocab_size
        )

        self.assertAllClose(s.tokens[:, :, 1], [[5, 4], [4, 6]])
        self.assertAllClose(
            s.scores[:, :, 1],
            [
                [
                    dec_out_log_prob_beam[0, 1, 5],
                    dec_out_log_prob_beam[0, 0, 4],
                ],
                [
                    dec_out_log_prob_beam[1, 1, 4],
                    dec_out_log_prob_beam[1, 0, 6],
                ],
            ],
        )

    def test_step_continue(self) -> None:
        token_meta = TokenMeta(vocab_size=8, UNK=0, BOS=1, EOS=2, PAD=3)

        beam_size = 2

        bs = search.BeamSearchStrategy(
            token_meta=token_meta, max_len=10, beam_size=beam_size
        )

        batch_size = 2
        src_len = 4

        src_tokens = torch.zeros(size=(batch_size, src_len), dtype=torch.int64)

        s = bs.new_search_job(src_tokens)
        self.assertEqual(s.step, 0)
        self.assertEqual(s.batch_size, 2)
        self.assertEqual(s.beam_size, 2)

        self.assertAllClose(
            s.tokens[:, :, 0],
            [[token_meta.BOS, token_meta.BOS], [token_meta.BOS, token_meta.BOS]],
        )
        self.assertAllClose(s.scores[:, :, 0], [[0.0, 0.0], [0.0, 0.0]])

        s.step = 1
        s.scores[:, :, 1] = torch.tensor(
            [
                [
                    [0.5, 0.0],
                    [0.0, 0.05],
                ]
            ]
        )
        s.tokens[:, :, 1] = torch.tensor(
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
        dec_out = dec_out_beam.view(-1, token_meta.vocab_size)

        s.update(dec_out)
        self.assertEqual(s.step, 2)

        dec_out_log_prob = s._log_prob(dec_out, step=s.step, max_len=s.max_len)

        dec_out_log_prob_beam = dec_out_log_prob.view(
            batch_size, beam_size, token_meta.vocab_size
        )

        self.assertAllClose(s.finished_mask, [[False, False], [False, False]])

        # in selecting beams, we restructure history:
        self.assertAllClose(s.tokens[:, :, s.step - 1], [[5, 4], [7, 6]])
        self.assertAllClose(s.tokens[:, :, s.step], [[7, 4], [4, 7]])
        self.assertAllClose(
            s.scores[:, :, s.step],
            [
                [
                    s.scores[0, 0, s.step - 1] + dec_out_log_prob_beam[0, 1, 7],
                    s.scores[0, 1, s.step - 1] + dec_out_log_prob_beam[0, 0, 4],
                ],
                [
                    s.scores[1, 0, s.step - 1] + dec_out_log_prob_beam[1, 1, 4],
                    s.scores[1, 1, s.step - 1] + dec_out_log_prob_beam[1, 0, 7],
                ],
            ],
        )

    def test_step_finished(self) -> None:
        token_meta = TokenMeta(vocab_size=8, UNK=0, BOS=1, EOS=2, PAD=3)

        beam_size = 2

        bs = search.BeamSearchStrategy(
            token_meta=token_meta,
            # force min_len == 0
            min_len=1,
            max_len=10,
            beam_size=beam_size,
        )

        batch_size = 2
        src_len = 4

        src_tokens = torch.zeros(size=(batch_size, src_len), dtype=torch.int64)

        s = bs.new_search_job(src_tokens)
        self.assertEqual(s.step, 0)
        self.assertEqual(s.batch_size, 2)
        self.assertEqual(s.beam_size, 2)

        self.assertAllClose(
            s.tokens[:, :, 0],
            [[token_meta.BOS, token_meta.BOS], [token_meta.BOS, token_meta.BOS]],
        )
        self.assertAllClose(s.scores[:, :, 0], [[0.0, 0.0], [0.0, 0.0]])

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
        dec_out = dec_out_beam.view(-1, token_meta.vocab_size)

        s.update(dec_out)
        self.assertEqual(s.step, 1)

        dec_out_log_prob = s._log_prob(dec_out, step=s.step, max_len=s.max_len)
        dec_out_log_prob_beam = dec_out_log_prob.view(
            batch_size, beam_size, token_meta.vocab_size
        )

        self.assertAllClose(s.finished_mask, [[False, False], [True, False]])

        self.assertAllClose(s.tokens[:, :, s.step], [[5, 4], [token_meta.EOS, 4]])

        self.assertAllClose(
            s.scores[:, :, s.step],
            [
                [
                    dec_out_log_prob_beam[0, 1, 5],
                    dec_out_log_prob_beam[0, 0, 4],
                ],
                [
                    dec_out_log_prob_beam[1, 0, token_meta.EOS],
                    dec_out_log_prob_beam[1, 1, 4],
                ],
            ],
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
        dec_out = dec_out_beam.view(-1, token_meta.vocab_size)

        s.update(dec_out)

        # finished (but still selected) beams have token PAD
        self.assertAllClose(s.tokens[:, :, s.step], [[4, 5], [token_meta.PAD, 4]])

        # finished (but still selected) beams have score[step] == score[step-1]
        self.assertEqual(s.scores[1, 0, s.step], s.scores[1, 0, s.step - 1])

    def test_finalize_notop(self) -> None:
        token_meta = TokenMeta(vocab_size=8, UNK=0, BOS=1, EOS=2, PAD=3)
        bs = search.BeamSearchStrategy(token_meta=token_meta, max_len=10, beam_size=3)

        src_tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], dtype=torch.int64)

        # ((bsz:2 * beam_size:3), (exp_max_len:10 + 2))
        # (6, 12)

        s = bs.new_search_job(src_tokens)

        # beam_size = 1
        self.assertEqual(s.tokens.shape, torch.Size((2, 3, 12)))
        s.tokens = torch.randint_like(s.tokens, low=0, high=1000)
        s.scores = torch.rand_like(s.scores)

        sr = s.finalize()
        self.assertAllClose(
            sr.tokens.view(2, 3, -1),
            # beam_size = 1
            s.tokens,
        )
        self.assertAllClose(
            sr.scores.view(2, 3, -1),
            # beam_size = 1
            s.scores,
        )

    def test_finalize_top(self) -> None:
        token_meta = TokenMeta(vocab_size=8, UNK=0, BOS=1, EOS=2, PAD=3)
        bs = search.BeamSearchStrategy(token_meta=token_meta, max_len=10, beam_size=3)

        src_tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], dtype=torch.int64)
        # ((bsz:2 * beam_size:3), (exp_max_len:10 + 2))
        # (6, 12)

        s = bs.new_search_job(src_tokens)
        s.step = 5

        # beam_size = 1
        self.assertEqual(s.tokens.shape, torch.Size((2, 3, 12)))
        s.tokens = torch.randint_like(s.tokens, low=0, high=1000)
        s.scores = torch.rand_like(s.scores)

        s.tokens[:, :, s.step + 1 :] = token_meta.PAD
        s.scores[:, :, s.step + 1 :] = -torch.inf

        # Force scores at step with a known sort order.
        # top-k [[1, 2], [1, 0]]
        s.scores[:, :, s.step] = torch.tensor([[0.1, 0.9, 0.3], [0.4, 0.7, 0.2]])

        sr = s.finalize(top=2)

        self.assertAllClose(
            sr.scores,
            torch.stack(
                [
                    torch.stack(
                        [
                            s.scores[0, 1, :],
                            s.scores[0, 2, :],
                        ]
                    ),
                    torch.stack(
                        [
                            s.scores[1, 1, :],
                            s.scores[1, 0, :],
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
                            s.tokens[0, 1, :],
                            s.tokens[0, 2, :],
                        ]
                    ),
                    torch.stack(
                        [
                            s.tokens[1, 1, :],
                            s.tokens[1, 0, :],
                        ]
                    ),
                ]
            ),
        )

    def test_choose_beams(self) -> None:
        token_meta = TokenMeta(vocab_size=8, UNK=0, BOS=1, EOS=2, PAD=3)
        bs = search.BeamSearchStrategy(
            token_meta=token_meta, min_len=10, max_len=20, beam_size=2
        )

        src_tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], dtype=torch.int64)

        s = bs.new_search_job(src_tokens)

        # (bsz=2, input_beam_size=2, vocab_size=4)
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

        sel = s._choose_beams(ps)

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
        token_meta = TokenMeta(vocab_size=8, UNK=0, BOS=1, EOS=2, PAD=3)

        max_len = 20
        bs = search.BeamSearchStrategy(
            token_meta=token_meta, min_len=10, max_len=max_len
        )

        src_tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], dtype=torch.int64)

        s = bs.new_search_job(src_tokens)

        # anything < vocab_size, not in the tokens.
        a_idx = 5

        t = torch.rand((2, token_meta.vocab_size))

        step = 1
        lprobs = s._log_prob(t, step=step, max_len=max_len)

        raw = search.dec_out_to_log_prob(
            t, temperature=0.1, pad=token_meta.PAD, bos=token_meta.BOS
        )
        self.assertEqual(
            lprobs[:, token_meta.UNK].tolist(),
            (raw[:, token_meta.UNK] - bs.unk_penalty).tolist(),
        )

        self.assertAllClose(
            lprobs[:, token_meta.PAD], torch.tensor([-torch.inf, -torch.inf])
        )

        # Since we aren't forcing EOS, other tokens should not have -inf
        assert step < bs.max_len, (step, bs.max_len)
        self.assertNotIn(-torch.inf, lprobs[:, a_idx].tolist())

        # Since we've not yet reached min_len, EOS should have -inf.
        assert step < bs.min_len, (step, bs.min_len)
        self.assertAllClose(
            lprobs[:, token_meta.EOS], torch.tensor([-torch.inf, -torch.inf])
        )

    def test_log_prob_running(self) -> None:
        token_meta = TokenMeta(vocab_size=8, UNK=0, BOS=1, EOS=2, PAD=3)
        max_len = 20
        bs = search.BeamSearchStrategy(
            token_meta=token_meta, min_len=10, max_len=max_len
        )

        src_tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], dtype=torch.int64)

        s = bs.new_search_job(src_tokens)

        # min_len < step < max_len
        step = 15

        # anything < vocab_size, not in the tokens.
        a_idx = 5

        t = torch.rand((2, token_meta.vocab_size))

        lprobs = s._log_prob(t, step=step, max_len=max_len)

        raw = search.dec_out_to_log_prob(
            t, temperature=0.1, pad=token_meta.PAD, bos=token_meta.BOS
        )

        self.assertEqual(
            lprobs[:, token_meta.UNK].tolist(),
            (raw[:, token_meta.UNK] - bs.unk_penalty).tolist(),
        )

        self.assertAllClose(
            lprobs[:, token_meta.PAD], torch.tensor([-torch.inf, -torch.inf])
        )

        # Since we aren't forcing EOS, other tokens should not have -inf
        assert step < bs.max_len, (step, bs.max_len)
        self.assertNotIn(-torch.inf, lprobs[:, a_idx].tolist())

        # Since we aren't preventing EOS, EOS should not have -inf
        assert step > bs.min_len, (step, bs.min_len)
        self.assertNotIn(-torch.inf, lprobs[:, token_meta.EOS].tolist())

    def test_log_prob_above_max(self) -> None:
        token_meta = TokenMeta(vocab_size=8, UNK=0, BOS=1, EOS=2, PAD=3)
        max_len = 20
        bs = search.BeamSearchStrategy(
            token_meta=token_meta, min_len=10, max_len=max_len
        )

        src_tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], dtype=torch.int64)
        s = bs.new_search_job(src_tokens)

        # anything < vocab_size, not in the tokens.
        a_idx = 5

        # force max_len trigger.
        step = 20
        t = torch.rand((2, token_meta.vocab_size))
        lprobs = s._log_prob(t, step=20, max_len=max_len)

        self.assertAllClose(
            lprobs[:, token_meta.PAD], torch.tensor([-torch.inf, -torch.inf])
        )

        # Since we are forcing EOS, other tokens should have -inf
        assert step >= bs.max_len, (step, bs.max_len)
        self.assertEqual(lprobs[:, a_idx].tolist(), [-torch.inf, -torch.inf])
        # And EOS should not have -inf
        self.assertNotIn(-torch.inf, lprobs[:, token_meta.EOS].tolist())
