import torch
from torch import nn


class RnntBeamSearchModule(nn.Module):
    """
    A fully-GPU implementation of beam search for RNN-T models.
    Performs batched decoding without moving tensors to CPU, generally fast enough
    for decoding large validation sets during training, or other use cases that
    require decoding during training (MWER training, etc.).


    Notes about the algorithm:
    - The algorithm generally follows the same logic as other implementations of
    the transducer beam search. We process the audio time step by time step,
    always keeping track of the nbest hypos so far.
    - A hypo state is comprised of: predictor out/state, tokens, n_emitted, score,
    active (True/False). An hypo is becomes inactive when it already emitted blank
    for the current time step, or if step_max_symbols has reached, or if it was
    merged into another hypo with the same non-blank tokens.
    - An example state is: hypos, encoder_t. The example can move to the next t
    when all its hypos are inactive. Different examples can be in differnt encoder_t.
    This allows us to have less iterations overall and always have a large effective
    batch size for GPU calls.

    A high level algorithm:
    - At every iteration, all active hypos are fed to the joiner, to get the distribution
    over next tokens.
    - From the possible nbest * n_sps ways to extend our nbest hypos, we choose the
    merge_beam with highest scores.
    - From the chosen merge_beam, we merge hypos with the same non-blank tokens.
    Scores of those similar hypos are added together.
    - Out of the hypos left after score merging, we choose the nbest.
    - For hypos that end with non-blank symbol, we run the predictor to update
    the hypo's predictor_out/predictor_state.
    - Before moving to the next iteration, we update hypos and examples state.

    Implementation notes:
    - For convinience of implementation, most tensors are of shape [B * n_hypos, ...],
    where B is the batch size.
    - pytorch matrices must be of square shapes. In order to run joiner / predictor
    just on active hypos, we often index all tensor by some active_mask, and scatter
    the results to the right indexes.
    """

    def __init__(
        self,
        text_frontend,
        predictor,
        joiner,
        blank,
        step_max_symbols,
        vocab_size,
        length_norm=False,
        merge_beam=0,
        always_merge_blank=False,
    ):
        super().__init__()

        self.text_frontend = text_frontend
        self.predictor = predictor
        self.joiner = joiner
        self.original_blank = blank
        self.blank = vocab_size
        self.step_max_symbols = step_max_symbols
        self.length_norm = length_norm
        self.merge_beam = merge_beam
        self.always_merge_blank = always_merge_blank

    @staticmethod
    def idx_1d_to_2d(idx, dim2):
        return idx // dim2, idx % dim2

    @staticmethod
    def idx_2d_to_1d(idx1, idx2, dim2):
        return dim2 * idx1 + idx2

    @staticmethod
    def repeat_like(t, like):
        t = t.reshape([-1] + [1] * (len(like.shape) - 1))
        t = t.repeat([1] + list(like.shape)[1:])
        return t

    def apply_length_norm(self, scores, tokens, low_score):
        # Compute non-blank count
        non_blank_count = (tokens != self.blank).sum(axis=1) + 1
        # Normalize
        normalized_scores = scores / non_blank_count
        normalized_scores[scores == low_score] = low_score  # Don't normalize low score
        return normalized_scores

    @torch.no_grad()
    def forward(
        self,
        encoder_out,
        encoder_lengths,
        nbest,
    ):
        return self.run_beam_search(encoder_out, encoder_lengths, nbest)

    @torch.no_grad()
    def run_beam_search(self, encoder_out, encoder_lengths, nbest):
        # Create the initial values
        device = encoder_out.device
        float_type = encoder_out.dtype
        bsz = encoder_out.shape[0]

        # The user specifes how many non-blank tokens to use for hypo expansions.
        # Blank tokens will always be used to, so this will be the actual tensor
        # size
        if self.always_merge_blank:
            nmerge = self.merge_beam + nbest if self.merge_beam > 0 else 2 * nbest
        else:
            nmerge = self.merge_beam if self.merge_beam > 0 else nbest

        # minimum score for a hypo in the log domain
        low_score = -10000
        low_score_t = torch.tensor(low_score, dtype=float_type, device=device)

        # Those are used to index into various tensors
        # [0, ... 0 (nbest times), 1, ... 1 (nbest times), ...]
        ex_ind_nbest = torch.tensor(
            [[i] * nbest for i in range(bsz)], dtype=torch.int64, device=device
        ).reshape(-1)
        # [0, ... 0 (nmerge times), 1, ... 1 (nmerge times), ...]
        ex_ind_nmerge = torch.tensor(
            [[i] * nmerge for i in range(bsz)], dtype=torch.int64, device=device
        ).reshape(-1)
        # [[blank, ..., (nbest - 1) * blank], ... bsz times]
        blank_multiplied = torch.full(
            [bsz, nbest], fill_value=self.blank + 1, dtype=torch.int64, device=device
        )
        multipier = (
            torch.arange(nbest, dtype=torch.int64, device=device).reshape(1, -1) + 1
        )
        blank_multiplied *= multipier
        blank_multiplied -= 1

        # The following are used for hashing and score merging of hypos
        # [0, 1, ..., bsz -1]
        ex_separator = torch.arange(bsz, dtype=torch.int64, device=device)
        max_hash = 10000
        # [[0], [max_hash], ..., [(bsz-1) * max_hash]]
        ex_separator1 = ex_separator.reshape(-1, 1) * max_hash
        # [[0], [nmerge], ..., [(bsz-1) * nmerge]]
        ex_separator2 = ex_separator.reshape(-1, 1) * nmerge
        # [[0], [nbest], ..., [(bsz-1) * nbest]]
        ex_separator3 = ex_separator.reshape(-1, 1) * nbest
        # [0, 1, ..., bsz * nmerge - 1]
        hypo_idx = torch.arange(0, bsz * nmerge, dtype=torch.int64, device=device)
        hypo_idx = hypo_idx.reshape(1, -1)
        # [0.5, 0.5, ... 0.5] (bsz * nmerge times)
        half = torch.full(
            [1, bsz * nmerge], fill_value=0.5, dtype=float_type, device=device
        )

        # Run predictor to initialize hypos
        predictor_out = self.text_frontend(
            torch.full(
                [bsz * nbest, 1],
                fill_value=self.original_blank,
                dtype=torch.int64,
                device=device,
            )
        )
        predictor_out, (predictor_state_h, predictor_state_c) = self.predictor(
            predictor_out
        )
        predictor_state_h = predictor_state_h.transpose(0, 1)  # [B, n_layers, D]
        predictor_state_c = predictor_state_c.transpose(0, 1)  # [B, n_layers, D]

        # Initialize the rest of hypo states needed for beam search
        active = torch.ones([bsz * nbest], dtype=torch.bool, device=device)
        encoder_t = torch.zeros([bsz * nbest], dtype=torch.int64, device=device)
        n_emitted_symbols = torch.zeros([bsz * nbest], dtype=torch.int64, device=device)
        scores = torch.zeros([bsz * nbest], dtype=torch.float32, device=device)
        tokens = (
            torch.zeros([bsz * nbest, 1], dtype=torch.int64, device=device) + self.blank
        )
        alignment = torch.zeros([bsz * nbest, 1], dtype=torch.int64, device=device)

        # Will hold scores of candidate hypos
        candidate_scores = torch.zeros(
            [bsz * nbest, self.blank + 1], dtype=torch.float32, device=device
        )
        nmerge_scores_temp = torch.zeros(
            bsz * nmerge, dtype=torch.float32, device=device
        )

        done = False
        itr = 0
        while not done:
            # Part A: Run joiner

            # Choose active hypos
            n_active = active.sum()
            masked_predictor_out = predictor_out[active]  # [B*H - c, 1, dim]
            masked_encoder_t = encoder_t[active]
            masked_ex_ind = ex_ind_nbest[active]
            encoder_frames = encoder_out[masked_ex_ind, masked_encoder_t]

            # Feed active hypos to the joiner
            joined = self.joiner(
                torch.unsqueeze(encoder_frames, 1),  # [B*H - c, 1, dim]
                masked_predictor_out,  # [B*H, 1, dim]
            )  # [B*H, 1, 1, n_sps]
            joined = joined.squeeze(1).squeeze(1)  # [B*H - c, n_sps]
            joined = torch.nn.functional.log_softmax(joined, dim=1)

            # The algorithm assumes blank is the last token
            joined = torch.cat(
                [joined, joined[:, self.original_blank : self.original_blank + 1]],
                dim=1,
            )
            joined[:, self.original_blank] = low_score

            # Part B: Choose nmerge best hypos

            # In order to choose nmerge candidates, we need to choose the
            # top nmerge from the score matrix of shape [bsz, nbest, nsps].
            # Problem is, not all hypos are active, so the newly calculated
            # scores may be of shape [bsz, nbest - c, nsps].
            # To overcome this, for inactive hypos, we set
            # new_scores[b, h, blank] = scores[b, h] and
            # new_scores[b, h, c] = -inf for other sps c.
            # We then scatter the new scores to the right places.
            masked_scores = scores[active]
            masked_candidate_scores = masked_scores.unsqueeze(1) + joined
            candidate_scores[:] = low_score  # [B*H, nsps]
            candidate_scores[:, self.blank] = scores
            candidate_scores.masked_scatter_(
                active.unsqueeze(1), masked_candidate_scores
            )

            # If we emitted the allowed number of symbols, force emitting blank
            force_blank_mask = n_emitted_symbols == self.step_max_symbols
            candidate_scores[force_blank_mask, :-1] = low_score

            # At the first iteration we have only 1 hypo, mask the duplicates
            if itr == 0:
                candidate_scores[[i for i in range(bsz * nbest) if i % nbest != 0]] = (
                    low_score
                )

            # From all possible ways to expand hypos, choose nmerge.
            # Get the new nbest new tokens, parent hypos idx and scores.
            if self.always_merge_blank:
                blank_scores = candidate_scores[:, -1].reshape(bsz, nbest).clone()
                candidate_scores[:, -1] = low_score
                nmerge_scores, nmerge_idx_raveled = torch.topk(
                    candidate_scores.reshape([bsz, -1]),
                    nmerge - nbest,
                    sorted=False,
                )  # [B, M]
                # Add all blank extensions
                nmerge_idx_raveled = torch.cat(
                    [nmerge_idx_raveled, blank_multiplied], dim=1
                )
                nmerge_scores = torch.cat([nmerge_scores, blank_scores], dim=1)
            else:
                nmerge_scores, nmerge_idx_raveled = torch.topk(
                    candidate_scores.reshape([bsz, -1]), nmerge, sorted=False
                )  # [B, M]
            nbest_parent_hypos, nmerge_tokens = self.idx_1d_to_2d(
                nmerge_idx_raveled, candidate_scores.shape[1]
            )  # [B, M]
            scores = nmerge_scores.reshape(-1)  # [B*M]
            nmerge_tokens = nmerge_tokens.reshape(-1)  # [B*M]

            # Choosing topk will shuffle the indices of hypos. Keep track
            # of hypo states based on the new order.
            nbest_parent_hypos = nbest_parent_hypos.reshape(-1)
            raveled_idx = self.idx_2d_to_1d(ex_ind_nmerge, nbest_parent_hypos, nbest)
            predictor_out = predictor_out[raveled_idx]
            predictor_state_h = predictor_state_h[raveled_idx]
            predictor_state_c = predictor_state_c[raveled_idx]
            n_emitted_symbols = n_emitted_symbols[raveled_idx]
            tokens = torch.cat(
                [tokens[raveled_idx], nmerge_tokens.reshape(-1, 1)], dim=1
            )

            # Part C: Merge hypos with the same tokens

            # Merge hypos that map to the same non-blank tokens
            # We torch.unique to find groups of hypos that have exact same
            # non-blank tokens.
            # Since each hypo can have a different number of tokens, we may
            # have shape issues. To avoid that, we apply torch.unique on a hypo
            # hash. The hash should: a) be the same for hypos with the same ordered
            # non-blank tokens. b) be different for hypos with same but reordered
            # non-blank tokens. c) be different on all hypos that do not end with
            # blank.
            if nbest > 1:
                # Copy tokens into a tensor we can modify
                hash_copy = tokens.clone()

                # Mask hypos that don't end with blank, set them to be the
                # hypo idx + 1 + self.blank, this should lead to a
                # unique hash.
                nonblank_end_mask = tokens[:, -1] != self.blank
                hash_copy[nonblank_end_mask] = (
                    hypo_idx.transpose(1, 0)[nonblank_end_mask] + 1 + self.blank
                )

                # Reorder such that all non-blanks are in at the beginning
                rng = torch.arange(
                    0, hash_copy.shape[1], dtype=torch.int64, device=device
                )
                rng += 1
                sort_idx = torch.sort(
                    (hash_copy != self.blank) * rng, dim=1, descending=True
                )[1]
                hash_copy = hash_copy.gather(1, sort_idx)

                # Replace blank with zero
                hash_copy[hash_copy == self.blank] = 0

                # Multiply by (a function of) the sequence location to have
                # different hash for permutations
                hash_copy = hash_copy.to(torch.float64)
                rng = rng.to(torch.float64)
                hash_copy *= torch.sqrt(rng)

                # Hash the sequence by taking the mean of sqrt (would rarely fail)
                hash_copy = torch.sqrt(hash_copy).mean(dim=1)
                assert hash_copy.max() < max_hash, hash_copy

                # Reshape by example, add ex_separator1 to have different hash
                # for different examples
                hash_copy = hash_copy.reshape(bsz, nmerge) + ex_separator1

                # Find identical hypos based on the hash
                _, groups = torch.unique(hash_copy, sorted=True, return_inverse=True)
                groups += ex_separator2 - groups.min(dim=1)[0].reshape(-1, 1)
                groups = groups.reshape(-1)

                # Add scores for each group of identical hypos
                nmerge_scores_temp[:] = 0.0  # avoid nans after log
                nmerge_scores_temp.scatter_add_(
                    dim=0, index=groups, src=torch.exp(scores)
                )
                nmerge_scores_temp = torch.maximum(
                    torch.log(nmerge_scores_temp), low_score_t
                )  # Replace -inf with low_score again

                # Scatter the added scores back to their original position,
                # but only for one of the merged hypos in the group of identical
                # hypos. Set scores to -inf for the other locations in the group
                equal_mat = (
                    groups.reshape(-1, 1) == hypo_idx
                )  # [i, j] == True if groups[i] = j
                equal_mat = torch.cat([half, equal_mat.float()], dim=0)
                first_idx = torch.argmax(equal_mat, dim=0) - 1
                valid_mask = first_idx != -1
                scores[:] = low_score
                scores.scatter_(
                    dim=0,
                    index=first_idx[valid_mask],
                    src=nmerge_scores_temp[valid_mask],
                )

            # Part D: choose nbest
            if nmerge > nbest:
                if self.length_norm:
                    normalized_scores = self.apply_length_norm(
                        scores, tokens, low_score
                    )
                else:
                    normalized_scores = scores

                # Choose nbest out of the nmerge candidates
                _, nbest_idx_unraveled = torch.topk(
                    normalized_scores.reshape([bsz, nmerge]), nbest, sorted=False
                )

                # Flatten the indices to index into a [bsz * nmerge] tensor
                nbest_idx_raveled = self.idx_2d_to_1d(
                    ex_ind_nbest, nbest_idx_unraveled.reshape(-1), nmerge
                )

                # Choosing topk will shuffle the indices of hypos. Keep track
                # of hypo states based on the new order.
                scores = scores[nbest_idx_raveled]
                predictor_out = predictor_out[nbest_idx_raveled]
                predictor_state_h = predictor_state_h[nbest_idx_raveled]
                predictor_state_c = predictor_state_c[nbest_idx_raveled]
                n_emitted_symbols = n_emitted_symbols[nbest_idx_raveled]
                tokens = tokens[nbest_idx_raveled]

            # The last tokens emitted in the hypo
            nbest_tokens = tokens[:, -1]

            # Part E: Run predictor

            # Update predictor output and state for hypos ending with
            # non-blank and score > -inf
            non_blank_mask = torch.logical_and(
                nbest_tokens != self.blank,
                scores > low_score,
            )
            n_non_blank = non_blank_mask.sum()

            if n_non_blank > 0:
                predictor_inputs = nbest_tokens[non_blank_mask]
                predictor_state_h_inputs = predictor_state_h[non_blank_mask]
                predictor_state_c_inputs = predictor_state_c[non_blank_mask]

                # Run predictor
                new_predictor_out = self.text_frontend(predictor_inputs.unsqueeze(1))
                new_predictor_out, (new_predictor_state_h, new_predictor_state_c) = (
                    self.predictor(
                        new_predictor_out.contiguous(),
                        (
                            predictor_state_h_inputs.transpose(0, 1).contiguous(),
                            predictor_state_c_inputs.transpose(0, 1).contiguous(),
                        ),
                    )
                )

                # Scatter out/state back to the right places
                predictor_out[non_blank_mask] = new_predictor_out
                predictor_state_h[non_blank_mask] = new_predictor_state_h.transpose(
                    0, 1
                )
                predictor_state_c[non_blank_mask] = new_predictor_state_c.transpose(
                    0, 1
                )

            # Part F: Update hypo and example state before next iteration

            # Update n_emitted_symbols, encoder_t, active mask
            alignment = torch.cat([alignment, encoder_t.reshape(-1, 1)], dim=1)
            n_emitted_symbols[non_blank_mask] += 1
            active = torch.logical_and(
                non_blank_mask, n_emitted_symbols <= self.step_max_symbols
            )  # Allow emitting blank after already emitting max tokens
            next_t_mask = torch.all(
                torch.logical_not(active.reshape(bsz, nbest)), dim=1
            )
            next_t_mask = torch.logical_and(
                next_t_mask, encoder_t[::nbest] + 1 < encoder_lengths
            )
            next_t_mask = next_t_mask.repeat_interleave(nbest)  # Apply to nbest
            encoder_t[next_t_mask] += 1
            n_emitted_symbols[next_t_mask] = 0
            active[next_t_mask] = True

            # If nothing is active we are done
            itr += 1
            if not torch.any(active):
                done = True

        # Some hypos are merged and have a score of low_score. Hide tokens for those.
        neg_ind_score_mask = scores == low_score
        tokens[neg_ind_score_mask] = self.blank
        alignment[neg_ind_score_mask] = 0

        # Sort nbest according to score before returning
        scores = self.apply_length_norm(scores, tokens, low_score)
        scores[neg_ind_score_mask] = low_score  # Set back to low_score after norm
        scores, sort_order = scores.reshape(bsz, nbest).sort(dim=1, descending=True)
        tokens = tokens[(sort_order + ex_separator3).reshape(-1)].reshape(
            bsz, nbest, -1
        )
        # No need to sort alignment, it's identical for all hypos in the example.
        alignment = alignment.reshape(bsz, nbest, -1)

        return tokens, scores, alignment
