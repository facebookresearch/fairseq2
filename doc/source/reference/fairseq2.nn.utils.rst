.. _nn_utils:

fairseq2.nn.utils
=================

Masking Utilities
-----------------

.. currentmodule:: fairseq2.nn.utils.mask

.. autofunction:: apply_mask

Apply boolean masks to sequences with proper broadcasting.

.. autofunction:: compute_row_mask

Generate random row masks for training objectives like MLM.

**Example Usage:**

.. code-block:: python

    from fairseq2.nn.utils.mask import apply_mask, compute_row_mask
    from fairseq2.nn import BatchLayout

    # Create batch with layout
    batch_tensor = torch.randn(4, 16, 512)
    batch_layout = BatchLayout((4, 16), seq_lens=[16, 14, 15, 16])

    # Apply padding mask
    padding_mask = batch_layout.position_indices >= 0
    masked_batch = apply_mask(batch_tensor, padding_mask, fill_value=0.0)

    # Generate random mask for MLM training
    random_mask = compute_row_mask(
        shape=(4, 16),
        span_len=2, # must be no greater than the row_lens [4, 4, 3, 5]
        max_mask_prob=0.65,
        device="cpu",
        row_lens=batch_layout.seq_lens_pt
    )
