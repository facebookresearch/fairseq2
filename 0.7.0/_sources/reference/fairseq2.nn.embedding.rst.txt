.. _embedding:

fairseq2.nn.embedding
=====================

.. currentmodule:: fairseq2.nn

.. autoclass:: Embedding
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: StandardEmbedding
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ShardedEmbedding
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: init_scaled_embedding

**Example Usage:**

.. code-block:: python

    from fairseq2.nn import StandardEmbedding, init_scaled_embedding

    # Create token embeddings
    embed = StandardEmbedding(
        num_embeddings=32000,  # vocabulary size
        embed_dim=512,
        pad_idx=0,
        init_fn=init_scaled_embedding
    )

    # Use with BatchLayout
    tokens = torch.randint(0, 32000, (4, 6))  # (batch, seq)
    batch_layout = BatchLayout.of(tokens, seq_lens=[4, 2, 3, 5])

    embeddings = embed(tokens)  # (4, 6, 512)
