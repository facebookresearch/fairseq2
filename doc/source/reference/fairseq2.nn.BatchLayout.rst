.. _batch_layout:

fairseq2.nn.BatchLayout
=======================

.. currentmodule:: fairseq2.nn

``BatchLayout`` is the cornerstone of fairseq2 v0.5's unified batching system. It
consolidates both padded and packed batching strategies under a single, consistent
API, making it easier to work with variable-length sequences efficiently.

**Key Features:**

- **Unified API**: Single interface for both padded and packed batch modes
- **Memory Efficient**: Optimized handling of variable-length sequences
- **Torch.compile Compatible**: Full integration with PyTorch's compilation system
- **Dynamic Sequences**: Support for dynamic sequence lengths during training

.. autoclass:: BatchLayout
   :members:
   :undoc-members:
   :show-inheritance:
  
Creating BatchLayout
--------------------

**For Padded Batches:**

.. code-block:: python

    from fairseq2.nn import BatchLayout
    from fairseq2.device import get_default_device
    import torch

    device = get_default_device()  # "cuda" or "cpu"
    # Create a padded batch layout
    # Shape: (batch_size=4, max_seq_len=6)
    # Individual sequence lengths: [4, 2, 3, 5]
    batch_layout = BatchLayout(
        shape=(4, 6),
        seq_lens=[4, 2, 3, 5],
        packed=False,
        device=device
    )

    print(f"Width: {batch_layout.width}")  # 6
    print(f"Sequence lengths: {list(batch_layout.seq_lens)}")  # [4, 2, 3, 5]
    print(f"Is padded: {batch_layout.padded}")  # True
    print(f"Is packed: {batch_layout.packed}")  # False

**For Packed Batches:**

.. code-block:: python

    # Create a packed batch layout
    # Total elements: 14 (4+2+3+5), sequences: [4, 2, 3, 5]
    packed_layout = BatchLayout(
        shape=(14,),  # 1D shape for packed mode
        seq_lens=[4, 2, 3, 5],
        packed=True,
        device=torch.device("cpu")
    )

    print(f"Sequence begin indices: {list(packed_layout.seq_begin_indices)}")
    # [0, 4, 6, 9, 14]
    print(f"Is packed: {packed_layout.packed}")  # True

    # For packed batches, get sequence boundaries
    if packed_layout.packed:
        seq_boundaries = packed_layout.seq_begin_indices_pt
        print(f"Sequence boundaries: {seq_boundaries}")


**From Existing Tensors:**

.. code-block:: python

    # Create from existing batch tensor
    batch_tensor = torch.randn(4, 6, 512)  # (batch, seq, features)
    batch_layout = BatchLayout.of(
        batch_tensor,
        seq_lens=[4, 2, 3, 5],
        packed=False
    )

Working with Position Indices and Masks
---------------------------------------

BatchLayout automatically computes position indices and masking information:

.. code-block:: python

    batch_layout = BatchLayout((4, 6), seq_lens=[4, 2, 3, 5])

    # Position indices for each element (-1 indicates padding)
    pos_indices = batch_layout.position_indices
    # Shape: (4, 6)
    # [[0, 1, 2, 3, -1, -1],
    #  [0, 1, -1, -1, -1, -1],
    #  [0, 1, 2, -1, -1, -1],
    #  [0, 1, 2, 3, 4, -1]]

    # Create padding mask (True for valid positions)
    padding_mask = pos_indices >= 0

    # Apply mask to hide padding positions
    from fairseq2.nn.utils.mask import apply_mask
    masked_batch = apply_mask(batch_tensor, padding_mask, fill_value=0.0)

Sequence Information
--------------------

BatchLayout provides comprehensive sequence metadata:

.. code-block:: python

    batch_layout = BatchLayout((4, 6), seq_lens=[4, 2, 3, 5])

    # Sequence properties
    print(f"Min sequence length: {batch_layout.min_seq_len}")  # 2
    print(f"Max sequence length: {batch_layout.max_seq_len}")  # 5
    print(f"Batch width: {batch_layout.width}")  # 6

    # Sequence lengths as tensors (for GPU operations)
    seq_lens_tensor = batch_layout.seq_lens_pt  # torch.Tensor([4, 2, 3, 5])

Integration with Neural Network Layers
--------------------------------------

BatchLayout is designed to work seamlessly with all fairseq2 neural network layers:

.. code-block:: python

    import torch.nn as nn
    from fairseq2.nn import BatchLayout

    # Example attention layer that works with BatchLayout
    class AttentionLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)

        def forward(self, x, batch_layout: BatchLayout):
            # Create attention mask from batch layout
            seq_len = x.size(1)
            attn_mask = batch_layout.position_indices < 0  # True for padding

            # Expand mask for attention heads if needed
            if batch_layout.packed:
                # Handle packed sequences differently
                # Use sequence boundaries for efficient attention
                pass
            else:
                # Standard padded attention with mask
                output, _ = self.attention(x, x, x, key_padding_mask=attn_mask)

            return output

Torch.compile Integration
-------------------------

BatchLayout is fully compatible with PyTorch's compilation system:

.. code-block:: python

    import torch
    from fairseq2.nn import BatchLayout

    @torch.compile
    def process_batch(batch_tensor, batch_layout: BatchLayout):
        # Position indices are automatically marked as dynamic
        pos_indices = batch_layout.position_indices

        # Use compiled operations with dynamic sequences
        mask = pos_indices >= 0
        return batch_tensor.masked_fill(~mask.unsqueeze(-1), 0.0)

    # The compiled function handles dynamic sequence lengths efficiently
    batch_layout = BatchLayout((4, 6), seq_lens=[4, 2, 3, 5])
    batch_tensor = torch.randn(4, 6, 512)
    result = process_batch(batch_tensor, batch_layout)

Performance Considerations
--------------------------

**Packed vs Padded Trade-offs:**

- **Packed**: More memory efficient, better for variable lengths, requires careful indexing
- **Padded**: Simpler operations, better for uniform attention, may waste memory on padding

.. code-block:: python

    # Memory comparison
    seq_lens = [100, 50, 75, 25]  # Variable length sequences

    # Padded: allocates max_len for all sequences
    padded_layout = BatchLayout((4, 100), seq_lens=seq_lens, packed=False)
    padded_memory = 4 * 100  # 400 positions

    # Packed: only allocates needed positions
    packed_layout = BatchLayout((250,), seq_lens=seq_lens, packed=True)
    packed_memory = sum(seq_lens)  # 250 positions (37.5% savings)
