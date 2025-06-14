from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    create_block_mask,
)

from fairseq2.device import Device
from fairseq2.error import NotSupportedError
from fairseq2.nn import BatchLayout

# isort: split

from fairseq2.models.transformer._attention_bias import (
    AttentionBias,
    CausalAttentionBias,
    IdentityBias,
)

MaskFunction: TypeAlias = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]

BLOCK_MASK_CACHE_MAX_SIZE = 1000


def _causal_mask_fn(q_lens: Tensor, kv_lens: Tensor) -> MaskFunction:
    """Creates a causal mask function."""

    def mask_fn(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        # Get sequence lengths for this batch
        q_len = q_lens[b]
        kv_len = kv_lens[b]

        # Calculate diagonal offset
        d = kv_len - q_len

        return q_idx >= kv_idx - d

    return mask_fn


def _sliding_window_causal_mask_fn(
    window_size: int,
    q_lens: Tensor,
    kv_lens: Tensor,
) -> MaskFunction:
    """Creates a sliding window causal mask functions."""

    def mask_fn(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        # Get sequence lengths for this batch
        q_len = q_lens[b]
        kv_len = kv_lens[b]

        # Calculate diagonal offset
        d = kv_len - q_len

        # For window_size=1, only allow the exact diagonal position
        if window_size == 1:
            return q_idx == kv_idx - d
        else:
            # For larger windows, use the range logic
            causal_mask = q_idx >= kv_idx - d
            window_mask = q_idx >= kv_idx - d - window_size + 1
            return causal_mask & window_mask

    return mask_fn


def _offsets_to_doc_ids_tensor(offsets: Tensor) -> Tensor:
    """Convert offsets to document IDs for packed sequences."""
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def _create_packed_mask_fn(
    seq_begin_indices: Tensor,
    keys_begin_indices: Tensor,
    base_mask_fn: MaskFunction | None = None,
) -> MaskFunction:
    """Creates a mask function for packed sequences using document-based masking."""
    # Create document IDs for queries and keys
    query_doc_ids = _offsets_to_doc_ids_tensor(seq_begin_indices)
    key_doc_ids = _offsets_to_doc_ids_tensor(keys_begin_indices)

    def packed_mask_fn(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        # Check if query and key belong to the same document
        same_doc = query_doc_ids[q_idx] == key_doc_ids[kv_idx]

        # Convert global indices to logical positions within documents
        q_doc_id = query_doc_ids[q_idx]
        kv_doc_id = key_doc_ids[kv_idx]
        q_logical = q_idx - seq_begin_indices[q_doc_id]
        kv_logical = kv_idx - keys_begin_indices[kv_doc_id]

        # Apply base mask (e.g., causal) to logical positions
        if base_mask_fn is not None:
            inner_mask = base_mask_fn(b, h, q_logical, kv_logical)
            return same_doc & inner_mask
        else:
            return same_doc

    return packed_mask_fn


def _create_padding_mask_fn(q_lens: Tensor, kv_lens: Tensor) -> MaskFunction:
    """Creates a padding mask function that masks out padding tokens."""

    def padding_mask_fn(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        q_valid = q_idx < q_lens[b]
        kv_valid = kv_idx < kv_lens[b]
        return q_valid & kv_valid

    return padding_mask_fn


def _create_composed_mask(
    bias: AttentionBias,
    seqs_layout: BatchLayout,
    keys_layout: BatchLayout,
    device: Device,
) -> BlockMask | None:
    """Creates a composed mask using and_masks for combining multiple mask functions."""
    masks = []

    if seqs_layout.packed:
        # For packed sequences, create the base mask function first
        base_mask_fn = None

        # Add attention bias mask as base mask
        if isinstance(bias, CausalAttentionBias):
            attn_window_len = bias.attn_window_len
            if attn_window_len is not None:
                base_mask_fn = _sliding_window_causal_mask_fn(
                    attn_window_len,
                    seqs_layout.seq_lens_pt,
                    keys_layout.seq_lens_pt,
                )
            else:
                base_mask_fn = _causal_mask_fn(
                    seqs_layout.seq_lens_pt,
                    keys_layout.seq_lens_pt,
                )
        elif not isinstance(bias, IdentityBias):
            raise NotSupportedError(f"Unsupported bias type: {bias}")

        # Create the packed sequence mask that incorporates the base mask
        packed_mask = _create_packed_mask_fn(
            seqs_layout.seq_begin_indices_pt,
            keys_layout.seq_begin_indices_pt,
            base_mask_fn,
        )
        masks.append(packed_mask)
    else:
        # Standard batch format - handle bias and padding separately
        if isinstance(bias, CausalAttentionBias):
            attn_window_len = bias.attn_window_len
            if attn_window_len is not None:
                masks.append(
                    _sliding_window_causal_mask_fn(
                        attn_window_len,
                        seqs_layout.seq_lens_pt,
                        keys_layout.seq_lens_pt,
                    )
                )
            else:
                masks.append(
                    _causal_mask_fn(
                        seqs_layout.seq_lens_pt,
                        keys_layout.seq_lens_pt,
                    )
                )
        elif not isinstance(bias, IdentityBias):
            raise NotSupportedError(f"Unsupported bias type: {bias}")

    # Add padding mask
    if seqs_layout.padded or keys_layout.padded:
        masks.append(
            _create_padding_mask_fn(seqs_layout.seq_lens_pt, keys_layout.seq_lens_pt)
        )

    # Compose masks
    mask_fn = None
    if len(masks) == 0:
        return None
    elif len(masks) == 1:
        mask_fn = masks[0]
    else:
        mask_fn = and_masks(*masks)

    if seqs_layout.packed:
        total_seq_len = int(seqs_layout.seq_begin_indices_pt[-1].item())
        total_keys_len = int(keys_layout.seq_begin_indices_pt[-1].item())
        batch_size = 1
    else:
        total_seq_len = seqs_layout.max_seq_len
        total_keys_len = keys_layout.max_seq_len
        batch_size = len(seqs_layout.seq_lens)

    # Create the block mask
    block_mask = create_block_mask(
        mask_fn,
        B=batch_size,
        H=None,
        Q_LEN=total_seq_len,
        KV_LEN=total_keys_len,
        device=str(device),
    )
    return block_mask


@dataclass
class BlockMaskCacheKey:
    """Key for caching block masks."""

    bias_type: str
    batch_size: int
    seq_len: int
    keys_len: int
    packed: bool
    attn_window_len: int | None = None

    def __hash__(self) -> int:
        return hash(
            (
                self.bias_type,
                self.batch_size,
                self.seq_len,
                self.keys_len,
                self.packed,
                self.attn_window_len,
            )
        )


class BlockMaskCache:
    """
    Cache for block masks to avoid recomputation across layers and (possibly) training
    steps.
    """

    def __init__(self) -> None:
        self._cache: dict[BlockMaskCacheKey, BlockMask | None] = {}

    def get_or_create_mask(
        self,
        bias: AttentionBias,
        seqs_layout: BatchLayout,
        keys_layout: BatchLayout,
        device: Device,
    ) -> BlockMask | None:
        """Get cached mask or create new one."""

        # Create cache key
        bias_type = type(bias).__name__
        attn_window_len = None
        if isinstance(bias, CausalAttentionBias):
            attn_window_len = bias.attn_window_len

        if seqs_layout.packed:
            batch_size = 1
            seq_len = int(seqs_layout.seq_begin_indices[-1])
            keys_len = int(keys_layout.seq_begin_indices[-1])
        else:
            batch_size = len(seqs_layout.seq_lens)
            seq_len = seqs_layout.max_seq_len
            keys_len = keys_layout.max_seq_len

        cache_key = BlockMaskCacheKey(
            bias_type=bias_type,
            batch_size=batch_size,
            seq_len=seq_len,
            keys_len=keys_len,
            packed=seqs_layout.packed,
            attn_window_len=attn_window_len,
        )

        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Create new mask
        block_mask = _create_composed_mask(
            bias,
            seqs_layout,
            keys_layout,
            device,
        )

        if len(self._cache) < BLOCK_MASK_CACHE_MAX_SIZE:
            self._cache[cache_key] = block_mask

        return block_mask

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
