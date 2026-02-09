# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


# TODO(Phase 2): Implement Gemma3n tokenizer
# Gemma3n uses the same tokenizer as Gemma (SentencePiece-based).
# We can likely reuse fairseq2's Gemma tokenizer implementation or
# adapt it with minimal changes.
#
# Key considerations:
# - Vocab size: 262,400 tokens
# - Special tokens: PAD (0), EOS (1), BOS (2)
# - Potentially different special tokens for multimodal (image/audio markers)
#
# Implementation approach:
# 1. Check if fairseq2 has a Gemma tokenizer we can reuse
# 2. If not, implement HuggingFaceTokenizer wrapper
# 3. Handle multimodal special tokens if needed

__all__: list[str] = []
