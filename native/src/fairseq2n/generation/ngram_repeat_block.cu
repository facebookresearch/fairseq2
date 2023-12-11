// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <ATen/Tensor.h>
#include <vector>

#include "fairseq2n/float.h"

namespace fairseq2n {
namespace detail {
namespace {

// Ban repeated ngrams of length = 'no_repeat_ngram_size'
__global__ void
ban_repeated_tokens(
    std::int64_t * __restrict__ tokens,
    float32 * __restrict__ lprobs,
    std::int64_t max_predict_len,
    std::int64_t vocab_size,
    std::int64_t no_repeat_ngram_size)
{
  auto row = blockIdx.x;
  auto col = threadIdx.x;
  auto start = row * (max_predict_len) + col;
  // Each thread compares ngram starting from
  // thread index with final ngram starting from
  // step - no_repeat_ngram_size +2
  auto check_start_pos = blockDim.x;
  auto lprob_start = row * vocab_size;
  bool is_banned = true;
  extern __shared__ std::int64_t tokens_shm[];
  tokens_shm[col] = tokens[start];
  if (col == blockDim.x - 1) {
    for (std::int64_t i = 1; i < no_repeat_ngram_size; i++) {
      if (col + i < max_predict_len) {
        tokens_shm[col + i] = tokens[start + i];
      }
    }
  }
  __syncthreads();

  for (int k = 0; k < no_repeat_ngram_size - 1; k++) {
    if (tokens_shm[col + k] != tokens_shm[check_start_pos + k]) {
      is_banned = false;
    }
  }
  if (is_banned == true) {
    auto token_to_be_banned = tokens_shm[col + no_repeat_ngram_size - 1];
    lprobs[lprob_start + token_to_be_banned] = -INFINITY;
  }
}


}  // namespace
}  // namespace detail

// Allocate blocks and threads based on
// batch size and sequence length and launch
// kernel
at::Tensor
ngram_repeat_block_cuda(
    at::Tensor tokens,
    at::Tensor lprobs,
    std::int64_t bsz,
    std::int64_t step,
    std::int64_t beam_size,
    std::int64_t no_repeat_ngram_size)
{
    std::int64_t threads = step - no_repeat_ngram_size + 2;

    if (threads <= 0)
        return lprobs;

    std::int64_t max_predict_len = tokens.size(1);
    std::int64_t vocab_size = lprobs.size(1);
  
    auto token_ptr = tokens.data_ptr<std::int64_t>();
    auto lprob_ptr = lprobs.data_ptr<float32>();

    std::int64_t blocks = bsz * beam_size;

    std::size_t shared_mem_size = (static_cast<std::size_t>(step) + 1) * sizeof(std::int64_t);

    // Launching N blocks where N is number of samples in a batch (beams*bsz)
    // Launching T threads where T is number of previous ngrams in a sample
    // Allocating shared mem per block for fastser access of input tokens since
    // each token will be accessed N times to compare with current Ngram where
    // N is Ngram size.
    detail::ban_repeated_tokens<<<
        static_cast<std::uint32_t>(blocks), static_cast<std::uint32_t>(threads), shared_mem_size>>>(
        token_ptr, lprob_ptr, max_predict_len, vocab_size, no_repeat_ngram_size);

    return lprobs;
}

}  // namespace fairseq2n
