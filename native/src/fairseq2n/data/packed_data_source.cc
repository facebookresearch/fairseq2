// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/packed_data_source.h"

#include <utility>
#include <vector>

#include <ATen/Functions.h>

#include "fairseq2n/data/data.h"
#include "fairseq2n/data/detail/exception.h"

namespace fairseq2n::detail {

std::optional<data>
packed_data_source::next()
{
    at::Tensor batch{};

    data_list seq_lens{};

    std::int64_t fill_pos = 0;

    bool filled = false;

    while (true) {
        at::Tensor tensor{};

        std::optional<data> maybe_example{};

        if (remainder_.defined()) {
            tensor = std::exchange(remainder_, {});
        } else {
           maybe_example = inner_->next();
           if (!maybe_example)
               break;

            data &example = *maybe_example;

            if (!example.is_tensor())
                throw_data_pipeline_error(std::move(maybe_example), /*recoverable=*/true,
                    "The pack operation can only be used with tensor inputs.");

            tensor = example.as_tensor();

            std::int64_t ndim = tensor.dim();
            if (ndim != 1)
                throw_data_pipeline_error(std::move(maybe_example), /*recoverable=*/true,
                    "The input tensors to the pack operation must be one-dimensional.");
        }

        if (!batch.defined()) {
            batch = at::full({capacity_}, pad_value_,
                at::dtype(tensor.dtype()).device(tensor.device()).pinned_memory(pinned_memory_));
        } else {
            tensor = tensor.to(batch.device());
        }

        std::int64_t numel = tensor.numel();
        if (numel == 0)
            continue;

        std::int64_t max_fill_size = std::min(capacity_ - fill_pos, max_seq_len_);

        std::int64_t fill_size = std::min(numel, max_fill_size);

        if (numel > fill_size) {
            if (truncate_) {
                if (fill_pos + fill_size == capacity_)
                    remainder_ = tensor.slice(/*dim=*/0, /*start=*/fill_size - 1);
                else
                    remainder_ = tensor.slice(/*dim=*/0, /*start=*/fill_size);

                tensor = tensor.slice(/*dim=*/0, /*start=*/0, /*end=*/fill_size);
            } else {
                if (fill_pos == 0)
                    throw_data_pipeline_error(std::move(maybe_example), /*recoverable=*/true,
                        "The input tensor is too long to pack.");

                remainder_ = std::move(tensor);

                break;
            }
        } else {
            remainder_.reset();
        }

        std::int64_t end_pos = fill_pos + fill_size;

        batch.slice(/*dim=*/0, /*start=*/fill_pos, /*end=*/end_pos) = tensor;

        seq_lens.push_back(fill_size);

        if (end_pos == capacity_) {
            filled = true;

            break;
        }

        fill_pos = end_pos;
    }

    if (truncate_ && !filled) {
        if (drop_remainder_)
            return std::nullopt;
    }

    if (batch.defined()) {
        data_dict output{};

        output.emplace("seqs", std::move(batch));
        output.emplace("seq_lens", std::move(seq_lens));

        return output;
    }

    return std::nullopt;
}

void
packed_data_source::reset(bool reset_rng)
{
    remainder_.reset();

    inner_->reset(reset_rng);
}

void
packed_data_source::record_position(tape &t, bool strict) const
{
    if (remainder_.defined()) {
        t.record(true);

        t.record(remainder_);
    } else {
        t.record(false);
    }

    inner_->record_position(t, strict);
}

void
packed_data_source::reload_position(tape &t, bool strict)
{
    bool has_remainder = t.read<bool>();
    if (has_remainder)
        remainder_ = t.read<at::Tensor>();

    inner_->reload_position(t, strict);
}

data_source_finitude_type
packed_data_source::finitude_type() const noexcept
{
    return inner_->finitude_type();
}

}  // namespace fairseq2n::detail
