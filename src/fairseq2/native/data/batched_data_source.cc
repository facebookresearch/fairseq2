// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/batched_data_source.h"

#include <vector>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include "fairseq2/native/exception.h"
#include "fairseq2/native/data/data_pipeline.h"

namespace fairseq2::detail {

std::optional<data>
batched_data_source::next()
{
    std::vector<data> batch{};

    batch.reserve(batch_size_);

    for (std::size_t i = 0; i < batch_size_; i++) {
        std::optional<data> d = inner_->next();
        if (!d)
            break;

        batch.emplace_back(*std::move(d));
    }

    if (batch.empty())
        return std::nullopt;

    if (drop_remainder_ && batch.size() < batch_size_)
        return std::nullopt;

    return make_batch(std::move(batch));
}

data
batched_data_source::make_batch(std::vector<data> batch) {
    if (batch.front().is_tensor()) {
        std::vector<at::Tensor> s{};
        s.reserve(batch.size());

        for (auto &v : batch)
            s.emplace_back(v.as_tensor());

        if (current_pad_idx_) {
            return at::pad_sequence(s, /*batch_first=*/true, *current_pad_idx_);
        }
        return at::stack(s);
    }

    if (batch.front().is_list()) {
        // We have a list of tuples of tensors,
        // let's return a tuple of batched tensors.
        auto bs = batch.size();
        auto n_cols = batch.front().as_list().size();
        if (pad_idx_.size() > 1 && pad_idx_.size() != n_cols)
                throw data_pipeline_error{"Received input with mismatched number of columns in `batch(pad_idx=[...])`."};

        std::vector<std::vector<data>> columns(n_cols);
        for (auto column : columns)
            column.reserve(bs);

        for (auto maybe_row: batch) {
            if (!maybe_row.is_list())
                throw not_supported_error{"All rows need to have the same type to be used with `batch`."};
            std::vector<data> row = maybe_row.as_list();
            if (row.size() != n_cols)
                throw not_supported_error{"All rows need to have the same size to be used with `batch`."};

            for (std::size_t col = 0; col < n_cols; ++col) {
                columns.at(col).emplace_back(std::move(row.at(col)));
            }
        }

        std::vector<data> batched_columns = {};
        batched_columns.reserve(n_cols);
        for (std::size_t col = 0; col < n_cols; ++col) {
            if (pad_idx_.size() == n_cols)
                current_pad_idx_ = pad_idx_[col];
            batched_columns.emplace_back(make_batch(columns[col]));
        }
        return batched_columns;
    }

    return batch;
}

std::size_t
batched_data_source::skip(std::size_t num_examples)
{
    return inner_->skip(num_examples * batch_size_) / batch_size_;
}

void
batched_data_source::reset()
{
    inner_->reset();
}

void
batched_data_source::record_position(tape &t) const
{
    inner_->record_position(t);
}

void
batched_data_source::reload_position(tape &t)
{
    inner_->reload_position(t);
}

}  // namespace fairseq2::detail
