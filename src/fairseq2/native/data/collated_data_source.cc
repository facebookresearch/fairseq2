// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/collated_data_source.h"

#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include "fairseq2/native/exception.h"

namespace fairseq2::detail {

std::optional<data>
collated_data_source::next()
{
    std::optional<data> d = inner_->next();
    if (!d)
        return std::nullopt;

    if (d->is_list())
        return collate(d->as_list());
    else
        return d;
}

data
collated_data_source::collate(const std::vector<data> &batch) {
    if (batch.front().is_tensor()) {
        std::vector<at::Tensor> s{};
        s.reserve(batch.size());

        for (auto &v : batch)
            s.emplace_back(v.as_tensor());

        if (pad_idx_) {
            return at::pad_sequence(s, /*batch_first=*/true, *pad_idx_);
        }
        return at::stack(s);
    }

    if (batch.front().is_list()) {
        // We have a list of tuples of tensors,
        // let's return a tuple of batched tensors.
        auto bs = batch.size();
        auto n_cols = batch.front().as_list().size();

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

        std::vector<data> collated_columns = {};
        collated_columns.reserve(n_cols);
        for (std::size_t col = 0; col < n_cols; ++col) {
            collated_columns.emplace_back(collate(columns[col]));
        }
        return collated_columns;
    }

    return batch;
}

void
collated_data_source::reset()
{
    inner_->reset();
}

void
collated_data_source::record_position(tape &t) const
{
    inner_->record_position(t);
}

void
collated_data_source::reload_position(tape &t)
{
    inner_->reload_position(t);
}

}  // namespace fairseq2::detail
