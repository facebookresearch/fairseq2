// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/collater.h"

#include <vector>
#include <utility>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include "fairseq2/native/exception.h"
#include "fairseq2/native/float.h"

namespace fairseq2 {

data
collater::process(data &&d) const
{
    if (!d.is_list())
        return d;

    std::vector<data> bucket = d.as_list();

    if (bucket.front().is_tensor()) {
        std::vector<at::Tensor> s{};
        s.reserve(bucket.size());

        for (auto &v : bucket)
            s.emplace_back(v.as_tensor());

        if (pad_idx_) {
            return at::pad_sequence(s, /*batch_first=*/true, static_cast<float64>(*pad_idx_));
        }
        return at::stack(s);
    }

    if (bucket.front().is_list()) {
        // We have a list of tuples of tensors,
        // let's return a tuple of batched tensors.
        auto bs = bucket.size();
        auto n_cols = bucket.front().as_list().size();

        std::vector<std::vector<data>> columns(n_cols);
        for (auto column : columns)
            column.reserve(bs);

        for (auto maybe_row: bucket) {
            if (!maybe_row.is_list())
                throw not_supported_error{"All rows need to have the same type to be used."};
            std::vector<data> row = maybe_row.as_list();
            if (row.size() != n_cols)
                throw not_supported_error{"All rows need to have the same size to be used."};

            for (std::size_t col = 0; col < n_cols; ++col) {
                columns.at(col).emplace_back(std::move(row.at(col)));
            }
        }

        std::vector<data> collated_columns = {};
        collated_columns.reserve(n_cols);
        for (std::size_t col = 0; col < n_cols; ++col) {
            collated_columns.emplace_back(process(columns[col]));
        }
        return collated_columns;
    }

    return bucket;
}

}  // namespace fairseq2
