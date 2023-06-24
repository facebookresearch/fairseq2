// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/processors/list_processor.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include <fmt/core.h>
#include <oneapi/tbb.h>

namespace fairseq2 {

list_processor::list_processor(
    std::vector<std::shared_ptr<const data_processor>> processors,
    std::optional<std::vector<std::size_t>> indices,
    bool disable_parallelism)
  : processors_{std::move(processors)}, disable_parallelism_{disable_parallelism}
{
    if (!indices || indices->empty())
        return;

    indices_ = *std::move(indices);

    if (processors_.size() != indices_.size())
        throw std::invalid_argument{
            fmt::format("`processors` and `indices` must have the same length, but have the lengths {0} and {1} instead.", processors_.size(), indices_.size())};

    for (std::size_t i = 1, last = indices_.front(); i < indices_.size(); ++i) {
        if (indices_[i] <= last)
            throw std::invalid_argument{"`indices` must be unique and in sorted order."};

        last = indices_[i];
    }
}

data
list_processor::operator()(const data &d) const
{
    validate(d);

    const std::vector<data> &src = d.as_list();

    std::vector<data> dst(src.size());

    if (indices_.empty()) {
        // We have no index, meaning each element has a corresponding processor.
        auto apply_procs = [this, &src, &dst](const tbb::blocked_range<std::size_t> &range) {
            for (std::size_t i = range.begin(); i < range.end(); ++i)
                dst[i] = (*processors_[i])(src[i]);
        };

        parallel_for(apply_procs, src.size());
    } else {
        auto apply_procs = [this, &src, &dst](const tbb::blocked_range<std::size_t> &range) {
            // Since the range can start from an arbitrary position, we have to
            // look up the closest index that has a processor.
            auto pos = std::lower_bound(indices_.begin(), indices_.end(), range.begin());

            auto j = static_cast<std::size_t>(pos - indices_.begin());

            for (std::size_t i = range.begin(); i < range.end(); ++i) {
                // Do a copy if the index has no corresponding processor.
                if (j == indices_.size() || i != indices_[j])
                    dst[i] = src[i];
                else
                    // Otherwise, call the processor.
                    dst[i] = (*processors_[j++])(src[i]);
            }
        };

        parallel_for(apply_procs, src.size());
    }

    return dst;
}

data
list_processor::operator()(data &&d) const
{
    validate(d);

    std::vector<data> &lst = d.as_list();

    if (indices_.empty()) {
        // We have no index, meaning each element has a corresponding processor.
        auto apply_procs = [this, &lst](const tbb::blocked_range<std::size_t> &range) {
            for (std::size_t i = range.begin(); i < range.end(); ++i)
                lst[i] = (*processors_[i])(std::move(lst[i]));
        };

        parallel_for(apply_procs, lst.size());
    } else {
        auto apply_procs = [this, &lst](const tbb::blocked_range<std::size_t> &range) {
            // Iterate only over the elements that have a corresponding processor.
            for (std::size_t j = range.begin(); j < range.end(); ++j)
                lst[indices_[j]] = (*processors_[j])(std::move(lst[indices_[j]]));
        };

        parallel_for(apply_procs, indices_.size());
    }

    return d;
}

void
list_processor::validate(const data &d) const
{
    if (!d.is_list())
        throw std::invalid_argument{"The input data must be of type list."};

    const std::vector<data> &lst = d.as_list();

    if (indices_.empty()) {
        if (processors_.size() != lst.size())
            throw std::invalid_argument{
                fmt::format("The length of the input list must equal {1}, but is {0} instead.", lst.size(), processors_.size())};
    } else {
        if (indices_.back() >= lst.size())
            throw std::invalid_argument{
                fmt::format("The length of the input list must be longer than {1}, but is {0} instead.", lst.size(), indices_.back())};
    }
}

template <typename F>
void
list_processor::parallel_for(F &f, std::size_t n) const
{
    tbb::blocked_range<std::size_t> full_range{0, n};

    if (disable_parallelism_ || n == 1)
        f(full_range);
    else
        tbb::parallel_for(full_range, f);
}

}  // namespace fairseq2
