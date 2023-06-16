// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/mapped_data_source.h"
#include <stdexcept>

#include <oneapi/tbb.h>

namespace fairseq2::detail {

std::optional<data>
mapped_data_source::next()
{
    // Mono-threaded behavior
    std::optional<data> d;
    if (chunk_size_ <= 1) {
        d = inner_->next();
        if (!d)
            return {};
        try {
            return fn_(*std::move(d));
        } catch (const data_pipeline_error &) {
            throw;
        } catch (const std::invalid_argument &) {
            data_pipeline_error::throw_nested("The map function has failed.", std::move(d));
        } catch (...) {
            data_pipeline_error::throw_nested("The map function has failed.");
        }
    }

    // If we're out of precomputed items, fetch some inputs, and compute results in parallel.
    if (buffer_iter_ == buffer_.end()) {
        buffer_.clear();
        buffer_iter_ = buffer_.begin();

        // Prefetch inputs into buffer_, because we can't call "next" in parallel
        for (std::size_t i = 0; i < chunk_size_; ++i) {
            d = inner_->next();
            if (!d)
                break;
            buffer_.emplace_back(std::move(*d));
        }

        // Inner data source is exhausted, let's return.
        if (buffer_.empty())
            return {};

        // Process fetched items in parallel.
        auto apply_fn = [this](const tbb::blocked_range<std::size_t> &rng) {
            for (auto i = rng.begin(); i < rng.end(); ++i) {
                this->map_at(i);
            }
        };

        parallel_for(tbb::blocked_range<std::size_t>(0, buffer_.size()), apply_fn);
    }

    // Yield previously computed items
    return std::move(*buffer_iter_++);
}

void
mapped_data_source::reset()
{
    inner_->reset();
}

void
mapped_data_source::record_position(tape &t) const
{
    inner_->record_position(t);
}

void
mapped_data_source::reload_position(tape &t)
{
    inner_->reload_position(t);
}

void mapped_data_source::map_at(std::size_t i) {
    try {
        // Apply fn in place
        buffer_[i] = fn_(std::move(buffer_[i]));
    } catch (const data_pipeline_error &) {
        throw;
    } catch (const std::invalid_argument &) {
        data_pipeline_error::throw_nested("The map function has failed.");
    } catch (...) {
        data_pipeline_error::throw_nested("The map function has failed.");
    }
}

}  // namespace fairseq2::detail
