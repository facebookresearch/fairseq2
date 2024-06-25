// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/bucket_by_length_data_source.h"

#include "fairseq2n/data/detail/exception.h"

namespace fairseq2n::detail {

bucket_by_length_data_source::bucket_by_length_data_source(
    std::unique_ptr<data_source> &&inner,
    std::vector<std::pair<std::size_t, std::size_t>> &&bucket_sizes,
    data_length_fn &&fn,
    std::size_t min_data_len,
    bool skip_below_min_examples,
    bool skip_above_max_examples,
    bool drop_remainder)
  : inner_{std::move(inner)},
    bucket_sizes_(std::move(bucket_sizes)),
    data_length_fn_{std::move(fn)},
    min_data_len_{min_data_len},
    max_data_len_{bucket_sizes_.back().second},
    skip_below_min_examples_{skip_below_min_examples},
    skip_above_max_examples_{skip_above_max_examples},
    drop_remainder_{drop_remainder}
{
    buckets_.reserve(bucket_sizes_.size());

    for (auto [bucket_batch_size, bucket_data_len] : bucket_sizes_)
        buckets_.emplace_back().reserve(bucket_batch_size);
}

std::optional<data>
bucket_by_length_data_source::next()
{
    while (std::optional<data> maybe_example = inner_->next()) {
        data &example = *maybe_example;

        std::size_t data_len = data_length_fn_(example);
        if (data_len < min_data_len_) {
            if (!skip_below_min_examples_)
                throw_data_pipeline_error(std::move(maybe_example), /*recoverable=*/true,
                    "The length of the input data must be greater than or equal to the minimum bucket data length ({}), but is {} instead.", min_data_len_, data_len);

            // TODO(balioglu): log

            continue;
        }

        if (data_len > max_data_len_) {
            if (!skip_above_max_examples_)
                throw_data_pipeline_error(std::move(maybe_example), /*recoverable=*/true,
                    "The length of the input data must be less than or equal to the maximum bucket data length ({}), but is {} instead.", max_data_len_, data_len);

            // TODO(balioglu): log

            continue;
        }

        // Find the smallest bucket that would fit `example`, and return that
        // bucket if it is full.
        for (std::size_t i = 0; i < buckets_.size(); i++) {
            auto [bucket_num_examples, bucket_data_len] = bucket_sizes_[i];

            if (data_len <= bucket_data_len) {
                data_list &bucket = buckets_[i];

                bucket.push_back(std::move(example));

                if (bucket.size() >= bucket_num_examples) {
                    data output = data{std::exchange(bucket, {})};

                    bucket.reserve(bucket_num_examples);

                    return output;
                }

                break;
            }
        }
    }

    // If we are here, it means we exhausted the inner data source. For the
    // remaining examples in the buckets, return them by chunking them together
    // starting from the last bucket. Going in reverse bucket order is important
    // to ensure that we never exceed the maximum number of elements.
    for (std::size_t i = buckets_.size(); i > 0; i--) {
        data_list &bucket = buckets_[i - 1];
        if (bucket.empty())
            continue;

        auto [bucket_num_examples, bucket_data_len] = bucket_sizes_[i - 1];

        if (i - 1 > 0) {
            std::size_t remaining = bucket_num_examples - bucket.size();

            for (std::size_t j = i - 1; j > 0; j--) {
                data_list &other_bucket = buckets_[j - 1];

                while (remaining > 0 && !other_bucket.empty()) {
                    bucket.push_back(std::move(other_bucket.back()));

                    other_bucket.pop_back();

                    remaining--;
                }

                if (remaining == 0)
                    break;
            }
        }

        // This can only be true for the very last chunked bucket.
        if (drop_remainder_ && bucket.size() != bucket_num_examples) {
            bucket.clear();

            return std::nullopt;
        }

        return std::exchange(bucket, {});
    }

    return std::nullopt;
}

void
bucket_by_length_data_source::reset(bool reset_rng)
{
    for (data_list &bucket : buckets_)
        bucket.clear();

    inner_->reset(reset_rng);
}

void
bucket_by_length_data_source::record_position(tape &t, bool strict) const
{
    t.record(buckets_);

    inner_->record_position(t, strict);
}

void
bucket_by_length_data_source::reload_position(tape &t, bool strict)
{
    buckets_ = t.read<std::vector<data_list>>();

    inner_->reload_position(t, strict);
}

data_source_finitude_type
bucket_by_length_data_source::finitude_type() const noexcept
{
    return inner_->finitude_type();
}

}  // namespace fairseq2n::detail
