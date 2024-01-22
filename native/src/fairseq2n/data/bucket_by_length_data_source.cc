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
    bool skip_long_examples,
    bool drop_remainder)
  : inner_{std::move(inner)},
    bucket_sizes_(std::move(bucket_sizes)),
    max_data_len_{bucket_sizes_.back().second},
    data_length_fn_{std::move(fn)},
    skip_long_examples_{skip_long_examples},
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

        std::size_t data_len{};
        try {
            data_len = data_length_fn_(example);
        } catch (const std::invalid_argument &) {
            throw_data_pipeline_error_with_nested(std::move(maybe_example), /*recoverable=*/true,
                "The length of the input data cannot be determined.");
        }

        if (data_len > max_data_len_) {
            if (!skip_long_examples_)
                throw_data_pipeline_error(std::move(maybe_example), /*recoverable=*/true,
                    "The length of the input data must be less than or equal to the maximum bucket data length ({}), but is {} instead.", max_data_len_, data_len);

            // TODO(balioglu): log info

            continue;
        }

        // Find the smallest bucket that would fit `example`, and return that bucket
        // if it is full.
        for (std::size_t i = 0; i < buckets_.size(); i++) {
            auto [bucket_batch_size, bucket_data_len] = bucket_sizes_[i];

            if (data_len <= bucket_data_len) {
                data_list &bucket = buckets_[i];

                bucket.push_back(std::move(example));

                if (bucket.size() >= bucket_batch_size) {
                    data output = data{std::exchange(bucket, {})};

                    bucket.reserve(bucket_batch_size);

                    return output;
                }

                break;
            }
        }
    }

    if (!drop_remainder_) {
        // Return the smallest partially filled bucket.
        for (data_list &bucket : buckets_) {
            if (bucket.empty())
                continue;

            return std::exchange(bucket, {});
        }
    }

    return std::nullopt;
}

void
bucket_by_length_data_source::reset()
{
    for (data_list &bucket : buckets_)
        bucket.clear();

    inner_->reset();
}

void
bucket_by_length_data_source::record_position(tape &t) const
{
    t.record(buckets_);

    inner_->record_position(t);
}

void
bucket_by_length_data_source::reload_position(tape &t)
{
    buckets_ = t.read<std::vector<data_list>>();

    inner_->reload_position(t);
}

bool
bucket_by_length_data_source::is_infinite() const noexcept
{
    return inner_->is_infinite();
}

}  // namespace fairseq2n::detail
