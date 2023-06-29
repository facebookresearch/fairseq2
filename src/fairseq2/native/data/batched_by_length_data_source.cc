// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/batched_by_length_data_source.h"

#include "fairseq2/native/data/data_pipeline.h"

namespace fairseq2::detail {

batched_by_length_data_source::batched_by_length_data_source(
    std::unique_ptr<data_source> &&inner,
    std::vector<std::pair<std::size_t, std::size_t>> &&bucket_sizes,
    std::size_t max_seq_len,
    std::optional<element_selector> &&selector,
    bool drop_remainder,
    bool warn_only)
  : inner_{std::move(inner)},
    bucket_sizes_(std::move(bucket_sizes)),
    max_seq_len_{max_seq_len},
    selector_{std::move(selector)},
    drop_remainder_{drop_remainder},
    warn_only_{warn_only}
{
    buckets_.reserve(bucket_sizes_.size());

    for (auto [bucket_batch_size, bucket_seq_len] : bucket_sizes_)
        buckets_.emplace_back().reserve(bucket_batch_size);
}

std::optional<data>
batched_by_length_data_source::next()
{
    while (std::optional<data> d = inner_->next()) {
        if (!d)
            break;

        std::optional<std::size_t> seq_len = determine_seq_len(*d);
        if (!seq_len)
            continue;

        for (std::size_t i = 0; i < buckets_.size(); i++) {
            auto [bucket_batch_size, bucket_seq_len] = bucket_sizes_[i];

            if (*seq_len <= bucket_seq_len) {
                std::vector<data> &b = buckets_[i];

                b.push_back(*std::move(d));

                if (b.size() >= bucket_batch_size) {
                    data output = data{std::exchange(b, {})};

                    b.reserve(bucket_batch_size);

                    return output;
                }

                break;
            }
        }
    }

    if (!drop_remainder_) {
        for (std::vector<data> &b : buckets_) {
            if (b.empty())
                continue;

            return std::exchange(b, {});
        }
    }

    return std::nullopt;
}

void
batched_by_length_data_source::reset()
{
    for (std::vector<data> &b : buckets_)
        b.clear();

    inner_->reset();
}

void
batched_by_length_data_source::record_position(tape &t) const
{
    t.record(buckets_);

    inner_->record_position(t);
}

void
batched_by_length_data_source::reload_position(tape &t)
{
    buckets_ = t.read<std::vector<std::vector<data>>>();

    inner_->reload_position(t);
}

std::optional<std::size_t>
batched_by_length_data_source::determine_seq_len(const data &d)
{
    auto get_seq_len = [](const data &e) {
        if (!e.is_tensor())
            throw std::invalid_argument{"The input data must be of type `torch.Tensor`."};

        return static_cast<std::size_t>(e.as_tensor().size(0));
    };

    std::size_t seq_len = 0;

    if (selector_) {
        selector_->visit(d, [&seq_len, &get_seq_len](const data &e) {
            seq_len = std::max(seq_len, get_seq_len(e));
        });
    } else
        seq_len = get_seq_len(d);

    if (seq_len <= max_seq_len_)
        return seq_len;

    if (!warn_only_)
        throw data_pipeline_error{"The input data is too long!"};

    // TODO: warn!

    return std::nullopt;
}

}  // namespace fairseq2::detail
