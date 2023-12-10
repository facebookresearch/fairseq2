// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/collater.h"

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include "fairseq2n/float.h"
#include "fairseq2n/fmt.h"
#include "fairseq2n/span.h"
#include "fairseq2n/data/data.h"
#include "fairseq2n/detail/exception.h"

using namespace fairseq2n::detail;

namespace fairseq2n {
namespace detail {

class collate_op {
    using bucket_visitor_fn = std::function<void(data &, std::size_t)>;

public:
    explicit
    collate_op(const collater *c, data_list &&bucket) noexcept
      : collater_{c}, bucket_(std::move(bucket))
    {}

    data
    run();

private:
    data
    collate_current_path();

    data
    collate(data_list &list);

    data
    collate(data_dict &dict);

    data
    collate(at::Tensor &tensor);

    data
    collate(data &d);

    void
    visit_bucket_items(data_type expected_type, const bucket_visitor_fn &visitor);

    data &
    get_current_element(std::size_t bucket_item_idx);

    const collate_options &
    get_options_for_current_path() const;

    static data
    pad_tensors(span<at::Tensor> tensors, std::int64_t pad_value, const collate_options &opts);

private:
    const collater *collater_;
    data_list bucket_;
    element_path path_{};
};

}  // namespace detail

data
collate_op::run()
{
    // Start from the root.
    return collate_current_path();
}

data
collate_op::collate_current_path()
{
    // Retrieve the element of the first bucket item and use it as the base line
    // for the remaining items.
    data &element = get_current_element(/*bucket_item_idx=*/0);

    if (element.is_list())
        return collate(element.as_list());

    if (element.is_dict())
        return collate(element.as_dict());

    if (element.is_tensor())
        return collate(element.as_tensor());

    // For all other non-composite data types, collate the elements into a list.
    return collate(element);
}

data
collate_op::collate(data_list &list)
{
    auto expected_dtype = data_type::list;

    // Make sure that each bucket item has the same list size.
    visit_bucket_items(expected_dtype, [this, &list](data &element, std::size_t item_idx)
    {
        std::size_t size = element.as_list().size();
        if (path_.empty()) {
            if (size != list.size())
                throw_<std::invalid_argument>(
                    "The bucket item {} must have a length of {}, but has a length of {} instead.", item_idx, list.size(), size);
        } else {
            if (size != list.size())
                throw_<std::invalid_argument>(
                    "The `list` at path '{}' in the bucket item {} must have a length of {}, but has a length of {} instead.", path_, item_idx, list.size(), size);
        }
    });

    data_list output{};

    output.reserve(bucket_.size());

    // Collate each element of the list.
    for (std::size_t idx = 0; idx < list.size(); ++idx) {
        // Move the path down to the next index in the list.
        path_.emplace_back(idx);

        output.push_back(collate_current_path());

        // Move the path up.
        path_.pop_back();
    }

    return output;
}

data
collate_op::collate(data_dict &dict)
{
    auto expected_dtype = data_type::dict;

    // Make sure that each bucket item has the same dict size.
    visit_bucket_items(expected_dtype, [this, &dict](data &element, std::size_t item_idx)
    {
        std::size_t size = element.as_dict().size();
        if (path_.empty()) {
            if (size != dict.size())
                throw_<std::invalid_argument>(
                    "The bucket item {} must have a length of {}, but has a length of {} instead.", item_idx, dict.size(), size);
        } else {
            if (size != dict.size())
                throw_<std::invalid_argument>(
                    "The `dict` at path '{}' in the bucket item {} must have a length of {}, but has a length of {} instead.", path_, item_idx, dict.size(), size);
        }
    });

    data_dict output{};

    // Collate each entry of the dictionary.
    for (auto &[key, value] : dict) {
        // Move the path down to the next key in the dict.
        path_.emplace_back(key);

        output.emplace(key, collate_current_path());

        // Move the path up.
        path_.pop_back();
    }

    return output;
}

data
collate_op::collate(at::Tensor &tensor)
{
    // All bucket items must have a tensor at the current path.
    auto expected_dtype = data_type::tensor;

    std::vector<at::Tensor> tensors{};

    tensors.reserve(bucket_.size());

    // We have already retrieved the tensor of the first bucket item in the
    // `collate_current_path()` call.
    tensors.push_back(std::move(tensor));

    // Collect tensors of the remaining bucket items.
    visit_bucket_items(expected_dtype, [&tensors](data &element, std::size_t)
    {
        tensors.push_back(std::move(element).as_tensor());
    });

    const collate_options &opts = get_options_for_current_path();
    if (std::optional<std::int64_t> maybe_pad_value = opts.maybe_pad_value(); maybe_pad_value)
        return pad_tensors(tensors, *maybe_pad_value, opts);

    try {
        return at::stack(tensors);
    } catch (const c10::Error &) {
        if (path_.empty()) {
            throw_with_nested<std::invalid_argument>(
                "The tensors in the bucket cannot be stacked. See nested exception for details.", path_);
        } else {
            throw_with_nested<std::invalid_argument>(
                "The tensors at path '{}' in the bucket cannot be stacked. See nested exception for details.", path_);
        }
    }
}

data
collate_op::collate(data &d)
{
    // All bucket items must have the same type of element at the current path.
    auto expected_dtype = d.type();

    data_list output{};

    output.reserve(bucket_.size());

    // We have already retrieved the element of the first bucket item in the
    // `collate_current_path()` call.
    output.push_back(std::move(d));

    // Collect elements of the remaining bucket items.
    visit_bucket_items(expected_dtype, [&output](data &element, std::size_t)
    {
        output.push_back(std::move(element));
    });

    return output;
}

void
collate_op::visit_bucket_items(data_type expected_type, const bucket_visitor_fn &visitor)
{
    // Start from the second bucket item since we have already retrieved the
    // element of the first item in the `collate_current_path()` call.
    for (std::size_t item_idx = 1; item_idx < bucket_.size(); ++item_idx) {
        data &element = get_current_element(item_idx);

        if (path_.empty()) {
            if (element.type() != expected_type)
                throw_<std::invalid_argument>(
                    "The bucket item {} must be of type `{}`, but is of type `{}` instead.", item_idx, expected_type, element.type());
        } else {
            if (element.type() != expected_type)
                throw_<std::invalid_argument>(
                    "The element at path '{}' in the bucket item {} must be of type `{}`, but is of type `{}` instead.", path_, item_idx, expected_type, element.type());
        }

        visitor(element, item_idx);
    }
}

data &
collate_op::get_current_element(std::size_t bucket_item_idx)
{
    data &bucket_item = bucket_[bucket_item_idx];

    data *element = nullptr;

    // Leverage `element_selector` to retrieve the element.
    element_selector::visit(bucket_item, path_, [&element](data &d, element_path_ref)
    {
        element = &d;
    });

    if (element == nullptr)
        throw_<std::invalid_argument>(
            "The bucket item {} does not have an element at path '{}'.", bucket_item_idx, path_);

    return *element;
}

const collate_options &
collate_op::get_options_for_current_path() const
{
    for (const collate_options_override &ov : collater_->opt_overrides_)
        if (ov.selector().matches(path_))
            return ov.options();

    // Fall back to default options.
    return collater_->opts_;
}

data
collate_op::pad_tensors(span<at::Tensor> tensors, std::int64_t pad_value, const collate_options &opts)
{
    at::Tensor seqs{};

    // Pad.
    at::Tensor tmp = at::pad_sequence(
        tensors, /*batch_first=*/true, static_cast<float64>(pad_value));

    at::IntArrayRef shape = tmp.sizes();

    // Pad to multiple.
    if (opts.pad_to_multiple() > 1 && shape[1] % opts.pad_to_multiple() > 0) {
        std::vector<std::int64_t> pad_shape(shape.begin(), shape.end());

        pad_shape[1] = opts.pad_to_multiple() - (shape[1] % opts.pad_to_multiple());

        at::Tensor pad = tmp.new_full(pad_shape, pad_value);

        // PyTorch has trouble with LSan when a tensor is used both as an input
        // and as an output to `concat`. `tmp` is a workaround for that.
        seqs = at::concat({tmp, pad}, /*dim=*/1);
    } else
        seqs = tmp;

    // Construct sequence length tensor.
    at::Tensor seq_lens = at::empty({shape[0]}, at::dtype(at::kLong));

    bool is_ragged = false;

    auto seq_lens_data = seq_lens.accessor<std::int64_t, 1>();

    std::int64_t i = 0;
    for (const at::Tensor &t : tensors) {
        std::int64_t seq_len = t.size(0);

        seq_lens_data[i] = seq_len;

        if (i > 0 && seq_lens_data[i - 1] != seq_len)
            is_ragged = true;

        ++i;
    }

    // We might still need to return as ragged even if all sequences have the
    // same length if `seqs` has extra padding due to `pad_to_multiple`.
    if (!is_ragged && !tensors.empty() && seq_lens_data[0] != seqs.size(1))
        is_ragged = true;

    seq_lens = seq_lens.to(seqs.device());

    // Pack the sequences and their lengths into a dict.
    data_dict output{{"is_ragged", is_ragged}};

    output.emplace("seqs", std::move(seqs));
    output.emplace("seq_lens", std::move(seq_lens));

    return output;
}

collater::collater(collate_options opts, std::vector<collate_options_override> opt_overrides)
  : opts_{opts}, opt_overrides_{std::move(opt_overrides)}
{
    if (opts_.pad_to_multiple() > 1 && !opts_.maybe_pad_value())
        throw_<std::invalid_argument>(
            "`pad_value` must be set when `pad_to_multiple` is greater than 1.");

    for (collate_options_override &ov : opt_overrides_)
        if (ov.options().pad_to_multiple() > 1 && !ov.options().maybe_pad_value())
            throw_<std::invalid_argument>(
                "`pad_value` of the selector '{}' must be set when `pad_to_multiple` is greater than 1.", ov.selector().string_());
}

data
collater::operator()(data &&d) const
{
    if (!d.is_list())
        d = data_list{std::move(d)};

    data_list &bucket = d.as_list();

    if (bucket.empty())
        throw_<std::invalid_argument>(
            "The bucket must contain at least one element, but is empty instead.");

    return collate_op{this, std::move(bucket)}.run();
}

}  // namespace fairseq2n
