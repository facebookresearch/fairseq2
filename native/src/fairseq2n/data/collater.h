// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "fairseq2n/api.h"
#include "fairseq2n/data/element_selector.h"

namespace fairseq2n {

class collate_options {
public:
    collate_options
    maybe_pad_value(std::optional<std::int64_t> value) noexcept
    {
        auto tmp = *this;

        tmp.maybe_pad_value_ = value;

        return tmp;
    }

    std::optional<std::int64_t>
    maybe_pad_value() const noexcept
    {
        return maybe_pad_value_;
    }

    collate_options
    pad_to_multiple(std::int64_t value) noexcept
    {
        auto tmp = *this;

        tmp.pad_to_multiple_ = value;

        return tmp;
    }

    std::int64_t
    pad_to_multiple() const noexcept
    {
        return pad_to_multiple_;
    }

private:
    std::optional<std::int64_t> maybe_pad_value_;
    std::int64_t pad_to_multiple_ = 1;
};

class collate_options_override {
public:
    explicit
    collate_options_override(std::string selector, collate_options opts)
      : selector_{std::move(selector)}, opts_{opts}
    {}

    const element_selector &
    selector() const noexcept
    {
        return selector_;
    }

    const collate_options &
    options() const noexcept
    {
        return opts_;
    }

private:
    element_selector selector_;
    collate_options opts_;
};

class data;

namespace detail {

class collate_op;

}  // namespace detail

class FAIRSEQ2_API collater {
    friend class detail::collate_op;

public:
    explicit
    collater(collate_options opts = {}, std::vector<collate_options_override> opt_overrides = {});

    data
    operator()(data &&d) const;

private:
    collate_options opts_;
    std::vector<collate_options_override> opt_overrides_;
};

}  // namespace fairseq2n
