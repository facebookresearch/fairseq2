// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data.h"

namespace fairseq2::detail {

class FAIRSEQ2_API element_selector {
    using path_segment = std::variant<std::string, std::size_t>;

    enum class parser_state { parsing_key, parsing_index, parsed_index };

public:
    explicit
    element_selector(std::string_view s);

    static std::optional<element_selector>
    maybe_parse(std::optional<std::string_view> s)
    {
        if (!s)
            return std::nullopt;

        return element_selector{*s};
    }

    void
    visit(data &d, const std::function<void(data &)> &f) const;

    void
    visit(const data &d, const std::function<void(const data &)> &f) const;

private:
    static std::vector<path_segment>
    parse_path(std::string_view &s);

    static void
    visit(
        const data &d,
        const std::function<void(const data &)> &f,
        const std::vector<path_segment> &p);

private:
    std::vector<std::vector<path_segment>> paths_{};
};

}  // namespace fairseq2::detail
