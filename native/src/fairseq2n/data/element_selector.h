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

#include "fairseq2n/api.h"
#include "fairseq2n/fmt.h"
#include "fairseq2n/span.h"
#include "fairseq2n/data/data.h"

namespace fairseq2n {

using element_path_segment = std::variant<std::string, std::size_t>;

using element_path = std::vector<element_path_segment>;
using element_path_ref = span<const element_path_segment>;

class FAIRSEQ2_API element_selector {
    enum class path_parser_state { parsing_key, parsing_index, parsed_index };

public:
    using visitor_fn = std::function<void(data &, element_path_ref)>;
    using const_visitor_fn = std::function<void(const data &, element_path_ref)>;

public:
    explicit
    element_selector(std::string selector);

private:
    static std::optional<element_path>
    maybe_parse_path(std::string_view path);

public:
    bool
    matches(element_path_ref path) const;

    void
    visit(data &d, const visitor_fn &visitor) const;

    void
    visit(const data &d, const const_visitor_fn &visitor) const;

    static bool
    visit(data &d, element_path_ref path, const visitor_fn &visitor);

    static bool
    visit(const data &d, element_path_ref path, const const_visitor_fn &visitor);

    const std::string &
    string_() const noexcept
    {
        return str_;
    }

private:
    template <typename T>
    static bool
    visit(T &d, element_path_ref path, const std::function<void(T &, element_path_ref)> &visitor);

private:
    std::string str_;
    std::vector<element_path> paths_{};
};

template <>
struct FAIRSEQ2_API repr<element_path_ref> {
    std::string
    operator()(element_path_ref path) const;
};

template <>
struct repr<element_path> {
    std::string
    operator()(const element_path &path) const
    {
        return repr<element_path_ref>{}(path);
    }
};

}  // namespace fairseq2n
