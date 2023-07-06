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
#include "fairseq2/native/span.h"
#include "fairseq2/native/data/data.h"

namespace fairseq2 {
namespace detail {

using element_path_segment = std::variant<std::string, std::size_t>;

using element_path = std::vector<element_path_segment>;

using element_path_ref = span<const element_path_segment>;

class FAIRSEQ2_API element_selector {
    enum class path_parser_state { parsing_key, parsing_index, parsed_index };

public:
    explicit
    element_selector(std::string_view selector);

    void
    visit(data &d, const std::function<void(data &, element_path_ref)> &visitor) const;

    void
    visit(const data &d, const std::function<void(const data &, element_path_ref)> &visitor) const;

private:
    static std::optional<element_path>
    maybe_parse_path(std::string_view path);

    static void
    visit(
        const data &d,
        const std::function<void(const data &, element_path_ref)> &visitor,
        const element_path &path);

    [[noreturn]] static void
    throw_invalid_path(element_path_ref path);

private:
    std::vector<element_path> paths_{};
};

}  // namespace detail

FAIRSEQ2_API std::string
repr(detail::element_path_ref path);

}  // namespace fairseq2
