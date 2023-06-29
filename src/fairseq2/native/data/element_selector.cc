// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/element_selector.h"

#include <stdexcept>

#include <fmt/core.h>

namespace fairseq2::detail {

element_selector::element_selector(std::string_view s)
{
    // TODO: handle comman separated multi-path.

    std::optional<std::vector<path_segment>> pth = maybe_parse_path(s);
    if (!pth)
        throw std::invalid_argument{
            fmt::format("`selector` must be a well-formatted element path, but is '{}' instead.", s)};

    paths_.push_back(*std::move(pth));
}

void
element_selector::visit(data &d, const std::function<void(data &)> &f) const
{
    auto const_f = [&f](const data &e) {
        f(const_cast<data &>(e)); // NOLINT(cppcoreguidelines-pro-type-const-cast)
    };

    visit(static_cast<const data &>(d), const_f);
}

void
element_selector::visit(const data &d, const std::function<void(const data &)> &f) const
{
    for (const std::vector<path_segment> &p : paths_)
        visit(d, f, p);
}

std::optional<std::vector<element_selector::path_segment>>
element_selector::maybe_parse_path(std::string_view &s)
{
    if (s.empty())
        return std::nullopt;

    std::vector<path_segment> output{};

    auto state = path_parser_state::parsing_key;

    std::size_t idx = 0;

    std::size_t path_segment_offset = 0;

    for (std::size_t chr_idx = 0; chr_idx < s.size(); ++chr_idx) {
        char chr = s[chr_idx];

        if (state == path_parser_state::parsing_key) {
            if (chr == '.') {
                if (chr_idx == path_segment_offset)
                    // Empty path segment.
                    return std::nullopt;

                output.emplace_back(std::string{s.substr(path_segment_offset, chr_idx)});

                path_segment_offset = chr_idx + 1;
            } else if (chr == '[') {
                if (chr_idx == path_segment_offset) {
                    // We allow indexing at the root (e.g. "[0]").
                    if (chr_idx != 0)
                        return std::nullopt;
                } else
                    output.emplace_back(std::string{s.substr(path_segment_offset, chr_idx)});

                path_segment_offset = chr_idx + 1;

                state = path_parser_state::parsing_index;
            }
        } else if (state == path_parser_state::parsing_index) {
            if (chr == ']') {
                if (chr_idx == path_segment_offset)
                    // Empty index.
                    return std::nullopt;

                output.emplace_back(idx);

                idx = 0;

                state = path_parser_state::parsed_index;
            } else if (chr >= '0' && chr <= '9') {
                idx = (10 * idx) + static_cast<std::size_t>(chr - '0');
            } else
                return std::nullopt;
        } else if (state == path_parser_state::parsed_index) {
            if (chr == '[') {
                path_segment_offset = chr_idx + 1;

                state = path_parser_state::parsing_index;
            } else if (chr == '.') {
                path_segment_offset = chr_idx + 1;

                state = path_parser_state::parsing_key;
            } else
                // An index op can only be followed by '[' or '.'.
                return std::nullopt;
        }
    }

    if (state == path_parser_state::parsing_key) {
        if (path_segment_offset == s.size())
            // If the segment is empty, it means the selector ends with a '.'.
            return std::nullopt;

        output.emplace_back(std::string{s.substr(path_segment_offset)});
    } else if (state != path_parser_state::parsed_index)
        // Incomplete index op.
        return std::nullopt;

    return output;
}

void
element_selector::visit(
    const data &d,
    const std::function<void(const data &)> &f,
    const std::vector<path_segment> &p)
{
    const data *node = &d;

    for (const path_segment &s : p) {
        if (std::holds_alternative<std::string>(s)) {
            if (!node->is_dict())
                throw_invalid_path(p);

            auto &dct = node->as_dict();

            auto pos = dct.find(std::get<std::string>(s));

            if (pos == dct.end())
                throw_invalid_path(p);

            node = &pos->second;
        } else {
            if (!node->is_list())
                throw_invalid_path(p);

            auto &lst = node->as_list();

            auto idx = std::get<std::size_t>(s);

            if (idx >= lst.size())
                throw_invalid_path(p);

            node = &lst[idx];
        }
    }

    f(*node);
}

void
element_selector::throw_invalid_path(const std::vector<path_segment> &p)
{
    std::string r{};

    for (const path_segment &s : p) {
        if (std::holds_alternative<std::string>(s)) {
            if (!r.empty())
                r += ".";

            r += std::get<std::string>(s);
        } else
            r += "[" + std::to_string(std::get<std::size_t>(s)) + "]";
    }

    throw std::invalid_argument{
        fmt::format("The input data does not have an element at path '{}'.", r)};
}

}  // namespace fairseq2::detail
