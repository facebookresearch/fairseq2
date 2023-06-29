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
//    for (xx):
    paths_.push_back(parse_path(s));
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

std::vector<element_selector::path_segment>
element_selector::parse_path(std::string_view &s)
{
    std::vector<path_segment> output{};

    if (s.empty())
        return output;

    auto state = parser_state::parsing_key;

    std::size_t idx = 0;

    std::size_t segment_offset = 0;

    for (std::size_t chr_idx = 0; chr_idx < s.size(); ++chr_idx) {
        char chr = s[chr_idx];

        if (state == parser_state::parsing_key) {
            if (chr == '.') {
                if (chr_idx == segment_offset)
                    throw std::invalid_argument{"empty"};

                output.emplace_back(std::string{s.substr(segment_offset, chr_idx)});

                segment_offset = chr_idx + 1;
            } else if (chr == '[') {
                if (chr_idx == segment_offset && chr_idx != 0)
                    throw std::invalid_argument{"empty"};

                output.emplace_back(std::string{s.substr(segment_offset, chr_idx)});

                segment_offset = chr_idx + 1;

                state = parser_state::parsing_index;
            }
        } else if (state == parser_state::parsing_index) {
            if (chr == ']') {
                if (chr_idx == segment_offset)
                    throw std::invalid_argument{"empty"};

                output.emplace_back(idx);

                idx = 0;

                state = parser_state::parsed_index;
            } else if (chr >= '0' && chr <= '9') {
                idx = (10 * idx) + static_cast<std::size_t>(chr - '0');
            } else
                throw std::invalid_argument{
                    fmt::format("invalid char {} in index", chr)
                };
        } else if (state == parser_state::parsed_index) {
            if (chr == '[') {
                segment_offset = chr_idx + 1;

                state = parser_state::parsing_index;
            } else if (chr == '.') {
                segment_offset = chr_idx + 1;

                state = parser_state::parsing_key;
            } else
                throw std::invalid_argument{"invalid char after parsed index"};
        }
    }

    if (state == parser_state::parsing_key) {
        if (segment_offset == s.size())
            throw std::invalid_argument{"empty"};

        output.emplace_back(std::string{s.substr(segment_offset)});
    } else if (state != parser_state::parsed_index)
        throw std::invalid_argument{"incomplete"};

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
        if (const auto *key = std::get_if<std::string>(&s)) {
            if (!node->is_dict())
                throw std::invalid_argument{"The node is expected to be a dict."};

            auto &dct = node->as_dict();

            auto pos = dct.find(*key);

            if (pos == dct.end())
                throw std::invalid_argument{"key not found"};

            node = &pos->second;
        } else {
            if (!node->is_list())
                throw std::invalid_argument{"cece"};

            auto &lst = node->as_list();

            auto idx = std::get<std::size_t>(s);

            if (idx >= lst.size())
                throw std::invalid_argument{"cece"};

            node = &lst[idx];
        }
    }

    f(*node);
}

}  // namespace fairseq2::detail
