// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/element_selector.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>

#include <fmt/format.h>

#include "fairseq2n/fmt.h"
#include "fairseq2n/detail/exception.h"
#include "fairseq2n/utils/string.h"

using namespace fairseq2n::detail;

namespace fairseq2n {

constexpr std::size_t wildcard_index = std::numeric_limits<std::size_t>::max();

element_selector::element_selector(std::string selector)
  : str_{std::move(selector)}
{
    // The selector might contain one or more paths separated by comma.
    auto parse_next_path = [this](
        std::string_view remaining_paths) -> std::optional<std::string_view>
    {
        std::size_t comma_idx = remaining_paths.find_first_of(',');

        std::optional<element_path> maybe_parsed_path = maybe_parse_path(
            /*path=*/remaining_paths.substr(0, comma_idx));

        if (!maybe_parsed_path)
            throw_<std::invalid_argument>(
                "`selector` must contain one or more well-formatted element paths, but is '{}' instead.", str_);

        paths_.push_back(*std::move(maybe_parsed_path));

        // We have reached the end of the selector string.
        if (comma_idx == std::string_view::npos)
            return std::nullopt;

        return remaining_paths.substr(comma_idx + 1);
    };

    std::optional<std::string_view> maybe_remaining_paths = str_;
    while (maybe_remaining_paths)
        maybe_remaining_paths = parse_next_path(*maybe_remaining_paths);
}

std::optional<element_path>
element_selector::maybe_parse_path(std::string_view path)
{
    path = trim(path);

    if (path.empty())
        return std::nullopt;

    element_path output{};

    auto record_key_segment = [&output, &path](
        std::size_t start_idx, std::size_t end_idx = std::string_view::npos)
    {
        output.emplace_back(std::string{path.substr(start_idx, end_idx - start_idx)});
    };

    auto record_index_segment = [&output](std::size_t idx)
    {
        output.emplace_back(idx);
    };

    auto state = path_parser_state::parsing_key;

    std::size_t idx = 0;

    std::size_t segment_offset = 0;

    for (std::size_t char_idx = 0; char_idx < path.size(); ++char_idx) {
        char chr = path[char_idx];

        if (state == path_parser_state::parsing_key) {
            if (chr == '.') {
                if (char_idx == segment_offset)
                    // Empty path segment.
                    return std::nullopt;

                record_key_segment(segment_offset, char_idx);

                segment_offset = char_idx + 1;
            } else if (chr == '[') {
                if (char_idx == segment_offset) {
                    // We allow indexing at the root (e.g. "[0]").
                    if (char_idx != 0)
                        return std::nullopt;
                } else
                    record_key_segment(segment_offset, char_idx);

                segment_offset = char_idx + 1;

                state = path_parser_state::parsing_index;
            } else if (std::isspace(chr) != 0)
                return std::nullopt;
        } else if (state == path_parser_state::parsing_index) {
            if (chr == ']') {
                if (char_idx == segment_offset)
                    // Empty index.
                    return std::nullopt;

                record_index_segment(idx);

                idx = 0;

                state = path_parser_state::parsed_index;
            } else if (chr >= '0' && chr <= '9') {
                auto tmp = (10 * idx) + static_cast<std::size_t>(chr - '0');

                // Check overflow.
                if (idx > tmp)
                    return std::nullopt;

                idx = tmp;
            } else if (chr == '*') {
                idx = wildcard_index;
            } else
                return std::nullopt;
        } else if (state == path_parser_state::parsed_index) {
            if (chr == '[') {
                segment_offset = char_idx + 1;

                state = path_parser_state::parsing_index;
            } else if (chr == '.') {
                segment_offset = char_idx + 1;

                state = path_parser_state::parsing_key;
            } else
                // An index op can only be followed by '[' or '.'.
                return std::nullopt;
        }
    }

    if (state == path_parser_state::parsing_key) {
        if (segment_offset == path.size())
            // If the segment is empty, it means the path ends with a '.'.
            return std::nullopt;

        record_key_segment(segment_offset);
    } else if (state != path_parser_state::parsed_index)
        // Incomplete index op.
        return std::nullopt;

    return output;
}

bool
element_selector::matches(element_path_ref path) const
{
    auto matches_path = [&path](const element_path &p)
    {
        if (p.size() != path.size())
            return false;

        for (std::size_t i = 0; i < p.size(); i++) {
            const element_path_segment &segment1 = p[i];
            const element_path_segment &segment2 = path[i];

            if (segment1 != segment2) {
                bool holds_index1 = std::holds_alternative<std::size_t>(segment1);
                bool holds_index2 = std::holds_alternative<std::size_t>(segment2);

                if (holds_index1 && holds_index2) {
                    if (std::get<std::size_t>(segment1) == wildcard_index)
                        continue;
                }

                return false;
            }
        }

        return true;
    };

    return std::any_of(paths_.begin(), paths_.end(), matches_path);
}

void
element_selector::visit(data &d, const visitor_fn &visitor) const
{
    for (const element_path &path : paths_)
        visit(d, path, visitor);
}

void
element_selector::visit(const data &d, const const_visitor_fn &visitor) const {
    for (const element_path &path : paths_)
        visit(d, path, visitor);
}

void
element_selector::visit(data &d, element_path_ref path, const visitor_fn &visitor)
{
    element_path resolved_path{};

    resolved_path.reserve(path.size());

    if (!visit<data>(d, path, path, resolved_path, visitor)) {
        for (std::size_t i = resolved_path.size(); i < path.size(); i++)
            resolved_path.push_back(path[i]);

        throw_<std::invalid_argument>(
            "The input data does not have an element at path '{}'.", resolved_path);
    }
}

void
element_selector::visit(const data &d, element_path_ref path, const const_visitor_fn &visitor)
{
    element_path resolved_path{};

    resolved_path.reserve(path.size());

    if (!visit<const data>(d, path, path, resolved_path, visitor)) {
        for (std::size_t i = resolved_path.size(); i < path.size(); i++)
            resolved_path.push_back(path[i]);

        throw_<std::invalid_argument>(
            "The input data does not have an element at path '{}'.", resolved_path);
    }
}

template <typename T>
bool
element_selector::visit(
    T &d,
    element_path_ref abs_path,
    element_path_ref path,
    element_path &resolved_path,
    const std::function<void(T &, element_path_ref)> &visitor)
{
    T *element = &d;

    for (std::size_t i = 0; i < path.size(); i++) {
        const element_path_segment &segment = path[i];

        resolved_path.push_back(segment);

        if (std::holds_alternative<std::string>(segment)) {
            if (!element->is_dict())
                return false;

            auto &dict = element->as_dict();

            auto dict_entry = dict.find(std::get<std::string>(segment));
            if (dict_entry == dict.end())
                return false;

            element = &dict_entry->second;
        } else {
            if (!element->is_list())
                return false;

            auto &list = element->as_list();

            auto idx = std::get<std::size_t>(segment);
            if (idx == wildcard_index) {
                for (std::size_t j = 0; j < list.size(); j++) {
                    resolved_path.back() = j;

                    if (!visit(list[j], abs_path, path.subspan(i + 1), resolved_path, visitor))
                        return false;
                }

                for (std::size_t j = 0; j < i; j++)
                    resolved_path.pop_back();

                return true;
            } else {
                if (idx >= list.size())
                    return false;

                element = &list[idx];
            }
        }
    }

    visitor(*element, resolved_path);

    for (std::size_t i = 0; i < path.size(); i++)
        resolved_path.pop_back();

    return true;
}

std::string
repr<element_path_ref>::operator()(element_path_ref path) const
{
    std::string output{};

    for (const element_path_segment &segment : path) {
        if (std::holds_alternative<std::string>(segment)) {
            if (!output.empty())
                output += ".";

            output += std::get<std::string>(segment);
        } else {
            std::size_t idx = std::get<std::size_t>(segment);
            if (idx == wildcard_index)
                output += "[*]";
            else
                output += "[" + fmt::to_string(std::get<std::size_t>(segment)) + "]";
        }
    }

    return output;
}

}  // namespace fairseq2n
