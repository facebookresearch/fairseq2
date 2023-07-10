// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/element_selector.h"

#include <cctype>
#include <stdexcept>

#include <fmt/format.h>

#include "fairseq2/native/fmt.h"
#include "fairseq2/native/detail/exception.h"
#include "fairseq2/native/utils/string.h"

using namespace fairseq2::detail;

namespace fairseq2 {
namespace detail {

element_selector::element_selector(std::string_view selector)
{
    // The selector might contain one or more paths separated by comma.
    auto parse_next_path = [this, selector](
        std::string_view remaining_paths) -> std::optional<std::string_view>
    {
        std::size_t comma_idx = remaining_paths.find_first_of(',');

        std::optional<element_path> parsed_path = maybe_parse_path(
            /*path=*/remaining_paths.substr(0, comma_idx));

        if (!parsed_path)
            throw_<std::invalid_argument>(
                "`selector` must contain one or more well-formatted element paths, but is '{}' instead.", selector);

        paths_.push_back(*std::move(parsed_path));

        // We have reached the end of the selector string.
        if (comma_idx == std::string_view::npos)
            return std::nullopt;

        return remaining_paths.substr(comma_idx + 1);
    };

    std::optional<std::string_view> remaining_paths = selector;
    while (remaining_paths)
        remaining_paths = parse_next_path(*remaining_paths);
}

void
element_selector::visit(data &d, const std::function<void(data &, element_path_ref)> &visitor) const
{
    auto const_visitor = [&visitor](const data &element, element_path_ref path)
    {
        visitor(const_cast<data &>(element), path); // NOLINT(cppcoreguidelines-pro-type-const-cast)
    };

    visit(static_cast<const data &>(d), const_visitor);
}

void
element_selector::visit(
    const data &d, const std::function<void(const data &, element_path_ref)> &visitor) const
{
    for (const element_path &path : paths_)
        visit(d, visitor, path);
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

void
element_selector::visit(
    const data &d,
    const std::function<void(const data &, element_path_ref)> &visitor,
    const element_path &path)
{
    const data *node = &d;

    for (const element_path_segment &segment : path) {
        if (std::holds_alternative<std::string>(segment)) {
            if (!node->is_dict())
                throw_invalid_path(path);

            auto &dict = node->as_dict();

            auto dict_entry = dict.find(std::get<std::string>(segment));
            if (dict_entry == dict.end())
                throw_invalid_path(path);

            node = &dict_entry->second;
        } else {
            if (!node->is_list())
                throw_invalid_path(path);

            auto &list = node->as_list();

            auto idx = std::get<std::size_t>(segment);
            if (idx >= list.size())
                throw_invalid_path(path);

            node = &list[idx];
        }
    }

    visitor(*node, path);
}

void
element_selector::throw_invalid_path(element_path_ref path)
{
    throw_<std::invalid_argument>("The input data does not have an element at path '{}'.", path);
}

}  // namespace detail

std::string
repr(element_path_ref path)
{
    std::string output{};

    for (const element_path_segment &segment : path) {
        if (std::holds_alternative<std::string>(segment)) {
            if (!output.empty())
                output += ".";

            output += std::get<std::string>(segment);
        } else
            output += "[" + fmt::to_string(std::get<std::size_t>(segment)) + "]";
    }

    return output;
}

}  // namespace fairseq2
