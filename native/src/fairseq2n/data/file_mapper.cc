// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/file_mapper.h"

#include <array>
#include <filesystem>
#include <stdexcept>

#include <fmt/format.h>

#include "fairseq2n/fmt.h"
#include "fairseq2n/utils/string.h"
#include "fairseq2n/data/immutable_string.h"
#include "fairseq2n/data/file.h"
#include "fairseq2n/detail/exception.h"

using namespace fairseq2n::detail;

namespace fairseq2n {

file_mapper::file_mapper(
    std::optional<std::string> maybe_root_dir,
    std::optional<std::size_t> maybe_cached_fd_count) noexcept
  : cache_{/*capacity=*/maybe_cached_fd_count.value_or(default_cached_fd_count)}
{
    if (maybe_root_dir)
        root_dir_ = *std::move(maybe_root_dir);
}

data
file_mapper::operator()(data &&d) const
{
    if (!d.is_string())
        throw_<std::invalid_argument>(
            "The input data must be of type `string`, but is of type `{}` instead.", d.type());

    immutable_string &pathname = d.as_string();

    std::array<immutable_string, 3> parts{};

    auto pos = parts.begin();

    // The pathname can optionally contain offset and size specifiers separated
    // by semicolons (i.e. `pathname:offset:size`).
    pathname.split(':', [&pathname, &parts, &pos](immutable_string &&part)
    {
        if (pos == parts.end())
            throw_<std::invalid_argument>(
                "The input string must be a pathname with optional offset and size specifiers, but is '{}' instead.", pathname);

        part = trim(part);

        if (part.empty())
            throw_<std::invalid_argument>(
                "The input string must be a pathname with optional offset and size specifiers, but is '{}' instead.", pathname);

        *pos++ = std::move(part);

        return true;
    });

    auto parse_specifier = [&pathname](std::string_view part, std::string_view specifier_name)
    {
        try {
            return from_string<std::size_t>(part);
        } catch (const std::out_of_range &) {
            throw_<std::invalid_argument>(
                "The {} specifier of '{}' must be a machine-representable integer, but is '{}' instead, which is out of range.", specifier_name, pathname, part);
        } catch (const std::invalid_argument &) {
            throw_<std::invalid_argument>(
                "The {} specifier of '{}' must be an integer, but is '{}' instead.", specifier_name, pathname, part);
        }
    };

    std::optional<std::size_t> maybe_offset{}, maybe_size{};

    if (!parts[1].empty()) {
        maybe_offset = parse_specifier(parts[1], "offset");

        if (!parts[2].empty())
            maybe_size = parse_specifier(parts[2], "size");
    }

    memory_block block = get_memory_map(parts[0]);

    auto pack_output = [&d](memory_block &&blk)
    {
        data_dict output{};

        output.emplace("path", std::move(d));
        output.emplace("data", std::move(blk));

        return output;
    };

    // If we don't have an offset, return the entire memory map.
    if (!maybe_offset)
        return pack_output(std::move(block));

    std::size_t offset = *maybe_offset;

    if (offset > block.size())
        throw_<std::invalid_argument>(
            "The specified offset within '{}' must be less than or equal to the file size ({} bytes), but is {} instead.", pathname, fmt::group_digits(block.size()), fmt::group_digits(offset));

    // If we have an offset but not a size, return the memory map from the
    // offset to the end.
    if (!maybe_size)
        return pack_output(block.share_slice(offset));

    std::size_t size = *maybe_size;

    if (std::size_t upper_boundary = offset + size; upper_boundary > block.size())
        throw_<std::invalid_argument>(
            "The end of the specified region within '{}' must be less than or equal to the file size ({} bytes), but is {} instead.", pathname, fmt::group_digits(block.size()), fmt::group_digits(upper_boundary));

    // Otherwise, return the memory map region specified by the offset and size.
    return pack_output(block.share_slice(offset, size));
}

memory_block
file_mapper::get_memory_map(const immutable_string &pathname) const
{
    std::lock_guard<std::mutex> cache_guard{cache_mutex_};

    // Check the LRU cache first.
    memory_block *maybe_cached_block = cache_.maybe_get(pathname);
    if (maybe_cached_block != nullptr)
        return *maybe_cached_block;

    memory_block block = memory_map_file(root_dir_ / pathname.to_string());

    cache_.add(pathname, block);

    return block;
}

}  // namespace fairseq2n
