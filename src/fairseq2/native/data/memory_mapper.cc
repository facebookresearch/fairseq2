// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/memory_mapper.h"

#include <array>
#include <filesystem>
#include <stdexcept>

#include "fairseq2/native/fmt.h"
#include "fairseq2/native/utils/string.h"
#include "fairseq2/native/data/immutable_string.h"
#include "fairseq2/native/data/file.h"

namespace fairseq2 {

memory_mapper::memory_mapper(
    std::optional<std::string> root_dir, std::optional<std::size_t> cached_fd_count) noexcept
  : cache_{/*capacity=*/cached_fd_count.value_or(default_cached_fd_count)}
{
    if (root_dir)
        root_dir_ = *std::move(root_dir);
}

data
memory_mapper::operator()(data &&d) const
{
    if (!d.is_string())
        throw std::invalid_argument{
            fmt::format("The input data must be of type `string`, but is of type `{}` instead.", d.type())};

    immutable_string &pathname = d.as_string();

    std::array<immutable_string, 3> parts{};

    auto iter = parts.begin();

    // The pathname can optionally contain offset and size specifiers separated
    // by semicolons (i.e. `pathname:offset:size`).
    pathname.split(':', [&pathname, &parts, &iter](immutable_string &&part)
    {
        if (iter == parts.end())
            throw std::invalid_argument{
                fmt::format("The input string must be a pathname with optional offset and size specifiers, but is '{}' instead.", pathname)};

        part = trim(part);

        if (part.empty())
            throw std::invalid_argument{
                fmt::format("The input string must be a pathname with optional offset and size specifiers, but is '{}' instead.", pathname)};

        *iter++ = std::move(part);
    });

    auto parse_specifier = [&pathname](std::string_view part, std::string_view specifier_name)
    {
        try {
            return from_string<std::size_t>(part);
        } catch (const std::out_of_range &) {
            throw std::invalid_argument{
                fmt::format("The {} specifier of '{}' must be a machine-representable integer, but is '{}' instead, which is out of range.", specifier_name, pathname, part)};
        } catch (const std::invalid_argument &) {
            throw std::invalid_argument{
                fmt::format("The {} specifier of '{}' must be an integer, but is '{}' instead.", specifier_name, pathname, part)};
        }
    };

    std::optional<std::size_t> offset{}, size{};

    if (!parts[1].empty()) {
        offset = parse_specifier(parts[1], "offset");

        if (!parts[2].empty())
            size = parse_specifier(parts[2], "size");
    }

    memory_block block = get_memory_map(parts[0]);

    auto pack_output = [&d](memory_block &&blk)
    {
        data_dict output{};

        output["path"] = std::move(d);
        output["data"] = std::move(blk);

        return output;
    };

    // If we don't have an offset, return the entire memory map.
    if (!offset)
        return pack_output(std::move(block));

    if (*offset > block.size())
        throw std::invalid_argument{
            fmt::format("The specified offset within '{}' must be less than or equal to the file size ({}), but is {} instead.", pathname, block.size(), *offset)};

    // If we have an offset but not a size, return the memory map from the
    // offset to the end.
    if (!size)
        return pack_output(block.share_slice(*offset));

    if (std::size_t upper_boundary = *offset + *size; upper_boundary > block.size())
        throw std::invalid_argument{
            fmt::format("The specified offset plus size within '{}' must be less than or equal to the file size ({}), but is {} instead.", pathname, block.size(), upper_boundary)};

    // Otherwise, return the memory map region specified by the offset and size.
    return pack_output(block.share_slice(*offset, *size));
}

memory_block
memory_mapper::get_memory_map(const immutable_string &pathname) const
{
    std::lock_guard<std::mutex> cache_lock{cache_mutex_};

    // Check the LRU cache first.
    if (memory_block *cached_block = cache_.get_if(pathname); cached_block != nullptr)
        return *cached_block;

    memory_block block = memory_map_file(root_dir_ / pathname.to_string());

    cache_.add(pathname, block);

    return block;
}

}  // namespace fairseq2
