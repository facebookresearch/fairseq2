// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <list>
#include <unordered_map>
#include <utility>

#include "fairseq2n/data/immutable_string.h"

namespace fairseq2n::detail {

template <typename T>
class lru_cache {
    using entry_list = std::list<std::pair<immutable_string, T>>;

public:
    explicit
    lru_cache(std::size_t capacity) noexcept
      : capacity_{capacity}
    {}

    // Adds `key`/`value` pair to the cache, and evicts the least-recently-used
    // entry if the cache is at its capacity.
    void
    add(immutable_string key, T value);

    // If `key` is in the cache, marks it as the most-recently-used entry, and
    // returns its value; otherwise, returns `nullptr`.
    T *
    maybe_get(const immutable_string &key);

    void
    clear() noexcept
    {
        map_.clear();

        entries_.clear();
    }

    std::size_t
    size() const noexcept
    {
        return map_.size();
    }

private:
    std::size_t capacity_;
    std::unordered_map<immutable_string, typename entry_list::iterator> map_{};
    entry_list entries_{};
};

template <typename T>
void
lru_cache<T>::add(immutable_string key, T value)
{
    if (capacity_ == 0)
        return;

    // If the key is already in the cache, just update its value.
    if (auto pos = map_.find(key); pos != map_.end()) {
        auto &entry_node = pos->second;

        // Move the entry to the front of the list to mark as the most-
        // recently-used one.
        entries_.splice(entries_.begin(), entries_, entry_node);

        entry_node->second = std::move(value);

        return;
    }

    // Otherwise, ensure that we do not exceed our capacity.
    if (map_.size() == capacity_) {
        // The back of the list holds the least-recently-used entry. Let's
        // remove it to make space for the new entry.
        immutable_string &least_recent_key = entries_.back().first;

        map_.erase(least_recent_key);

        entries_.pop_back();
    }

    // Put the entry to the front of the list so that it becomes the most-
    // recently-used one.
    entries_.emplace_front(key, std::move(value));

    // And, associate the key and the entry in the map.
    map_[std::move(key)] = entries_.begin();
}

template <typename T>
T *
lru_cache<T>::maybe_get(const immutable_string &key)
{
    if (auto pos = map_.find(key); pos != map_.end()) {
        auto &entry_node = pos->second;

        // Move the entry to the front of the list to mark as the most-
        // recently-used one.
        entries_.splice(entries_.begin(), entries_, entry_node);

        return &entry_node->second;
    }

    return nullptr;
}

}  // namespace fairseq2n::detail
