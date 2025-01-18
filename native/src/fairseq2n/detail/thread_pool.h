// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>

namespace fairseq2n::detail {

class thread_pool {
public:
    explicit
    thread_pool(size_t numThreads) : stop(false) {
        
        workers.reserve(numThreads);
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });
                        
                        if (stop && tasks.empty()) {
                            return;
                        }
                        
                        num_working_++;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                    num_working_--;
                }
            });
        }
    }
    
    template<class F, class... Args>
    void
    enqueue(F&& f, Args&&... args) {
        auto task = std::make_shared<std::tuple<std::decay_t<F>, std::decay_t<Args>...>>(
            std::forward<F>(f), std::forward<Args>(args)...);
        
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (stop) {
                throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
            }
            
            tasks.emplace([task]() {
                std::apply(std::move(std::get<0>(*task)), 
                    [&task]() {
                        return std::tuple<Args...>(std::move(std::get<Args>(*task))...);
                    }());
            });
        }
        condition.notify_one();
    }
    
    ~thread_pool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

    // Delete copy constructor and assignment operator
    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;

    bool
    is_busy()
    {
        return num_working_ > 0;
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queueMutex;
    std::condition_variable condition;
    std::atomic<int> num_working_{0};
    bool stop;
};

}
