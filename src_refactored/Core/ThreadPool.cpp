#include "Core/ThreadPool.h"
#include <iostream>

namespace CudaGame {
namespace Core {

ThreadPool::ThreadPool(size_t numThreads) {
    // Auto-detect thread count if not specified
    if (numThreads == 0) {
        numThreads = std::thread::hardware_concurrency();
        // Leave at least 1 core for main thread
        if (numThreads > 1) {
            numThreads -= 1;
        }
        // Clamp to reasonable range
        numThreads = std::max(size_t(2), std::min(numThreads, size_t(16)));
    }
    
    std::cout << "[ThreadPool] Initializing with " << numThreads << " worker threads" << std::endl;
    
    m_workers.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        m_workers.emplace_back(&ThreadPool::WorkerLoop, this);
    }
}

ThreadPool::~ThreadPool() {
    std::cout << "[ThreadPool] Shutting down..." << std::endl;
    
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_stop = true;
    }
    
    m_condition.notify_all();
    
    for (std::thread& worker : m_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    std::cout << "[ThreadPool] All workers stopped" << std::endl;
}

void ThreadPool::WorkerLoop() {
    while (true) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            
            // Wait for work or stop signal
            m_condition.wait(lock, [this] {
                return m_stop || !m_tasks.empty() || !m_priorityTasks.empty();
            });
            
            // Exit if stopped and no remaining work
            if (m_stop && m_tasks.empty() && m_priorityTasks.empty()) {
                return;
            }
            
            // Priority tasks first
            if (!m_priorityTasks.empty()) {
                task = std::move(m_priorityTasks.front());
                m_priorityTasks.pop();
            } else if (!m_tasks.empty()) {
                task = std::move(m_tasks.front());
                m_tasks.pop();
            } else {
                continue;
            }
            
            ++m_activeTasks;
        }
        
        // Execute task outside lock
        try {
            task();
        } catch (const std::exception& e) {
            std::cerr << "[ThreadPool] Task exception: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[ThreadPool] Task unknown exception" << std::endl;
        }
        
        --m_activeTasks;
        m_completionCondition.notify_all();
    }
}

size_t ThreadPool::GetPendingTaskCount() const {
    std::lock_guard<std::mutex> lock(m_queueMutex);
    return m_tasks.size() + m_priorityTasks.size() + m_activeTasks;
}

void ThreadPool::WaitForAll() {
    std::unique_lock<std::mutex> lock(m_queueMutex);
    m_completionCondition.wait(lock, [this] {
        return m_tasks.empty() && m_priorityTasks.empty() && m_activeTasks == 0;
    });
}

ThreadPool& ThreadPool::GetInstance() {
    static ThreadPool instance;
    return instance;
}

} // namespace Core
} // namespace CudaGame
