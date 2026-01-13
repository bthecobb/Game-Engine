#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <memory>

namespace CudaGame {
namespace Core {

/**
 * @brief AAA-quality thread pool for parallel job execution
 * 
 * Used for:
 * - Async chunk generation
 * - Procedural mesh building
 * - Background asset loading
 * - Physics simulation tasks
 */
class ThreadPool {
public:
    /**
     * @brief Construct thread pool with specified number of workers
     * @param numThreads Number of worker threads (0 = auto-detect)
     */
    explicit ThreadPool(size_t numThreads = 0);
    
    /**
     * @brief Destructor - waits for all tasks to complete
     */
    ~ThreadPool();
    
    // Non-copyable, non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
    
    /**
     * @brief Enqueue a task for execution
     * @tparam F Callable type
     * @tparam Args Argument types
     * @param f Function to execute
     * @param args Arguments to pass
     * @return std::future for the result
     */
    template<class F, class... Args>
    auto Enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type>;
    
    /**
     * @brief Enqueue a high-priority task (executed before normal tasks)
     */
    template<class F, class... Args>
    auto EnqueuePriority(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;
    
    /**
     * @brief Get number of worker threads
     */
    size_t GetWorkerCount() const { return m_workers.size(); }
    
    /**
     * @brief Get number of pending tasks
     */
    size_t GetPendingTaskCount() const;
    
    /**
     * @brief Check if pool is running
     */
    bool IsRunning() const { return !m_stop; }
    
    /**
     * @brief Wait for all current tasks to complete
     */
    void WaitForAll();
    
    /**
     * @brief Get singleton instance
     */
    static ThreadPool& GetInstance();

private:
    std::vector<std::thread> m_workers;
    std::queue<std::function<void()>> m_tasks;
    std::queue<std::function<void()>> m_priorityTasks;
    
    mutable std::mutex m_queueMutex;
    std::condition_variable m_condition;
    std::condition_variable m_completionCondition;
    
    std::atomic<bool> m_stop{false};
    std::atomic<size_t> m_activeTasks{0};
    
    void WorkerLoop();
};

// ============== Template Implementation ==============

template<class F, class... Args>
auto ThreadPool::Enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::invoke_result<F, Args...>::type>
{
    using ReturnType = typename std::invoke_result<F, Args...>::type;
    
    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<ReturnType> result = task->get_future();
    
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        
        if (m_stop) {
            throw std::runtime_error("ThreadPool: Cannot enqueue on stopped pool");
        }
        
        m_tasks.emplace([task]() { (*task)(); });
    }
    
    m_condition.notify_one();
    return result;
}

template<class F, class... Args>
auto ThreadPool::EnqueuePriority(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type>
{
    using ReturnType = typename std::invoke_result<F, Args...>::type;
    
    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<ReturnType> result = task->get_future();
    
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        
        if (m_stop) {
            throw std::runtime_error("ThreadPool: Cannot enqueue on stopped pool");
        }
        
        m_priorityTasks.emplace([task]() { (*task)(); });
    }
    
    m_condition.notify_one();
    return result;
}

} // namespace Core
} // namespace CudaGame
