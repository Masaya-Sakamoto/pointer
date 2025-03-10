#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <vector>
#include <future>

class ThreadPool {
public:
    // Constructor with number of threads (default: hardware concurrency)
    ThreadPool(size_t numThreads = std::thread::hardware_concurrency()) {
        start(numThreads);
    }

    // Destructor
    ~ThreadPool() {
        stop();
    }

    // Delete copy constructor and assignment operator
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Submit a task to the thread pool and get a future for the result
    template<class F, class... Args>
    auto submit(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        // Create a packaged task with the function and its arguments
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        // Get the future from the task before pushing to the queue
        std::future<return_type> result = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            
            // Check if the pool is being stopped
            if (stop_flag) {
                throw std::runtime_error("submit on stopped ThreadPool");
            }
            
            // Wrap the packaged task in a void function
            tasks.emplace([task]() { (*task)(); });
        }
        
        // Notify one waiting thread that a task is available
        condition.notify_one();
        
        return result;
    }

    // Get the number of active threads in the pool
    size_t size() const {
        return workers.size();
    }

private:
    // Start the thread pool with a specific number of threads
    void start(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        
                        // Wait until there's a task or stop is called
                        condition.wait(lock, [this] {
                            return stop_flag || !tasks.empty();
                        });
                        
                        // Exit if the pool is being stopped and there are no more tasks
                        if (stop_flag && tasks.empty()) {
                            return;
                        }
                        
                        // Get the next task
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    
                    // Execute the task
                    task();
                }
            });
        }
    }

    // Stop the thread pool
    void stop() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop_flag = true;
        }
        
        // Notify all waiting threads
        condition.notify_all();
        
        // Join all threads
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    std::vector<std::thread> workers;              // Thread container
    std::queue<std::function<void()>> tasks;       // Task queue
    
    std::mutex queueMutex;                         // Mutex for queue access
    std::condition_variable condition;             // Condition variable for thread synchronization
    bool stop_flag = false;                        // Flag to indicate pool shutdown
};