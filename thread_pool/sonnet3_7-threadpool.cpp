#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <vector>
#include <future>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>

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


// Include the ThreadPool class defined above

int calculate(int id, int duration)
{
    std::cout << "Task " << id << " started on thread " << std::this_thread::get_id() << std::endl;

    // Simulate work with sleep
    // std::this_thread::sleep_for(std::chrono::milliseconds(duration));

    std::cout << "Task " << id << " completed after " << duration << "ms\n";
    return id * 10 + duration;
}

int main()
{
    // Create a thread pool with 4 threads
    ThreadPool pool(4);
    std::cout << "Thread pool created with " << pool.size() << " threads\n";

    // Random number generator for task durations
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(100, 1000);

    // Vector to store futures for result retrieval
    std::vector<std::future<int>> results;

    // Submit tasks to the pool
    for (int i = 0; i < INT32_MAX; ++i)
    {
        int duration = dist(gen);
        results.emplace_back(pool.submit(calculate, i, duration));
    }

    // Get and print results
    std::cout << "\nResults:\n";
    for (size_t i = 0; i < results.size(); ++i)
    {
        std::cout << "Task " << i << " result: " << results[i].get() << std::endl;
    }

    // Example of handling exceptions
    try
    {
        auto future = pool.submit([]() -> int {
            throw std::runtime_error("Task failed with exception");
            return 42;
        });

        // This will rethrow the exception
        future.get();
    }
    catch (const std::exception &e)
    {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }

    // Demonstrate a task that returns a string
    auto stringTask = pool.submit([]() -> std::string {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return "Hello from thread pool!";
    });

    std::cout << "String task result: " << stringTask.get() << std::endl;

    std::cout << "Main thread exiting\n";
    // ThreadPool destructor will automatically stop all threads
    return 0;
}