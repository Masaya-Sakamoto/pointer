#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <type_traits>
#include <vector>

// ThreadPool class definition with detailed comments explaining each part
class ThreadPool
{
  public:
    // Constructor initializes the thread pool with a specified number of threads
    ThreadPool(size_t numThreads = std::thread::hardware_concurrency())
    {
        start(numThreads);
    }

    // Destructor to clean up the thread pool resources
    ~ThreadPool()
    {
        stop();
    }

    // Copy constructor and assignment operator are deleted to prevent unintended copies
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    // Submit a task to the thread pool and get a future for its result
    template <class F, class... Args>
    auto submit(F &&f, Args &&...args) -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using return_type = typename std::invoke_result<F, Args...>::type;

        // Create a packaged task with the function and its arguments
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        // Get the future associated with this task
        std::future<return_type> result = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queueMutex);

            // Check if the pool is stopping; throw exception in such case
            if (stop_flag)
            {
                auto except_task = [task]() {
                    try
                    {
                        throw std::runtime_error("submit on stopped ThreadPool");
                    }
                    catch (...)
                    {
                        task->make_exception_ptr(std::current_exception());
                    }
                };
                except_task();
                return result;
            }

            // Enqueue the wrapped task
            tasks.emplace([task]() {
                try
                {
                    (*task)();
                }
                catch (...)
                {
                    // Catch exceptions to prevent worker thread crashes
                }
            });
        }

        condition.notify_one();

        return result;
    }

    // Get the current number of active threads in the pool
    size_t size() const
    {
        return workers.size();
    }

    // Get the number of pending tasks in the queue
    size_t pending_tasks() const
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        return tasks.size();
    }

  private:
    // Initialize the thread pool with a specified number of threads
    void start(size_t numThreads)
    {
        running_threads = numThreads;
        for (size_t i = 0; i < numThreads; ++i)
        {
            workers.emplace_back([this] {
                while (true)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this]() { return stop_flag || !tasks.empty(); });

                        if (stop_flag && tasks.empty())
                        {
                            --running_threads;
                            return;
                        }

                        task = std::move(tasks.front());
                        tasks.pop();
                    }

                    try
                    {
                        task();
                    }
                    catch (...)
                    {
                        // Catch exceptions to prevent worker thread crashes
                    }
                }
            });
        }
    }

    // Stop all threads and clean up the pool resources
    void stop()
    {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop_flag = true;
        }

        condition.notify_all();

        for (auto &worker : workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        std::unique_lock<std::mutex> lock(queueMutex);
        std::queue<std::function<void()>> empty;
        std::swap(tasks, empty);
    }

    // Vector to hold the worker threads
    std::vector<std::thread> workers;

    // Queue to store pending tasks
    std::queue<std::function<void()>> tasks;

    // Mutex for synchronizing access to the task queue
    mutable std::mutex queueMutex;

    // Condition variable for thread synchronization
    std::condition_variable condition;

    // Flag indicating whether the pool is stopping
    bool stop_flag = false;

    // Atomic counter for tracking running threads
    std::atomic<size_t> running_threads{0};
};

class Task
{
  public:
    Task(int id, int duration) : id(id), duration(duration)
    {
    }

    int execute()
    {
        std::cout << "Task " << id << " started on thread " << std::this_thread::get_id() << std::endl;

        // Simulate work with sleep
        std::this_thread::sleep_for(std::chrono::milliseconds(duration));

        std::cout << "Task " << id << " completed after " << duration << "ms\n";
        return id * 10 + duration;
    }

  private:
    int id;
    int duration;
};

// Memory leak test function that doesn't explicitly call stop()
void test_memory_leaks()
{
    std::cout << "Starting memory leak test...\n";

    for (int i = 0; i < 5; ++i)
    {
        // Create a thread pool in a scope
        {
            ThreadPool pool(4);

            // Submit tasks
            std::vector<std::future<int>> results;
            for (int j = 0; j < 50; ++j)
            {
                results.emplace_back(pool.submit([j]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    return j;
                }));
            }

            // Don't wait for tasks - let the pool destructor handle cleanup
            std::cout << "Scope ending with " << pool.pending_tasks() << " pending tasks\n";

            // The pool will be destroyed when leaving this scope, which should trigger stop()
        }
        std::cout << "Pool " << i << " destroyed\n";
    }

    std::cout << "Memory leak test completed\n";
}

// Function to test exception propagation without explicit stop()
void test_exception_handling()
{
    std::cout << "\nTesting exception handling...\n";

    ThreadPool pool(2);

    // Submit a task that throws an exception
    auto future = pool.submit([]() -> int {
        throw std::runtime_error("Task failed with exception");
        return 42;
    });

    // This will rethrow the exception
    try
    {
        int result = future.get();
        std::cout << "Result: " << result << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }
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

    // Submit tasks to the pool using Task class
    for (int i = 0; i < 10; ++i)
    {
        int duration = dist(gen);
        // Using shared_ptr to ensure proper memory management
        auto task = std::make_shared<Task>(i, duration);
        // Capture by value to avoid dangling references
        results.emplace_back(pool.submit([task]() { return task->execute(); }));
    }

    // Get and print results
    std::cout << "\nResults:\n";
    for (size_t i = 0; i < results.size(); ++i)
    {
        try
        {
            std::cout << "Task " << i << " result: " << results[i].get() << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cout << "Task " << i << " failed: " << e.what() << std::endl;
        }
    }

    // Test exception handling separately
    test_exception_handling();

    // Test for memory leaks
    test_memory_leaks();

    // Submit a few more tasks to demonstrate that the pool is still working
    std::cout << "\nSubmitting additional tasks...\n";
    std::vector<std::future<std::string>> additional_results;

    for (int i = 0; i < 5; ++i)
    {
        additional_results.emplace_back(pool.submit([i]() -> std::string {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            return "Additional task " + std::to_string(i) + " completed";
        }));
    }

    // Get and print additional results
    for (auto &future : additional_results)
    {
        std::cout << future.get() << std::endl;
    }

    std::cout << "\nMain thread exiting - thread pool will be automatically stopped in destructor\n";
    // ThreadPool destructor will automatically stop all threads
    return 0;
}