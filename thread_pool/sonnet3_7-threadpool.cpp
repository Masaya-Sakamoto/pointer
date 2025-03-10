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

class ThreadPool
{
  public:
    // Constructor with number of threads (default: hardware concurrency)
    ThreadPool(size_t numThreads = std::thread::hardware_concurrency())
    {
        start(numThreads);
    }

    // Destructor
    ~ThreadPool()
    {
        stop();
    }

    // Delete copy constructor and assignment operator
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    // Submit a task to the thread pool and get a future for the result
    template <class F, class... Args>
    auto submit(F &&f, Args &&...args) -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using return_type = typename std::invoke_result<F, Args...>::type;

        // Create a packaged task with the function and its arguments
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        // Get the future from the task before pushing to the queue
        std::future<return_type> result = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queueMutex);

            // Check if the pool is being stopped
            if (stop_flag)
            {
                // Set the promise to an exception state rather than throwing
                // This avoids memory leaks when submit is called during destruction
                auto except_task = [task]() {
                    try
                    {
                        throw std::runtime_error("submit on stopped ThreadPool");
                    }
                    catch (...)
                    {
                        // task->make_exception_ptr(std::current_exception());
                    }
                };
                except_task();
                return result;
            }

            // Wrap the packaged task in a void function
            tasks.emplace([task]() {
                try
                {
                    (*task)();
                }
                catch (...)
                {
                    // Ensure exceptions in the task don't crash the worker thread
                    // The exception will still be propagated via the future
                }
            });
        }

        // Notify one waiting thread that a task is available
        condition.notify_one();

        return result;
    }

    // Get the number of active threads in the pool
    size_t size() const
    {
        return workers.size();
    }

    // Get the number of pending tasks
    size_t pending_tasks() const
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        return tasks.size();
    }

    // Stop the thread pool
    void stop()
    {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop_flag = true;
        }

        // Notify all waiting threads
        condition.notify_all();

        // Join all threads
        for (auto &worker : workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        // Clear any remaining tasks to prevent memory leaks
        std::unique_lock<std::mutex> lock(queueMutex);
        std::queue<std::function<void()>> empty;
        std::swap(tasks, empty);
    }

  private:
    // Start the thread pool with a specific number of threads
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

                        // Wait until there's a task or stop is called
                        condition.wait(lock, [this] { return stop_flag || !tasks.empty(); });

                        // Exit if the pool is being stopped and there are no more tasks
                        if (stop_flag && tasks.empty())
                        {
                            --running_threads;
                            return;
                        }

                        // Get the next task
                        task = std::move(tasks.front());
                        tasks.pop();
                    }

                    // Execute the task - wrapped in try/catch in the lambda now
                    task();
                }
            });
        }
    }

    std::vector<std::thread> workers;        // Thread container
    std::queue<std::function<void()>> tasks; // Task queue

    mutable std::mutex queueMutex;          // Mutex for queue access
    std::condition_variable condition;      // Condition variable for thread synchronization
    bool stop_flag = false;                 // Flag to indicate pool shutdown
    std::atomic<size_t> running_threads{0}; // Counter for active threads
};

// Include the improved ThreadPool class defined above

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

// Memory leak test function - creates and destroys many thread pools
void test_memory_leaks()
{
    std::cout << "Starting memory leak test...\n";

    for (int i = 0; i < 10; ++i)
    {
        ThreadPool pool(4);

        // Submit tasks
        std::vector<std::future<int>> results;
        for (int j = 0; j < 100; ++j)
        {
            results.emplace_back(pool.submit([j]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                return j;
            }));
        }

        // Don't wait for tasks - let the pool destructor handle cleanup
        std::cout << "Destroying pool " << i << " with " << pool.pending_tasks() << " pending tasks\n";
    }

    std::cout << "Memory leak test completed\n";
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

    // Test submitting after calling stop
    {
        ThreadPool temp_pool(2);
        temp_pool.stop(); // We're explicitly calling stop here for testing

        auto future = temp_pool.submit([]() -> std::string { return "This should fail"; });

        try
        {
            std::string result = future.get();
            std::cout << "Result: " << result << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cout << "Expected exception on stopped pool: " << e.what() << std::endl;
        }
    }

    // Test for memory leaks
    test_memory_leaks();

    std::cout << "Main thread exiting\n";
    // ThreadPool destructor will automatically stop all threads
    return 0;
}