#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

template <typename T> class OrderedThreadPool
{
  private:
    /*
    2. **OrderedResult Structure**
        - Pairs each result with its sequence ID
        - Custom comparator creates a min-heap in the priority queue (smallest ID at top)
        - Note the reversed comparison (`>`) which is necessary for priority_queue
    */
    struct OrderedResult
    {
        size_t sequence_id;
        T result;

        // For priority queue (sorts by sequence_id)
        bool operator<(const OrderedResult &other) const
        {
            return sequence_id > other.sequence_id; // Note: reversed for min-heap
        }
    };

    /*
    3. **Thread Pool Resources**
        - `workers`: Collection of worker threads
        - `tasks`: Queue of pending work
        - `results`: Priority queue that automatically sorts results by sequence ID
    */
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::priority_queue<OrderedResult> results;

    /*
    1. **Sequence ID Tracking**
        - `next_sequence_id`: Assigns a unique sequential ID to each incoming task
        - `next_to_emit`: Tracks the next expected sequence ID for output
    */
    size_t next_sequence_id = 0;
    size_t next_to_emit = 0;

    /*
    4. **Synchronization Primitives**
        - `queue_mutex`/`condition`: Protects task queue access and signals workers
        - `results_mutex`/`results_condition`: Protects result queue and signals when results are ready
        - `stop`: Flag for graceful shutdown
    */
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;

    std::mutex results_mutex;
    std::condition_variable results_condition;

  public:
    /*
    5. **Constructor**
        - Creates specified number of worker threads
        - Each worker runs a loop that:
            - Waits for a task or stop signal
            - Takes a task from the queue if available
            - Executes the task
    */
    OrderedThreadPool(size_t num_threads)
    {
        for (size_t i = 0; i < num_threads; ++i)
        {
            workers.emplace_back([this] {
                while (true)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty())
                            return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    /*
6. **Task Enqueue Method**
        - Assigns a sequence ID to the task
        - Wraps the task in a `std::packaged_task` and gets a future
        - Creates a lambda that:
            - Executes the original task
            - Captures the result with its sequence ID
            - Adds the result to the priority queue
            - Notifies any waiting consumers
        - Returns a future for async access to the result
    */
    template <typename F, typename... Args> std::future<T> enqueue(F &&f, Args &&...args)
    {
        size_t id = next_sequence_id++;

        auto task_ptr =
            std::make_shared<std::packaged_task<T()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<T> res = task_ptr->get_future();

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            tasks.emplace([this, id, task_ptr]() {
                (*task_ptr)();
                OrderedResult result{id, task_ptr->get_future().get()};

                std::unique_lock<std::mutex> lock(results_mutex);
                results.push(result);
                results_condition.notify_one();
            });
        }
        condition.notify_one();
        return res;
    }

    /*
    7. **Ordered Result Retrieval**
        - Waits until the next sequential result is available
        - Extracts and returns the result
        - Increments the expected sequence counter
    */
    T get_next()
    {
        std::unique_lock<std::mutex> lock(results_mutex);
        results_condition.wait(lock, [this] { return !results.empty() && results.top().sequence_id == next_to_emit; });

        auto result = results.top().result;
        results.pop();
        next_to_emit++;
        return result;
    }

    /*
    8. **Destructor**
        - Sets stop flag to terminate worker threads
        - Notifies all threads to check the stop flag
        - Joins all worker threads to ensure clean shutdown
    */
    ~OrderedThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
        {
            worker.join();
        }
    }
};

/*
### Comprehensive Explanation

The `OrderedThreadPool` class solves the problem of maintaining order when processing data in parallel. Here's how it
all works together:

1. **Initialization**: When you create an `OrderedThreadPool`, you specify the number of worker threads. Each thread
runs continuously looking for tasks to process.

2. **Submission Process**:
   - When you submit a task via `enqueue()`, it gets assigned a unique sequence ID.
   - The task is wrapped in a function that will capture both the result and its sequence ID.
   - This wrapped task is placed in the task queue for any available worker to pick up.

3. **Parallel Processing**:
   - Worker threads independently take tasks from the queue.
   - Tasks with different processing times will complete in an unpredictable order.
   - When a task completes, its result and sequence ID are placed in the priority queue.

4. **Ordered Retrieval**:
   - The `get_next()` method ensures results are retrieved in the correct sequence.
   - It waits until the next expected sequence number is available at the top of the priority queue.
   - This creates a "holding pattern" where out-of-order results wait in the priority queue until their turn.

5. **Flow Control**:
   - If task 1, 3, and 4 complete quickly but task 2 takes longer, tasks 3 and 4 will wait in the priority queue.
   - Once task 2 completes, `get_next()` will return results for tasks 2, 3, and 4 in that order.

This pattern effectively decouples the parallel processing from the sequential consumption of results, giving you both
the performance benefits of parallel execution and the correctness of sequential processing.

The implementation uses standard C++ threading primitives and containers, making it portable and efficient. The use of
`std::future` allows both synchronous and asynchronous access to results, while the priority queue automatically handles
the reordering of results based on their sequence IDs.

Think of it like a numbered ticket system at a service counter: customers may be served at different speeds, but they
exit in the order of their ticket numbers.
*/