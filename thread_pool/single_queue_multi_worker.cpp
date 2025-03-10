#include <condition_variable>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

// A worker pool template class designed to manage a fixed number of worker threads.
template <size_t Count> class worker_pool
{
  public:
    // Constructor initializes each inner worker in the pool.
    worker_pool()
    {
        int index = 0;
        for (auto &inner : inner_workers_)
        {
            inner.initialize(this, index++);
        }
    }

    // Destructor ensures all workers terminate gracefully after waiting until idle.
    ~worker_pool()
    {
        wait_until_idle();
        request_termination();
    };

    // Submits a task to the worker pool's queue and notifies all workers.
    template <typename F> void run(F &&task)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (is_requested_termination)
        {
            return;
        }

        queue_.emplace_back(task);
        for (auto &inner : inner_workers_)
        {
            inner.wakeup();
        }
    }

    // Waits until all tasks in the queue are processed and workers are idle.
    void wait_until_idle()
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait(lock, [this]() { return queue_.empty() || is_requested_termination; });
        }
        for (auto &inner : inner_workers_)
        {
            inner.wait_until_idle();
        }
    }

    // Requests termination of all workers and notifies them.
    void request_termination()
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            is_requested_termination = true;
            cond_.notify_all();
        }
        for (auto &inner : inner_workers_)
        {
            inner.request_termination();
        }
    }

    // Retrieves and removes the front task from the queue, returning it as a function.
    std::function<void()> pull()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty())
            return {};

        auto task = queue_.front();
        queue_.pop_front();
        cond_.notify_all();
        return task;
    }

  private:
    // Inner worker class representing individual threads within the pool.
    class inner_worker
    {
      public:
        // Constructor initializes a new thread for the worker.
        inner_worker() : thread_([this]() { proc_worker(); })
        {
        }

        // Destructor ensures the worker thread terminates properly after waiting until idle.
        ~inner_worker()
        {
            wait_until_idle();
            request_termination();
            if (thread_.joinable())
            {
                thread_.join();
            }
        };

        // Initializes the inner worker with a parent pool and index.
        void initialize(worker_pool *parent, int index)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            parent_ = parent;
            index_ = index;
            cond_.notify_all();
        }

        // Wakes up the worker by notifying its condition variable.
        void wakeup()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.notify_all();
        }

        void wait_until_idle()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait(lock, [this]() { return !current_task_ || is_requested_termination; });
        }

        // Requests termination of this worker and notifies its condition variable.
        void request_termination()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            is_requested_termination = true;
            cond_.notify_all();
        }

      private:
        // The main processing loop for the inner worker thread.
        void proc_worker()
        {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_.wait(lock, [this]() { return parent_ != nullptr && index_ >= 0; });
            }
            while (true)
            {
                auto task = parent_->pull();
                if (!task)
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    current_task_ = {};
                    cond_.notify_all();
                    cond_.wait(lock, [&]() { return !!(current_task_ = parent_->pull()) || is_requested_termination; });
                    if (is_requested_termination)
                        break;
                }
                else
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    current_task_ = std::move(task);
                }

                current_task_();
            }
        }

        // Pointer to the parent worker pool.
        worker_pool *parent_{nullptr};
        // Index identifying this inner worker within the pool.
        int index_{-1};
        // The currently executing task, if any.
        std::function<void()> current_task_{};
        // Flag indicating whether termination has been requested.
        bool is_requested_termination{false};
        // The thread object representing this inner worker.
        std::thread thread_;
        // Mutex for synchronizing access to the inner worker's state.
        std::mutex mutex_;
        // Condition variable for controlling the worker's execution.
        std::condition_variable cond_;
    };

    // Array of inner_workers, one for each thread in the pool.
    inner_worker inner_workers_[Count];
    // Flag indicating whether termination has been requested for the entire pool.
    bool is_requested_termination{false};
    // Queue holding tasks to be executed by workers.
    std::deque<std::function<void()>> queue_;
    // Mutex for synchronizing access to the worker pool's state and queue.
    std::mutex mutex_;
    // Condition variable for controlling the worker pool's execution.
    std::condition_variable cond_;
};

int main()
{
    // 4スレッドのワーカープールを作成
    worker_pool<4> pool;

    // タスクを送信
    for (int i = 0; i < 10; ++i)
    {
        pool.run([i]() {
            std::cout << "タスク " << i << " がスレッド " << std::this_thread::get_id() << " によって実行されました。"
                      << std::endl;
        });
    }

    // すべてのタスクが完了するまで待機
    pool.wait_until_idle();

    return 0;
}