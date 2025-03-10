#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

template<size_t Count>
class worker_pool {
public:
    worker_pool() {
        int index = 0;
        for (auto& inner : inner_workers_) {
            inner.initialize(this, index++);
        }
    }
    ~worker_pool() {
        wait_until_idle();
        request_termination();
    };

    template<typename F>
    void run(F&& task) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (is_requested_termination) { return; }

        queue_.emplace_back(task);
        for (auto& inner : inner_workers_) {
            inner.wakeup();
        }
    }

    void wait_until_idle() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait(lock, [this]() { return queue_.empty() || is_requested_termination; });
        }
        for (auto& inner : inner_workers_) {
            inner.wait_until_idle();
        }
    }

    void request_termination() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            is_requested_termination = true;
            cond_.notify_all();
        }
        for (auto& inner : inner_workers_) {
            inner.request_termination();
        }
    }

    std::function<void()> pull() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) return {};

        auto task = queue_.front();
        queue_.pop_front();
        cond_.notify_all();
        return task;
    }

private:
    class inner_worker {
    public:
        inner_worker() : thread_([this]() { proc_worker(); }) {}
        ~inner_worker() {
            wait_until_idle();
            request_termination();
            if (thread_.joinable()) {
                thread_.join();
            }
        };
	
        void initialize(worker_pool* parent, int index) {
            std::unique_lock<std::mutex> lock(mutex_);
            parent_ = parent;
            index_ = index;
            cond_.notify_all();
        }
        
	void wakeup() {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.notify_all();
        }
        
	void wait_until_idle() {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait(lock, [this]() { return !current_task_ || is_requested_termination; });
        }
        
	void request_termination() {
            std::unique_lock<std::mutex> lock(mutex_);
            is_requested_termination = true;
            cond_.notify_all();
        }
	
    private:
        void proc_worker() {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_.wait(lock, [this]() { return parent_ != nullptr && index_ >= 0; });
            }
            while (true) {
                auto task = parent_->pull();
                if (!task) {
                    std::unique_lock<std::mutex> lock(mutex_);
                    current_task_ = {};
                    cond_.notify_all();
                    cond_.wait(lock, [&]() { return !!(current_task_ = parent_->pull()) || is_requested_termination; });
                    if (is_requested_termination) break;
                } else {
                    std::unique_lock<std::mutex> lock(mutex_);
                    current_task_ = std::move(task);
                }

                current_task_();
            }
        }

        worker_pool* parent_{ nullptr };
        int index_{ -1 };
        std::function<void()> current_task_{};
        bool is_requested_termination{ false };
        std::thread thread_;
        std::mutex mutex_;
        std::condition_variable cond_;
    };

    inner_worker inner_workers_[Count];
    bool is_requested_termination{ false };
    std::deque<std::function<void()>> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
};
