#include <gtest/gtest.h>
#include "ThreadPool.h"
#include <atomic>
#include <functional>
#include <vector>

namespace {
class IndexedAccumulator {
public:
    explicit IndexedAccumulator(int n) : values_(n, 0) {}
    void add(int idx, int val) { values_[idx] += val; }
    int get(int idx) const { return values_[idx]; }

private:
    std::vector<int> values_;
};
} // namespace

TEST(ThreadPoolTest, ExecutesJobsAcrossMultipleWaitBarriers) {
    ThreadPool* pool = ThreadPool::GetPool(4);
    ASSERT_NE(pool, nullptr);

    std::atomic<int> counter(0);
    const int first_batch = 200;
    for (int i = 0; i < first_batch; ++i) {
        THREADS.AddJob([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    }
    THREADS.WaitAll();
    EXPECT_EQ(counter.load(), first_batch);

    const int second_batch = 150;
    for (int i = 0; i < second_batch; ++i) {
        THREADS.AddJob([&counter]() {
            counter.fetch_add(2, std::memory_order_relaxed);
        });
    }
    THREADS.WaitAll();
    EXPECT_EQ(counter.load(), first_batch + 2 * second_batch);

    const int n_slots = 64;
    IndexedAccumulator acc(n_slots);
    for (int i = 0; i < n_slots; ++i) {
        THREADS.AddJob(std::bind(&IndexedAccumulator::add, &acc, i, i * 3 + 1));
    }
    THREADS.WaitAll();
    for (int i = 0; i < n_slots; ++i) {
        EXPECT_EQ(acc.get(i), i * 3 + 1);
    }
}
