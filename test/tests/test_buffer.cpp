#include <gtest/gtest.h>
#include "AsyncBuffer.hpp"
#include <chrono>
#include <cstdint>
#include <thread>
#include <tuple>

TEST(AsyncBufferTest, AlignmentAndInitStatus) {
    AsyncBuffer<double> buf(128);
    ASSERT_TRUE(buf.init_status());
    double* slot = buf.start_write();
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(slot) % 64ULL, 0ULL);
    buf.setEOF();
    buf.end_write();

    double* rslot = nullptr;
    bool eof = false;
    std::tie(rslot, eof) = buf.start_read();
    EXPECT_EQ(rslot, slot);
    EXPECT_TRUE(eof);
    buf.end_read();
}

TEST(AsyncBufferTest, ProducerConsumerPreserveOrderAndValues) {
    const int n_chunks = 64;
    AsyncBuffer<uint8_t> buf(3);
    ASSERT_TRUE(buf.init_status());

    std::thread writer([&buf, n_chunks]() {
        for (int i = 0; i < n_chunks; ++i) {
            uint8_t* slot = buf.start_write();
            slot[0] = static_cast<uint8_t>(i);
            slot[1] = static_cast<uint8_t>(i + 10);
            slot[2] = static_cast<uint8_t>(i + 100);
            if (i + 1 == n_chunks) {
                buf.setEOF();
            }
            buf.end_write();
        }
    });

    int seen = 0;
    while (true) {
        bool eof = false;
        uint8_t* slot = nullptr;
        std::tie(slot, eof) = buf.start_read();
        ASSERT_NE(slot, nullptr);
        EXPECT_EQ(slot[0], static_cast<uint8_t>(seen));
        EXPECT_EQ(slot[1], static_cast<uint8_t>(seen + 10));
        EXPECT_EQ(slot[2], static_cast<uint8_t>(seen + 100));
        buf.end_read();
        ++seen;
        if (eof) break;
    }
    writer.join();
    EXPECT_EQ(seen, n_chunks);
}

TEST(AsyncBufferTest, BackpressureWhenConsumerIsSlow) {
    const int n_chunks = 20;
    const int block_words = 256;
    AsyncBuffer<uint64_t> buf(block_words);
    ASSERT_TRUE(buf.init_status());

    std::thread writer([&buf, n_chunks, block_words]() {
        for (int c = 0; c < n_chunks; ++c) {
            uint64_t* slot = buf.start_write();
            for (int k = 0; k < block_words; ++k) {
                slot[k] = static_cast<uint64_t>(c) * 10000ULL + static_cast<uint64_t>(k);
            }
            if (c + 1 == n_chunks) {
                buf.setEOF();
            }
            buf.end_write();
        }
    });

    int read_chunks = 0;
    while (true) {
        bool eof = false;
        uint64_t* slot = nullptr;
        std::tie(slot, eof) = buf.start_read();
        ASSERT_NE(slot, nullptr);
        if (read_chunks % 5 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        for (int k = 0; k < block_words; ++k) {
            ASSERT_EQ(slot[k], static_cast<uint64_t>(read_chunks) * 10000ULL + static_cast<uint64_t>(k));
        }
        buf.end_read();
        ++read_chunks;
        if (eof) break;
    }
    writer.join();
    EXPECT_EQ(read_chunks, n_chunks);
}

TEST(AsyncBufferTest, SentinelEofChunkSignalsEndCleanly) {
    AsyncBuffer<uintptr_t> buf(16);
    ASSERT_TRUE(buf.init_status());
    std::thread writer([&buf]() {
        uintptr_t* slot = buf.start_write();
        slot[0] = 42ULL;
        buf.setEOF();
        buf.end_write();
    });

    bool eof = false;
    uintptr_t* slot = nullptr;
    std::tie(slot, eof) = buf.start_read();
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot[0], 42ULL);
    EXPECT_TRUE(eof);
    buf.end_read();
    writer.join();
}
