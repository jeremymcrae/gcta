#include <gtest/gtest.h>
#include "tables.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

namespace {
void ExpectedCounts16(uint16_t word, uint16_t out[4]) {
    out[0] = out[1] = out[2] = out[3] = 0;
    for (int shift = 0; shift < 16; shift += 2) {
        const uint8_t code = static_cast<uint8_t>((word >> shift) & 0x3u);
        switch (code) {
            case 0x0: out[0]++; break; // 00 = hom1 (A1A1)
            case 0x2: out[1]++; break; // 10 = het  (A1A2)
            case 0x3: out[2]++; break; // 11 = hom2 (A2A2)
            default:  out[3]++; break; // 01 = missing
        }
    }
}
} // namespace

TEST(Table, ZeroAndConstantPatterns) {
    std::unique_ptr<GBitCountTable> g_table(new GBitCountTable());
    // 0x0000 = eight 00 genotypes -> 8 A1A1
    EXPECT_EQ(8, g_table->get(0x0000, 0));
    EXPECT_EQ(0, g_table->get(0x0000, 1));
    EXPECT_EQ(0, g_table->get(0x0000, 2));
    EXPECT_EQ(0, g_table->get(0x0000, 3));

    // 0xAAAA = 0b10101010_10101010 -> eight 10 (het) genotypes
    EXPECT_EQ(0, g_table->get(0xAAAA, 0));
    EXPECT_EQ(8, g_table->get(0xAAAA, 1));
    EXPECT_EQ(0, g_table->get(0xAAAA, 2));
    EXPECT_EQ(0, g_table->get(0xAAAA, 3));

    // 0xFFFF = eight 11 (hom2) genotypes
    EXPECT_EQ(0, g_table->get(0xFFFF, 0));
    EXPECT_EQ(0, g_table->get(0xFFFF, 1));
    EXPECT_EQ(8, g_table->get(0xFFFF, 2));
    EXPECT_EQ(0, g_table->get(0xFFFF, 3));

    // 0x5555 = eight 01 (missing) genotypes
    EXPECT_EQ(0, g_table->get(0x5555, 0));
    EXPECT_EQ(0, g_table->get(0x5555, 1));
    EXPECT_EQ(0, g_table->get(0x5555, 2));
    EXPECT_EQ(8, g_table->get(0x5555, 3));
}

TEST(Table, Exhaustive65536LookupAndSetCount) {
    // Zero-initialize backing storage first so we isolate lookup arithmetic from constructor init
    std::vector<uint8_t> zero_mem(sizeof(GBitCountTable), 0);
    GBitCountTable* tbl = new (zero_mem.data()) GBitCountTable();

    for (uint32_t idx = 0; idx < 65536u; ++idx) {
        uint16_t exp[4];
        ExpectedCounts16(static_cast<uint16_t>(idx), exp);
        const uint16_t w = static_cast<uint16_t>(idx);
        ASSERT_EQ(exp[0], tbl->get(w, 0)) << "mismatch at index=" << idx << " col=0";
        ASSERT_EQ(exp[1], tbl->get(w, 1)) << "mismatch at index=" << idx << " col=1";
        ASSERT_EQ(exp[2], tbl->get(w, 2)) << "mismatch at index=" << idx << " col=2";
        ASSERT_EQ(exp[3], tbl->get(w, 3)) << "mismatch at index=" << idx << " col=3";
        ASSERT_EQ(8, tbl->get(w, 0) + tbl->get(w, 1) + tbl->get(w, 2) + tbl->get(w, 3));
    }

    std::vector<uint16_t> words = {0x0000, 0xAAAA, 0xFFFF, 0x5555, 0xE4E4};
    uint32_t a1a1 = 0, a1a2 = 0, a2a2 = 0;
    tbl->set_count(words, a1a1, a1a2, a2a2);
    // 0xE4 = 0b11_10_01_00 -> one of each across low and high byte = 2 of each
    EXPECT_EQ(8u + 2u, a1a1);
    EXPECT_EQ(8u + 2u, a1a2);
    EXPECT_EQ(8u + 2u, a2a2);
    tbl->~GBitCountTable();
}

// Regression test: GBitCountTable constructor must initialize geno_count8 even when
// constructed in non-zeroed stack/heap memory.
TEST(Table, PlacementNewOnDirtyMemoryInitializesCleanly) {
    const size_t align = alignof(GBitCountTable);
    std::vector<uint8_t> dirty_mem(sizeof(GBitCountTable) + align, 0xAB);
    uintptr_t raw = reinterpret_cast<uintptr_t>(dirty_mem.data());
    uintptr_t aligned = (raw + align - 1) & ~(static_cast<uintptr_t>(align - 1));
    GBitCountTable* tbl = new (reinterpret_cast<void*>(aligned)) GBitCountTable();
    EXPECT_EQ(8, tbl->get(0x0000, 0));
    EXPECT_EQ(0, tbl->get(0x0000, 1));
    EXPECT_EQ(0, tbl->get(0x0000, 2));
    EXPECT_EQ(0, tbl->get(0x0000, 3));
    EXPECT_EQ(8, tbl->get(0xAAAA, 1));
    EXPECT_EQ(8, tbl->get(0xFFFF, 2));
    EXPECT_EQ(8, tbl->get(0x5555, 3));
    tbl->~GBitCountTable();
}
