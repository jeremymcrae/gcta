#include <gtest/gtest.h>
#include <string>

extern int test_argc;
extern char** test_argv;

namespace {
const char* kActualTrue = "hello gtest";
const char* kActualFalse = "hello world";
const char* kExpected = "hello gtest";

class SampleFixture : public ::testing::Test {
protected:
    int value = 0;
    void SetUp() override { value = 42; }
    void TearDown() override { value = 0; }
};
} // namespace

TEST(StrCompare, CStrEqual) {
    EXPECT_GE(test_argc, 1);
    EXPECT_NE(test_argv[0], nullptr);
    EXPECT_STREQ(kExpected, kActualTrue);
}

TEST(StrCompare, CStrNotEqual) {
    EXPECT_STRNE(kExpected, kActualFalse);
}

TEST_F(SampleFixture, FixtureLifecycleWorks) {
    EXPECT_EQ(value, 42);
    value += 8;
    EXPECT_EQ(value, 50);
}
