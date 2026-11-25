#ifndef GCTA_LITE_GTEST_H
#define GCTA_LITE_GTEST_H

#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace testing {

class Test {
public:
    virtual ~Test() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
    virtual void TestBody() = 0;
};

struct FatalAssertionFailure : public std::exception {
    const char* what() const noexcept override { return "Fatal test assertion failed"; }
};

struct TestCaseEntry {
    std::string suite;
    std::string name;
    std::function<void()> run;
};

inline std::vector<TestCaseEntry>& GetRegistry() {
    static std::vector<TestCaseEntry> reg;
    return reg;
}

inline int& CurrentTestFailures() {
    static int count = 0;
    return count;
}

inline std::string& FilterPattern() {
    static std::string pat = "*";
    return pat;
}

inline bool WildcardMatch(const char* pat, const char* str) {
    if (*pat == '\0') return *str == '\0';
    if (*pat == '*') {
        return WildcardMatch(pat + 1, str) || (*str != '\0' && WildcardMatch(pat, str + 1));
    }
    if (*pat == '?' || *pat == *str) {
        return *str != '\0' && WildcardMatch(pat + 1, str + 1);
    }
    return false;
}

inline bool MatchesFilter(const std::string& full_name) {
    const std::string& filter = FilterPattern();
    if (filter.empty() || filter == "*") return true;
    std::size_t start = 0;
    while (start <= filter.size()) {
        std::size_t colon = filter.find(':', start);
        std::string sub = (colon == std::string::npos) ? filter.substr(start) : filter.substr(start, colon - start);
        if (!sub.empty() && WildcardMatch(sub.c_str(), full_name.c_str())) {
            return true;
        }
        if (colon == std::string::npos) break;
        start = colon + 1;
    }
    return false;
}

inline void InitGoogleTest(int* argc, char** argv) {
    if (!argc || !argv) return;
    const char prefix[] = "--gtest_filter=";
    const std::size_t prefix_len = sizeof(prefix) - 1;
    int write_idx = 1;
    for (int i = 1; i < *argc; ++i) {
        if (std::strncmp(argv[i], prefix, prefix_len) == 0) {
            FilterPattern() = argv[i] + prefix_len;
        } else {
            argv[write_idx++] = argv[i];
        }
    }
    *argc = write_idx;
}

class MessageStream {
public:
    MessageStream(bool failed, bool fatal, const char* file, int line, const std::string& msg)
        : failed_(failed), fatal_(fatal) {
        if (failed_) {
            ++CurrentTestFailures();
            ss_ << file << ":" << line << ": Failure\n  " << msg;
        }
    }

    MessageStream(MessageStream&& other) noexcept
        : failed_(other.failed_), fatal_(other.fatal_), ss_(std::move(other.ss_)) {
        other.failed_ = false;
        other.fatal_ = false;
    }

    ~MessageStream() noexcept(false) {
        if (failed_) {
            std::cout << ss_.str() << std::endl;
            if (fatal_) {
                throw FatalAssertionFailure();
            }
        }
    }

    template <typename T>
    MessageStream& operator<<(const T& val) {
        if (failed_) {
            if (!has_user_msg_) {
                ss_ << "\n  Message: ";
                has_user_msg_ = true;
            }
            ss_ << val;
        }
        return *this;
    }

    MessageStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        if (failed_) {
            if (!has_user_msg_) {
                ss_ << "\n  Message: ";
                has_user_msg_ = true;
            }
            manip(ss_);
        }
        return *this;
    }

private:
    bool failed_;
    bool fatal_;
    bool has_user_msg_ = false;
    std::ostringstream ss_;
};

template <typename V>
inline void PrintTestValue(std::ostream& os, const V& v) {
    os << v;
}

inline void PrintTestValue(std::ostream& os, std::nullptr_t) {
    os << "nullptr";
}

inline void PrintTestValue(std::ostream& os, const unsigned char* p) {
    os << static_cast<const void*>(p);
}

inline void PrintTestValue(std::ostream& os, unsigned char* p) {
    os << static_cast<const void*>(p);
}

template <typename E>
inline void PrintTestValue(std::ostream& os, const std::vector<E>& vec) {
    os << "[";
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) os << ", ";
        if (i >= 8) { os << "... (" << vec.size() << " items)"; break; }
        PrintTestValue(os, vec[i]);
    }
    os << "]";
}

template <typename A, typename B>
MessageStream CheckCmp(bool ok, bool fatal, const char* file, int line,
                       const char* expr_a, const char* op, const char* expr_b,
                       const A& a, const B& b) {
    if (ok) return MessageStream(false, false, file, line, "");
    std::ostringstream msg;
    msg << "Expected: (" << expr_a << ") " << op << " (" << expr_b << "), actual: ";
    PrintTestValue(msg, a);
    msg << " vs ";
    PrintTestValue(msg, b);
    return MessageStream(true, fatal, file, line, msg.str());
}

inline MessageStream CheckNear(bool fatal, const char* file, int line,
                               const char* expr_a, const char* expr_b, const char* expr_tol,
                               double a, double b, double tol) {
    const double diff = std::fabs(a - b);
    const bool ok = std::isfinite(a) && std::isfinite(b) && (diff <= tol);
    if (ok) return MessageStream(false, false, file, line, "");
    std::ostringstream msg;
    msg << "The difference between " << expr_a << " (" << a << ") and " << expr_b << " (" << b
        << ") is " << diff << ", which exceeds " << expr_tol << " (" << tol << ").";
    return MessageStream(true, fatal, file, line, msg.str());
}

inline MessageStream CheckStrEq(bool expect_eq, bool fatal, const char* file, int line,
                                const char* expr_a, const char* expr_b,
                                const char* a, const char* b) {
    const bool both_null = (a == nullptr && b == nullptr);
    const bool eq = both_null || (a != nullptr && b != nullptr && std::strcmp(a, b) == 0);
    const bool ok = expect_eq ? eq : !eq;
    if (ok) return MessageStream(false, false, file, line, "");
    std::ostringstream msg;
    msg << "Expected C-strings (" << expr_a << ") " << (expect_eq ? "==" : "!=") << " (" << expr_b
        << "), actual: \"" << (a ? a : "(null)") << "\" vs \"" << (b ? b : "(null)") << "\"";
    return MessageStream(true, fatal, file, line, msg.str());
}

} // namespace testing

inline int RUN_ALL_TESTS() {
    int total_ran = 0;
    int failed_tests = 0;
    std::vector<std::string> failed_names;
    const auto t_suite_start = std::chrono::steady_clock::now();

    for (const auto& tc : ::testing::GetRegistry()) {
        const std::string full_name = tc.suite + "." + tc.name;
        if (!::testing::MatchesFilter(full_name)) continue;

        ++total_ran;
        ::testing::CurrentTestFailures() = 0;
        std::cout << "[ RUN      ] " << full_name << std::endl;
        const auto t0 = std::chrono::steady_clock::now();
        try {
            tc.run();
        } catch (const ::testing::FatalAssertionFailure&) {
            // Recorded in CurrentTestFailures
        } catch (const std::exception& ex) {
            ++::testing::CurrentTestFailures();
            std::cout << "  Unhandled std::exception: " << ex.what() << std::endl;
        } catch (...) {
            ++::testing::CurrentTestFailures();
            std::cout << "  Unhandled unknown exception" << std::endl;
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();

        if (::testing::CurrentTestFailures() == 0) {
            std::cout << "[       OK ] " << full_name << " (" << ms << " ms)" << std::endl;
        } else {
            ++failed_tests;
            failed_names.push_back(full_name);
            std::cout << "[  FAILED  ] " << full_name << " (" << ms << " ms)" << std::endl;
        }
    }

    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_suite_start).count();
    std::cout << "[==========] " << total_ran << " test(s) ran (" << total_ms << " ms total)." << std::endl;
    std::cout << "[  PASSED  ] " << (total_ran - failed_tests) << " test(s)." << std::endl;
    if (failed_tests > 0) {
        std::cout << "[  FAILED  ] " << failed_tests << " test(s), listed below:" << std::endl;
        for (const auto& name : failed_names) {
            std::cout << "[  FAILED  ] " << name << std::endl;
        }
        return 1;
    }
    return 0;
}

#define GCTA_TEST_CLASS_NAME_(suite, name) GctaTest_##suite##_##name

#define TEST(suite, name) \
    class GCTA_TEST_CLASS_NAME_(suite, name) : public ::testing::Test { \
    public: \
        void TestBody() override; \
    }; \
    static bool gcta_reg_##suite##_##name = ([]() { \
        ::testing::GetRegistry().push_back({#suite, #name, []() { \
            GCTA_TEST_CLASS_NAME_(suite, name) inst; \
            inst.SetUp(); \
            try { inst.TestBody(); } catch (...) { inst.TearDown(); throw; } \
            inst.TearDown(); \
        }}); \
        return true; \
    })(); \
    void GCTA_TEST_CLASS_NAME_(suite, name)::TestBody()

#define TEST_F(fixture, name) \
    class GCTA_TEST_CLASS_NAME_(fixture, name) : public fixture { \
    public: \
        void TestBody() override; \
        void RunFixture() { \
            this->SetUp(); \
            try { this->TestBody(); } catch (...) { this->TearDown(); throw; } \
            this->TearDown(); \
        } \
    }; \
    static bool gcta_reg_##fixture##_##name = ([]() { \
        ::testing::GetRegistry().push_back({#fixture, #name, []() { \
            GCTA_TEST_CLASS_NAME_(fixture, name) inst; \
            inst.RunFixture(); \
        }}); \
        return true; \
    })(); \
    void GCTA_TEST_CLASS_NAME_(fixture, name)::TestBody()

#define EXPECT_TRUE(cond) \
    ::testing::MessageStream(!(cond), false, __FILE__, __LINE__, std::string("Expected true: ") + #cond)
#define ASSERT_TRUE(cond) \
    ::testing::MessageStream(!(cond), true, __FILE__, __LINE__, std::string("Expected true: ") + #cond)

#define EXPECT_FALSE(cond) \
    ::testing::MessageStream(!!(cond), false, __FILE__, __LINE__, std::string("Expected false: ") + #cond)
#define ASSERT_FALSE(cond) \
    ::testing::MessageStream(!!(cond), true, __FILE__, __LINE__, std::string("Expected false: ") + #cond)

#define EXPECT_EQ(a, b) ::testing::CheckCmp(((a) == (b)), false, __FILE__, __LINE__, #a, "==", #b, (a), (b))
#define ASSERT_EQ(a, b) ::testing::CheckCmp(((a) == (b)), true, __FILE__, __LINE__, #a, "==", #b, (a), (b))

#define EXPECT_NE(a, b) ::testing::CheckCmp(((a) != (b)), false, __FILE__, __LINE__, #a, "!=", #b, (a), (b))
#define ASSERT_NE(a, b) ::testing::CheckCmp(((a) != (b)), true, __FILE__, __LINE__, #a, "!=", #b, (a), (b))

#define EXPECT_LT(a, b) ::testing::CheckCmp(((a) < (b)), false, __FILE__, __LINE__, #a, "<", #b, (a), (b))
#define ASSERT_LT(a, b) ::testing::CheckCmp(((a) < (b)), true, __FILE__, __LINE__, #a, "<", #b, (a), (b))

#define EXPECT_LE(a, b) ::testing::CheckCmp(((a) <= (b)), false, __FILE__, __LINE__, #a, "<=", #b, (a), (b))
#define ASSERT_LE(a, b) ::testing::CheckCmp(((a) <= (b)), true, __FILE__, __LINE__, #a, "<=", #b, (a), (b))

#define EXPECT_GT(a, b) ::testing::CheckCmp(((a) > (b)), false, __FILE__, __LINE__, #a, ">", #b, (a), (b))
#define ASSERT_GT(a, b) ::testing::CheckCmp(((a) > (b)), true, __FILE__, __LINE__, #a, ">", #b, (a), (b))

#define EXPECT_GE(a, b) ::testing::CheckCmp(((a) >= (b)), false, __FILE__, __LINE__, #a, ">=", #b, (a), (b))
#define ASSERT_GE(a, b) ::testing::CheckCmp(((a) >= (b)), true, __FILE__, __LINE__, #a, ">=", #b, (a), (b))

#define EXPECT_NEAR(a, b, tol) ::testing::CheckNear(false, __FILE__, __LINE__, #a, #b, #tol, (a), (b), (tol))
#define ASSERT_NEAR(a, b, tol) ::testing::CheckNear(true, __FILE__, __LINE__, #a, #b, #tol, (a), (b), (tol))

#define EXPECT_DOUBLE_EQ(a, b) \
    ::testing::CheckNear(false, __FILE__, __LINE__, #a, #b, "1e-12*scale", (a), (b), \
                         1e-12 * std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b))))
#define EXPECT_FLOAT_EQ(a, b) \
    ::testing::CheckNear(false, __FILE__, __LINE__, #a, #b, "1e-5f*scale", (a), (b), \
                         1e-5 * std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b))))

#define EXPECT_STREQ(a, b) ::testing::CheckStrEq(true, false, __FILE__, __LINE__, #a, #b, (a), (b))
#define ASSERT_STREQ(a, b) ::testing::CheckStrEq(true, true, __FILE__, __LINE__, #a, #b, (a), (b))
#define EXPECT_STRNE(a, b) ::testing::CheckStrEq(false, false, __FILE__, __LINE__, #a, #b, (a), (b))

#define EXPECT_THROW(stmt, exc_type) \
    do { \
        bool caught_expected_ = false; \
        try { stmt; } catch (const exc_type&) { caught_expected_ = true; } catch (...) {} \
        ::testing::MessageStream(!caught_expected_, false, __FILE__, __LINE__, \
            std::string("Expected ") + #stmt + " to throw " + #exc_type); \
    } while (0)

#define EXPECT_ANY_THROW(stmt) \
    do { \
        bool caught_any_ = false; \
        try { stmt; } catch (...) { caught_any_ = true; } \
        ::testing::MessageStream(!caught_any_, false, __FILE__, __LINE__, \
            std::string("Expected ") + #stmt + " to throw an exception"); \
    } while (0)

#endif // GCTA_LITE_GTEST_H
