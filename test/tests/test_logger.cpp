#include <gtest/gtest.h>
#include "Logger.h"
#include "test_helpers.h"
#include <chrono>
#include <fstream>
#include <string>
#include <thread>

TEST(LoggerTest, WritesInfoAndWarnToLogFileButNotTerminalOnlyMessages) {
    gcta_test::TempDir tmp("gcta_logger_");
    const std::string log_path = tmp.file("unit.log");

    const int rc = gcta_test::RunIsolated([&]() {
        Logger* log = Logger::GetLogger();
        log->open(log_path);
        log->i(0, "Info message alpha");
        log->i(1, "Tag", "Indented message beta");
        log->w(2, "Warning message gamma");
        log->l(0, "FileOnlyLine delta");
        log->p(1, "ProgressOnlyLine epsilon");
        log->m(1, "TerminalOnlyLine zeta");

        log->ts("timer_check");
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        const float elapsed = log->tp("timer_check");
        EXPECT_GT(elapsed, 0.005f);
        EXPECT_LT(elapsed, 5.0f);
        log->flush();
        log->close();
    });
    ASSERT_EQ(rc, 0);

    std::ifstream in(log_path);
    ASSERT_TRUE(in.good());
    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("Info message alpha"), std::string::npos);
    EXPECT_NE(content.find("Indented message beta"), std::string::npos);
    EXPECT_NE(content.find("Warning message gamma"), std::string::npos);
    EXPECT_NE(content.find("FileOnlyLine delta"), std::string::npos);
    // p() is progress-only (terminal) and must not be written to the log file;
    // m() and l() write prompt/file lines into the log file.
    EXPECT_EQ(content.find("ProgressOnlyLine epsilon"), std::string::npos);
    EXPECT_NE(content.find("TerminalOnlyLine zeta"), std::string::npos);
}

TEST(LoggerTest, ErrorMethodLogsAndExitsWithFailureCode) {
    gcta_test::TempDir tmp("gcta_logger_err_");
    const std::string log_path = tmp.file("err.log");

    const int rc = gcta_test::RunIsolated([&]() {
        LOGGER.open(log_path);
        LOGGER.e(0, "Intentional fatal error for test");
    });
    EXPECT_EQ(rc, EXIT_FAILURE);

    std::ifstream in(log_path);
    ASSERT_TRUE(in.good());
    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("Intentional fatal error for test"), std::string::npos);
}
