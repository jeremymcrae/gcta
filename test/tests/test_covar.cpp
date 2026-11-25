#include <gtest/gtest.h>
#include "Covar.h"
#include "OptionIO.h"
#include "test_helpers.h"
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>

TEST(OptionIOTest, ReadsTabularListsAndNumericOptionsCleanly) {
    gcta_test::TempDir tmp("gcta_optio_");
    const std::string tbl_path = tmp.file("table.txt");
    {
        std::ofstream out(tbl_path);
        out << "#FID\tIID\tVAL\n";
        out << "F1\tI1\t12.5\n";
        out << "F2\tI2\t-3.25\n";
    }

    EXPECT_TRUE(checkFileReadable(tbl_path));
    EXPECT_FALSE(checkFileReadable(tmp.file("does_not_exist.txt")));

    std::vector<std::string> head;
    int n_header = 0;
    std::map<int, std::vector<std::string>> cols;
    ASSERT_TRUE(readTxtList(tbl_path, 3, head, n_header, cols));
    ASSERT_EQ(head.size(), 3u);
    EXPECT_EQ(head[0], "#FID");
    EXPECT_EQ(head[2], "VAL");
    ASSERT_EQ(cols[0].size(), 2u);
    EXPECT_EQ(cols[1][1], "I2");
    EXPECT_EQ(cols[2][0], "12.5");

    std::map<std::string, std::vector<std::string>> opts_in = {
        {"--maf", {"0.025"}}
    };
    std::map<std::string, double> opts_out;
    addOneValOption<double>("maf", "--maf", opts_in, opts_out, 0.01, 0.0, 0.5);
    EXPECT_NEAR(opts_out["maf"], 0.025, 1e-12);

    addOneValOption<double>("default_opt", "--missing-flag", opts_in, opts_out, 0.15, 0.0, 1.0);
    EXPECT_NEAR(opts_out["default_opt"], 0.15, 1e-12);
}

TEST(CovarTest, AlignsQuantitativeAndCategoricalCovariatesAndDropsMissingRows) {
    gcta_test::TempDir tmp("gcta_covar_");
    const std::string qcov_path = tmp.file("test.qcov");
    const std::string cov_path = tmp.file("test.cov");
    const std::string env_path = tmp.file("test.env");

    {
        std::ofstream fq(qcov_path);
        fq << "F1\tI1\t20.0\t1.5\n";
        fq << "F2\tI2\t30.0\t2.5\n";
        fq << "F3\tI3\tNA\t3.5\n";   // missing quantitative covariate -> excluded
        fq << "F4\tI4\t40.0\t4.5\n";
        fq << "F5\tI5\t50.0\t5.5\n";
    }
    {
        // 3 levels {"BatchA", "BatchB", "BatchC"} -> 2 indicator columns after dropping reference level
        std::ofstream fc(cov_path);
        fc << "F5\tI5\tBatchC\n";
        fc << "F4\tI4\tBatchB\n";
        fc << "F3\tI3\tBatchA\n";
        fc << "F2\tI2\tBatchB\n";
        fc << "F1\tI1\tBatchA\n";
    }
    {
        std::ofstream fe(env_path);
        fe << "F1\tI1\t0.0\n";
        fe << "F2\tI2\t1.0\n";
        fe << "F4\tI4\t0.0\n";
        fe << "F5\tI5\t1.0\n";
    }

    const int rc = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("covar.log"));
        std::map<std::string, std::vector<std::string>> opts = {
            {"--qcovar", {qcov_path}},
            {"--covar", {cov_path}},
            {"--envir", {env_path}}
        };
        Covar::registerOption(opts);
        Covar cov;
        EXPECT_TRUE(cov.hasCovar());
        EXPECT_TRUE(cov.hasEnvir());

        // Query samples in custom order; F3\tI3 had NA in qcovar so 4 samples should remain
        const std::vector<std::string> query_ids = {
            "F4\tI4", "F3\tI3", "F1\tI1", "F5\tI5", "F2\tI2"
        };
        std::vector<double> X;
        std::vector<uint32_t> keep_idx;
        ASSERT_TRUE(cov.getCovarX(query_ids, X, keep_idx));
        ASSERT_EQ(keep_idx.size(), 4u);
        // Kept indices in query_ids should be {0 (F4), 2 (F1), 3 (F5), 4 (F2)}
        EXPECT_EQ(keep_idx[0], 0u);
        EXPECT_EQ(keep_idx[1], 2u);
        EXPECT_EQ(keep_idx[2], 3u);
        EXPECT_EQ(keep_idx[3], 4u);

        // 2 qcovars + (3 - 1) categorical dummy columns + 1 envir column = 5 columns -> 5 * 4 = 20 entries
        ASSERT_EQ(X.size(), 20u);
        // Column 0 is first qcovar in query order: F4=40, F1=20, F5=50, F2=30
        EXPECT_NEAR(X[0], 40.0, 1e-12);
        EXPECT_NEAR(X[1], 20.0, 1e-12);
        EXPECT_NEAR(X[2], 50.0, 1e-12);
        EXPECT_NEAR(X[3], 30.0, 1e-12);

        std::vector<double> E;
        std::vector<uint32_t> env_keep;
        ASSERT_TRUE(cov.getEnvirX(query_ids, E, env_keep));
        ASSERT_EQ(E.size(), 4u);
        EXPECT_NEAR(E[0], 0.0, 1e-12); // F4
        EXPECT_NEAR(E[1], 0.0, 1e-12); // F1
        EXPECT_NEAR(E[2], 1.0, 1e-12); // F5
        EXPECT_NEAR(E[3], 1.0, 1e-12); // F2
        LOGGER.close();
    });
    EXPECT_EQ(rc, 0);
}

TEST(CovarTest, RankedCovariateWithDiscreteCovarEncodesOrthogonalContrasts) {
    gcta_test::TempDir tmp("gcta_rcovar_both_");
    const std::string cov_path = tmp.file("test.cov");
    const std::string rcov_path = tmp.file("test.rcov");
    {
        std::ofstream fc(cov_path);
        fc << "F1\tI1\tB0\nF2\tI2\tB1\nF3\tI3\tB0\nF4\tI4\tB1\n";
    }
    {
        std::ofstream fr(rcov_path);
        fr << "F1\tI1\tL1\nF2\tI2\tL2\nF3\tI3\tL3\nF4\tI4\tL1\n";
    }

    const int rc = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("rcovar_both.log"));
        std::map<std::string, std::vector<std::string>> opts = {
            {"--covar", {cov_path}},
            {"--rcovar", {rcov_path}}
        };
        Covar::registerOption(opts);
        Covar cov;
        const std::vector<std::string> ids = {"F1\tI1", "F2\tI2", "F3\tI3", "F4\tI4"};
        std::vector<double> X;
        std::vector<uint32_t> keep_idx;
        ASSERT_TRUE(cov.getCovarX(ids, X, keep_idx));
        ASSERT_EQ(keep_idx.size(), 4u);
        // 1 covar dummy + 2 rcovar Helmert contrasts = 3 columns -> 12 entries
        ASSERT_EQ(X.size(), 12u);
        // F1 (row 0) and F4 (row 3) share rcovar level L1 -> identical rcovar contrast entries
        EXPECT_NEAR(X[4], X[7], 1e-12);
        EXPECT_NEAR(X[8], X[11], 1e-12);
        LOGGER.close();
    });
    EXPECT_EQ(rc, 0);
}

// Regression test for upstream bug in Covar::setCovarMapping (src/Covar.cpp:404):
// Checking this->covar.size() == 0 before switching pcovar = &this->rcovar caused
// --rcovar without --covar to leave labels_rcovar_mapping empty and segfault in getCovarX.
TEST(CovarBugRegressionTest, RankedCovariateWithoutDiscreteCovarDoesNotSegfault) {
    gcta_test::TempDir tmp("gcta_rcovar_only_");
    const std::string rcov_path = tmp.file("test.rcov");
    {
        std::ofstream fr(rcov_path);
        fr << "F1\tI1\tL1\n";
        fr << "F2\tI2\tL2\n";
        fr << "F3\tI3\tL3\n";
        fr << "F4\tI4\tL1\n";
    }

    const int rc = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("rcovar.log"));
        std::map<std::string, std::vector<std::string>> opts = {
            {"--rcovar", {rcov_path}}
        };
        Covar::registerOption(opts);
        Covar cov;
        const std::vector<std::string> ids = {"F1\tI1", "F2\tI2", "F3\tI3", "F4\tI4"};
        std::vector<double> X;
        std::vector<uint32_t> keep_idx;
        ASSERT_TRUE(cov.getCovarX(ids, X, keep_idx));
        ASSERT_EQ(keep_idx.size(), 4u);
        ASSERT_EQ(X.size(), 8u);
        EXPECT_NEAR(X[0], X[3], 1e-12);
        EXPECT_NEAR(X[4], X[7], 1e-12);
        LOGGER.close();
    });
    EXPECT_EQ(rc, 0);
}
