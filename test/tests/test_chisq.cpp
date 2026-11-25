#include <gtest/gtest.h>
#include "StatLib.h"
#include "StatFunc.h"
#include "CommFunc.h"
#include "StrFunc.h"
#include "utils.hpp"
#include "acat.hpp"
#include "test_helpers.h"
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

TEST(StatLibTest, ChiSquaredTailAndQuantileRoundTrip) {
    // 1-df chi-square critical points and deep right tail (x = 300 -> 3.294362e-67)
    EXPECT_NEAR(StatLib::pchisqd1(3.841458820694124), 0.05, 1e-12);
    EXPECT_NEAR(StatLib::pchisqd1(6.6348966010212145), 0.01, 1e-12);
    const double p300 = StatLib::pchisqd1(300.0);
    EXPECT_GT(p300, 3.2943e-67);
    EXPECT_LT(p300, 3.2945e-67);

    // 2-df chi-square equals exp(-x/2)
    for (double x : {0.5, 2.0, 10.0, 40.0}) {
        EXPECT_NEAR(StatLib::pchisqd2(x), std::exp(-0.5 * x), 1e-14 * std::exp(-0.5 * x));
    }

    // Quantile <-> survival round-trip
    for (double p : {0.5, 0.05, 1e-4, 1e-10, 5e-8}) {
        const double q = StatLib::qchisqd1(p);
        EXPECT_NEAR(StatLib::pchisqd1(q), p, 1e-11 * p);
    }

    // Invalid inputs return NaN / +Inf cleanly
    EXPECT_TRUE(std::isnan(StatLib::pchisqd1(-1.0)));
    EXPECT_TRUE(std::isinf(StatLib::qchisqd1(0.0)));
    EXPECT_TRUE(std::isnan(StatLib::qchisqd1(1.5)));
}

TEST(StatLibTest, NormalTailAndBetaMafWeights) {
    EXPECT_NEAR(StatLib::pnorm(0.0, false), 0.5, 1e-14);
    EXPECT_NEAR(StatLib::pnorm(0.0, true), 0.5, 1e-14);
    EXPECT_NEAR(StatLib::pnorm(1.959963984540054, false), 0.025, 1e-12);
    EXPECT_NEAR(StatLib::pnorm(-1.959963984540054, true), 0.025, 1e-12);

    Eigen::VectorXd maf(3);
    maf << 0.001, 0.01, 0.20;
    const Eigen::VectorXd w = StatLib::weightBetaMAF(maf, 1.0, 25.0);
    // Beta(1, 25) pdf = 25 * (1 - p)^24
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(w[i], 25.0 * std::pow(1.0 - maf[i], 24.0), 1e-11);
    }
    EXPECT_GT(w[0], w[1]);
    EXPECT_GT(w[1], w[2]);
}

TEST(StatLibTest, RankContrastProducesOrthonormalColumns) {
    const int n = 5;
    std::vector<double> Z(n * n, 0.0);
    ASSERT_TRUE(StatLib::rankContrast(n, Z.data()));

    // Columns 0..n-1 of Z (column-major) must be mutually orthonormal; columns 1..n-1 sum to 0
    for (int col = 0; col < n; ++col) {
        double norm2 = 0.0;
        double col_sum = 0.0;
        for (int row = 0; row < n; ++row) {
            const double v = Z[col * n + row];
            norm2 += v * v;
            col_sum += v;
        }
        EXPECT_NEAR(norm2, 1.0, 1e-12);
        if (col >= 1) {
            EXPECT_NEAR(col_sum, 0.0, 1e-12);
        }
        for (int col2 = col + 1; col2 < n; ++col2) {
            double dot = 0.0;
            for (int row = 0; row < n; ++row) {
                dot += Z[col * n + row] * Z[col2 * n + row];
            }
            EXPECT_NEAR(dot, 0.0, 1e-12);
        }
    }
}

TEST(StatFuncV1Test, DistributionsQuantilesFdrAndChiSqMixture) {
    EXPECT_NEAR(StatFunc::chi_prob(1.0, 3.841458820694124), 0.05, 1e-6);
    EXPECT_NEAR(StatFunc::chi_val(1.0, 0.05), 3.841458820694124, 1e-4);

    const double t975 = StatFunc::t_val(20.0, 0.025);
    EXPECT_NEAR(StatFunc::t_prob(20.0, t975, true), 0.05, 1e-5);
    EXPECT_NEAR(StatFunc::t_prob(20.0, t975, false), 0.025, 1e-5);

    const double f95 = StatFunc::F_val(3.0, 30.0, 0.05);
    EXPECT_NEAR(StatFunc::F_prob(3.0, 30.0, f95), 0.05, 1e-5);

    // Benjamini-Hochberg adjusted p-values are monotone and bounded in [0, 1]
    const std::vector<double> raw_p = {0.001, 0.01, 0.04, 0.20, 0.80};
    const std::vector<double> bh_q = StatFunc::ControlFDR_BH(raw_p);
    ASSERT_EQ(bh_q.size(), raw_p.size());
    EXPECT_NEAR(bh_q[0], 0.005, 1e-12);
    EXPECT_NEAR(bh_q[1], 0.025, 1e-12);
    for (std::size_t i = 1; i < bh_q.size(); ++i) {
        EXPECT_GE(bh_q[i], bh_q[i - 1] - 1e-14);
        EXPECT_LE(bh_q[i], 1.0);
    }

    // Sum of two unit-weight 1-df chi-squares equals a 2-df chi-square: P(Q > x) = exp(-x/2)
    Eigen::VectorXd lam(2);
    lam << 1.0, 1.0;
    const double x = 8.0;
    const double exact_p = std::exp(-0.5 * x);
    EXPECT_NEAR(StatFunc::pchisqsum(x, lam), exact_p, 2e-3);
    EXPECT_NEAR(StatFunc::psadd(x, lam), exact_p, 2e-3);
    EXPECT_NEAR(StatFunc::psatt(x, lam), exact_p, 1e-6);
}

TEST(CommStrAndUtilsTest, SummaryStatsStringOpsAndIndexAlignment) {
    const std::vector<double> vals = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    EXPECT_NEAR(CommFunc::sum(vals), 40.0, 1e-12);
    EXPECT_NEAR(CommFunc::mean(vals), 5.0, 1e-12);
    EXPECT_NEAR(CommFunc::median(vals), 4.5, 1e-12);
    EXPECT_NEAR(CommFunc::var(vals), 32.0 / 7.0, 1e-12);

    const std::vector<double> odd_vals = {10.0, 1.0, 5.0};
    EXPECT_NEAR(CommFunc::median(odd_vals), 5.0, 1e-12);

    std::vector<std::string> toks;
    EXPECT_EQ(StrFunc::split_string("chr1\trs100\t12345 A G", toks), 5);
    EXPECT_EQ(toks[1], "rs100");
    EXPECT_TRUE(StrFunc::i_compare("FastGWA", "fastgwa"));

    const std::vector<std::string> a = {"S3", "S1", "S4", "S2"};
    const std::vector<std::string> b = {"S2", "S5", "S1"};
    std::vector<int> match_idx;
    StrFunc::match(b, a, match_idx);
    ASSERT_EQ(match_idx.size(), 3u);
    EXPECT_EQ(match_idx[0], 3);
    EXPECT_EQ(match_idx[1], -9);
    EXPECT_EQ(match_idx[2], 1);

    // utils.hpp helpers
    EXPECT_TRUE(hasVectorDuplicate(std::vector<int>{3, 1, 4, 1, 5}));
    EXPECT_FALSE(hasVectorDuplicate(std::vector<int>{3, 1, 4, 2, 5}));

    std::vector<uint32_t> k1, k2;
    vector_commonIndex_sorted1(a, b, k1, k2);
    ASSERT_EQ(k1.size(), 2u);
    EXPECT_EQ(k1[0], 1u); // S1 in a
    EXPECT_EQ(k2[0], 2u); // S1 in b
    EXPECT_EQ(k1[1], 3u); // S2 in a
    EXPECT_EQ(k2[1], 0u); // S2 in b

    std::vector<std::vector<int>> grid = {{1, 2}, {10, 20, 30}};
    std::vector<std::vector<int>> combos;
    permute_vector(grid, combos);
    EXPECT_EQ(combos.size(), 6u);

    EXPECT_EQ(getFileName("/tmp/dir/sample.fam"), "sample.fam");
    EXPECT_EQ(to_string_precision(3.1415926, 3), "3.142");
}

// Regression test for upstream bug in joinPath (src/utils.cpp:119):
// joinPath defined sep = "/" but returned dir + path without inserting sep.
TEST(UtilsBugRegressionTest, JoinPathInsertsDirectorySeparator) {
    EXPECT_EQ(joinPath("", "sample.fam"), "sample.fam");
    EXPECT_EQ(joinPath("/tmp/dir", "sample.fam"), "/tmp/dir/sample.fam");
}

TEST(AcatTest, CombinesRareVariantPvaluesPerGene) {
    gcta_test::TempDir tmp("gcta_acat_");
    const std::string gene_file = tmp.file("genes.txt");
    const std::string snp_file = tmp.file("snps.fastGWA");
    const std::string out_file = tmp.file("acat_out.txt");

    {
        std::ofstream gf(gene_file);
        gf << "1\t1000\t5000\tGENE_SIG\n";
        gf << "1\t6000\t9000\tGENE_NULL\n";
    }
    {
        // 14-column binary .fastGWA summary layout:
        // 0:CHR 1:SNP 2:POS 3:A1 4:A2 5:N 6:AF1 7:T 8:SE_T 9:P_noSPA 10:BETA 11:SE 12:P 13:CONVERGE
        std::ofstream sf(snp_file);
        sf << "CHR\tSNP\tPOS\tA1\tA2\tN\tAF1\tT\tSE_T\tP_noSPA\tBETA\tSE\tP\tCONVERGE\n";
        sf << "1\trs1\t1500\tA\tG\t5000\t0.005\t4.5\t1.0\t1e-5\t0.8\t0.18\t1.0e-6\t1\n";
        sf << "1\trs2\t2500\tA\tG\t5000\t0.004\t0.2\t1.0\t0.84\t0.03\t0.15\t0.80\t1\n";
        sf << "1\trs3\t7000\tA\tG\t5000\t0.005\t0.1\t1.0\t0.92\t0.02\t0.15\t0.75\t1\n";
        sf << "1\trs4\t8000\tA\tG\t5000\t0.006\t-0.3\t1.0\t0.76\t-0.04\t0.15\t0.65\t1\n";
    }

    ASSERT_EQ(acat_func(gene_file.c_str(), snp_file.c_str(), 0.01, 10, 0, out_file.c_str()), 0);
    const auto lines = gcta_test::ReadLines(out_file);
    ASSERT_EQ(lines.size(), 3u); // header + 2 genes
    const auto row1 = gcta_test::SplitTabsOrSpaces(lines[1]);
    const auto row2 = gcta_test::SplitTabsOrSpaces(lines[2]);
    ASSERT_GE(row1.size(), 6u);
    ASSERT_GE(row2.size(), 6u);
    EXPECT_EQ(row1[0], "1");
    EXPECT_EQ(row1[1], "GENE_SIG");
    EXPECT_EQ(row2[0], "1");
    EXPECT_EQ(row2[1], "GENE_NULL");
    const double p_sig = std::stod(row1.back());
    const double p_null = std::stod(row2.back());
    EXPECT_LT(p_sig, 1e-5);
    EXPECT_GT(p_null, 0.5);
}
