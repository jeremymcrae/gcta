#include <gtest/gtest.h>
#include "cpu.h"
#include "Logger.h"
#include "Geno.h"
#include "Pheno.h"
#include "Marker.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mutex>
#include <omp.h>
#include <vector>
#define private public
#include "FastFAM.h"
#undef private
#include "test_helpers.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <random>
#include <string>

namespace {

struct AssocDataset {
    int N = 0;
    int M = 0;
    std::string prefix;
    std::string qphen_path;
    std::string bphen_path;
    std::string qcov_path;
    std::string cov_path;
    std::string set_path;
};

AssocDataset CreateFastGwaDataset(const gcta_test::TempDir& tmp, int N = 200, int M = 2100) {
    AssocDataset ds;
    ds.N = N;
    ds.M = M;
    ds.prefix = tmp.file("fgwa");
    ds.qphen_path = tmp.file("fgwa.qphen");
    ds.bphen_path = tmp.file("fgwa.bphen");
    ds.qcov_path = tmp.file("fgwa.qcov");
    ds.cov_path = tmp.file("fgwa.cov");
    ds.set_path = tmp.file("fgwa.set");

    std::mt19937 rng(424242);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    std::vector<gcta_test::SampleSpec> samples(N);
    for (int i = 0; i < N; ++i) {
        samples[i] = {"F" + std::to_string(i + 1), "I" + std::to_string(i + 1), (i % 2) + 1, -9.0};
    }

    // Sparse family relatedness: first 80 samples form 40 sibling pairs (r = 0.5), remaining 120 are singletons
    {
        std::ofstream fid_out(ds.prefix + ".grm.id");
        for (const auto& s : samples) fid_out << s.fid << "\t" << s.iid << "\n";

        std::ofstream sp_out(ds.prefix + ".grm.sp");
        for (int i = 0; i < N; ++i) {
            sp_out << i << "\t" << i << "\t1.050000\n";
        }
        for (int p = 0; p < 40 && (2 * p + 1) < N; ++p) {
            const int a = 2 * p;
            const int b = 2 * p + 1;
            sp_out << b << "\t" << a << "\t0.500000\n";
        }
    }

    // Simulate family random effects consistent with the sparse GRM
    std::vector<double> u_fam(N, 0.0);
    for (int p = 0; p < 40 && (2 * p + 1) < N; ++p) {
        const double shared = std::sqrt(0.5) * norm(rng);
        u_fam[2 * p] = shared + std::sqrt(0.55) * norm(rng);
        u_fam[2 * p + 1] = shared + std::sqrt(0.55) * norm(rng);
    }
    for (int i = 80; i < N; ++i) {
        u_fam[i] = std::sqrt(1.05) * norm(rng);
    }

    std::vector<double> qcov1(N), qcov2(N);
    std::vector<int> batch(N);
    {
        std::ofstream fq(ds.qcov_path), fc(ds.cov_path);
        for (int i = 0; i < N; ++i) {
            qcov1[i] = norm(rng);
            qcov2[i] = norm(rng);
            batch[i] = i % 2;
            fq << samples[i].fid << "\t" << samples[i].iid << "\t" << qcov1[i] << "\t" << qcov2[i] << "\n";
            fc << samples[i].fid << "\t" << samples[i].iid << "\tB" << batch[i] << "\n";
        }
    }

    std::vector<gcta_test::SnpSpec> snps(M);
    std::vector<std::vector<int>> geno(M, std::vector<int>(N, 0));
    for (int m = 0; m < M; ++m) {
    const int chr = (m * 22) / M + 1;
        snps[m] = {std::to_string(chr), "rs" + std::to_string(m + 1),
                   static_cast<uint32_t>((m + 1) * 100), "A", "G"};
        // Make SNPs 10..19 rare variants (MAF ~ 0.025) on chr 1..10 for burden testing; others common
        const double af = (m >= 10 && m < 20) ? 0.03 : (0.15 + 0.30 * ((m % 7) / 6.0));
        int s = 0;
        for (int i = 0; i < N; ++i) {
            const int g = (u01(rng) < af ? 1 : 0) + (u01(rng) < af ? 1 : 0);
            geno[m][i] = g;
            s += g;
        }
        if (s < 3) {
            geno[m][0] = 1;
            geno[m][1] = 1;
            geno[m][2] = 1;
        }
        if (s > 2 * N - 3) {
            geno[m][0] = 0;
            geno[m][1] = 0;
            geno[m][2] = 0;
        }
    }

    // SNP 0 (rs1) is a strong causal common variant; SNPs 10..19 contribute to rare burden gene
    {
        std::ofstream fqp(ds.qphen_path), fbp(ds.bphen_path);
        for (int i = 0; i < N; ++i) {
            double rare_sum = 0.0;
            for (int m = 10; m < 20; ++m) rare_sum += geno[m][i];
            const double eta = 0.45 * qcov1[i] - 0.30 * qcov2[i] + 0.35 * batch[i]
                             + 1.35 * (geno[0][i] - 0.8) + 0.85 * rare_sum + 0.85 * u_fam[i];
            const double y_q = eta + 0.8 * norm(rng);
            const double prob = 1.0 / (1.0 + std::exp(-(eta - 0.4)));
            const int y_b = (u01(rng) < prob) ? 1 : 0;
            fqp << samples[i].fid << "\t" << samples[i].iid << "\t" << y_q << "\n";
            fbp << samples[i].fid << "\t" << samples[i].iid << "\t" << y_b << "\n";
        }
    }

    // Gene set file for --burden --set-list
    {
        std::ofstream fs(ds.set_path);
        fs << "gene\tchr\tstart\tend\tstrand\n";
        fs << "GENE_CAUSAL\t1\t50\t5000\t+\n";
        fs << "GENE_NULL\t2\t5000\t25000\t+\n";
    }

    gcta_test::WritePlinkBedDataset(ds.prefix, samples, snps, geno);
    gcta_test::WritePlink2PgenDataset(ds.prefix, samples, snps, geno);
    return ds;
}

} // namespace

TEST(FastFamUnitTest, ReadFamLoadsSymmetricSparseGrmAndSubsetsSamples) {
    gcta_test::TempDir tmp("gcta_readfam_");
    const std::string sp_prefix = tmp.file("toy_sp");
    {
        std::ofstream fid(sp_prefix + ".grm.id");
        fid << "F1\tI1\nF2\tI2\nF3\tI3\nF4\tI4\n";

        std::ofstream fsp(sp_prefix + ".grm.sp");
        fsp << "0\t0\t1.02\n";
        fsp << "1\t0\t0.50\n";
        fsp << "1\t1\t0.98\n";
        fsp << "2\t2\t1.01\n";
        fsp << "3\t3\t1.00\n";
    }

    const int rc = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("rf.log"));
        // Query a 3-sample subset in permuted order: {F3, F1, F2}
        const std::vector<std::string> query_ids = {"F3\tI3", "F1\tI1", "F2\tI2"};
        SpMat fam;
        std::vector<uint32_t> remain_idx;
        FastFAM::readFAM(sp_prefix, fam, query_ids, remain_idx);

        ASSERT_EQ(remain_idx.size(), 3u);
        ASSERT_EQ(fam.rows(), 3);
        ASSERT_EQ(fam.cols(), 3);
        // Index 0 = F3, Index 1 = F1, Index 2 = F2
        EXPECT_NEAR(fam.coeff(0, 0), 1.01, 1e-12);
        EXPECT_NEAR(fam.coeff(1, 1), 1.02, 1e-12);
        EXPECT_NEAR(fam.coeff(2, 2), 0.98, 1e-12);
        // Symmetric off-diagonal between F1 (1) and F2 (2)
        EXPECT_NEAR(fam.coeff(1, 2), 0.50, 1e-12);
        EXPECT_NEAR(fam.coeff(2, 1), 0.50, 1e-12);
        EXPECT_NEAR(fam.coeff(0, 1), 0.0, 1e-12);
        LOGGER.close();
    });
    EXPECT_EQ(rc, 0);
}

TEST(FastFamUnitTest, HasemanElstonAndCovarGlmConvergeAccurately) {
    // 1. FastFAM::HEreg on synthetic related pairs
    const int N = 120;
    SpMat fam(N, N);
    std::vector<Eigen::Triplet<double>> trips;
    std::mt19937 rng(777);
    std::normal_distribution<double> norm(0.0, 1.0);
    Eigen::VectorXd y(N);

    for (int p = 0; p < N / 2; ++p) {
        const int a = 2 * p, b = 2 * p + 1;
        trips.emplace_back(a, a, 1.0);
        trips.emplace_back(b, b, 1.0);
        trips.emplace_back(a, b, 0.5);
        trips.emplace_back(b, a, 0.5);
        const double g_shared = std::sqrt(0.3) * norm(rng);
        y[a] = g_shared + std::sqrt(0.7) * norm(rng);
        y[b] = g_shared + std::sqrt(0.7) * norm(rng);
    }
    fam.setFromTriplets(trips.begin(), trips.end());

    bool is_sig = false;
    double pval = 1.0;
    const double h2_est = FastFAM::HEreg(fam, y, is_sig, pval);
    EXPECT_TRUE(std::isfinite(h2_est));
    EXPECT_GT(h2_est, 0.0);
    EXPECT_LT(h2_est, 1.2);
    EXPECT_GE(pval, 0.0);
    EXPECT_LE(pval, 1.0);

    // 2. FastFAM::covarGLM satisfies logistic regression score equations X^T (y - mu) = 0
    const int n_glm = 200;
    Eigen::MatrixXd X(n_glm, 3);
    Eigen::VectorXd y_bin(n_glm);
    Eigen::VectorXd true_b(3);
    true_b << -0.8, 0.9, -0.6;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (int i = 0; i < n_glm; ++i) {
        X(i, 0) = 1.0;
        X(i, 1) = norm(rng);
        X(i, 2) = norm(rng);
        const double p_i = 1.0 / (1.0 + std::exp(-X.row(i).dot(true_b)));
        y_bin[i] = (u01(rng) < p_i) ? 1.0 : 0.0;
    }

    Eigen::VectorXd est_b = Eigen::VectorXd::Zero(3);
    ASSERT_TRUE(FastFAM::covarGLM(y_bin, X, est_b));
    Eigen::VectorXd mu = (1.0 + (-X * est_b).array().exp()).inverse().matrix();
    const double max_grad = (X.transpose() * (y_bin - mu)).cwiseAbs().maxCoeff();
    // Generous tolerance to accommodate solver stopping rules across branches
    EXPECT_LT(max_grad, 1e-4);
}

TEST(FastFamPipelineTest, QuantitativeLinearAndFastGwaMlmTwoStagePipelinesWork) {
    gcta_test::TempDir tmp("gcta_fgwa_quant_");
    const auto ds = CreateFastGwaDataset(tmp, 180, 2050);

    // 1. --fastGWA-lr
    const std::string out_lr = tmp.file("out_lr");
    const auto r_lr = gcta_test::RunGctaCli({
        "--bfile", ds.prefix, "--pheno", ds.qphen_path,
        "--qcovar", ds.qcov_path, "--covar", ds.cov_path,
        "--fastGWA-lr", "--threads", "2", "--out", out_lr
    });
    ASSERT_EQ(r_lr.exit_code, 0) << r_lr.output;
    const auto lr_lines = gcta_test::ReadLines(out_lr + ".fastGWA");
    ASSERT_EQ(lr_lines.size(), static_cast<std::size_t>(ds.M + 1));
    // rs1 is the simulated causal SNP -> must be strongly significant
    const auto rs1_lr = gcta_test::SplitTabsOrSpaces(lr_lines[1]);
    EXPECT_EQ(rs1_lr[1], "rs1");
    EXPECT_GT(std::stod(rs1_lr[7]), 0.5);   // positive BETA
    EXPECT_LT(std::stod(rs1_lr[9]), 1e-6);  // strong P-value

    // 2. --fastGWA-mlm one-shot vs --model-only + --load-model
    const std::string out_mlm = tmp.file("out_mlm");
    const std::string out_mdl = tmp.file("out_qmdl");
    const std::string out_ld = tmp.file("out_qload");
    ASSERT_EQ(gcta_test::RunGctaCli({
        "--bfile", ds.prefix, "--grm-sparse", ds.prefix, "--pheno", ds.qphen_path,
        "--qcovar", ds.qcov_path, "--covar", ds.cov_path,
        "--fastGWA-mlm", "--seed", "12345", "--threads", "2", "--out", out_mlm
    }).exit_code, 0);
    ASSERT_EQ(gcta_test::RunGctaCli({
        "--bfile", ds.prefix, "--grm-sparse", ds.prefix, "--pheno", ds.qphen_path,
        "--qcovar", ds.qcov_path, "--covar", ds.cov_path,
        "--fastGWA-mlm", "--model-only", "--seed", "12345", "--threads", "2", "--out", out_mdl
    }).exit_code, 0);
    ASSERT_EQ(gcta_test::RunGctaCli({
        "--bfile", ds.prefix, "--load-model", out_mdl + ".fastGWA",
        "--threads", "2", "--out", out_ld
    }).exit_code, 0);

    const auto mlm_lines = gcta_test::ReadLines(out_mlm + ".fastGWA");
    const auto ld_lines = gcta_test::ReadLines(out_ld + ".fastGWA");
    ASSERT_EQ(mlm_lines.size(), ld_lines.size());
    const auto rs1_mlm = gcta_test::SplitTabsOrSpaces(mlm_lines[1]);
    const auto rs1_ld = gcta_test::SplitTabsOrSpaces(ld_lines[1]);
    EXPECT_LT(std::stod(rs1_mlm[9]), 1e-5);
    // Two-stage --load-model must match one-shot scan closely
    EXPECT_NEAR(std::stod(rs1_mlm[7]), std::stod(rs1_ld[7]), 1e-4);
    EXPECT_NEAR(std::stod(rs1_mlm[8]), std::stod(rs1_ld[8]), 1e-4);
}

TEST(FastFamPipelineTest, BinaryFastGwaMlmSpaLoadModelJointCovarExactPgenAndBurdenWork) {
    gcta_test::TempDir tmp("gcta_fgwa_bin_");
    const auto ds = CreateFastGwaDataset(tmp, 200, 2050);

    // 1. One-shot --fastGWA-mlm-binary vs two-stage --model-only + --load-model
    const std::string out_b1 = tmp.file("bin_oneshot");
    const std::string out_bmdl = tmp.file("bin_mdl");
    const std::string out_bload = tmp.file("bin_load_bed");
    const std::string out_pload = tmp.file("bin_load_pgen");

    const auto rb1 = gcta_test::RunGctaCli({
        "--bfile", ds.prefix, "--grm-sparse", ds.prefix, "--pheno", ds.bphen_path,
        "--qcovar", ds.qcov_path, "--covar", ds.cov_path,
        "--fastGWA-mlm-binary", "--seed", "12345", "--threads", "2", "--out", out_b1
    });
    ASSERT_EQ(rb1.exit_code, 0) << rb1.output;

    const auto rb_mdl = gcta_test::RunGctaCli({
        "--bfile", ds.prefix, "--grm-sparse", ds.prefix, "--pheno", ds.bphen_path,
        "--qcovar", ds.qcov_path, "--covar", ds.cov_path,
        "--fastGWA-mlm-binary", "--model-only", "--seed", "12345", "--threads", "2", "--out", out_bmdl
    });
    ASSERT_EQ(rb_mdl.exit_code, 0) << rb_mdl.output;

    ASSERT_EQ(gcta_test::RunGctaCli({
        "--bfile", ds.prefix, "--load-model", out_bmdl + ".fastGWA",
        "--threads", "2", "--out", out_bload
    }).exit_code, 0);

    // PGEN association scan with the same saved model must match BED scan
    ASSERT_EQ(gcta_test::RunGctaCli({
        "--pfile", ds.prefix, "--load-model", out_bmdl + ".fastGWA",
        "--threads", "2", "--out", out_pload
    }).exit_code, 0);

    const auto l_one = gcta_test::ReadLines(out_b1 + ".fastGWA");
    const auto l_bed = gcta_test::ReadLines(out_bload + ".fastGWA");
    const auto l_pgen = gcta_test::ReadLines(out_pload + ".fastGWA");
    ASSERT_EQ(l_one.size(), l_bed.size());
    ASSERT_EQ(l_bed.size(), l_pgen.size());
    ASSERT_GT(l_one.size(), 2000u);

    // Causal rs1 must be strongly detected with positive effect and SPA converged
    const auto c_one = gcta_test::SplitTabsOrSpaces(l_one[1]);
    const auto c_bed = gcta_test::SplitTabsOrSpaces(l_bed[1]);
    const auto c_pgen = gcta_test::SplitTabsOrSpaces(l_pgen[1]);
    EXPECT_EQ(c_one[1], "rs1");
    EXPECT_GT(std::stod(c_one[10]), 0.3);  // BETA > 0.3
    EXPECT_LT(std::stod(c_one[12]), 1e-3); // P < 1e-3
    EXPECT_EQ(c_one[13], "1");             // SPA converged

    // Allow sensible solver tolerance (~1e-3) while requiring BED == PGEN on identical model
    EXPECT_NEAR(std::stod(c_one[10]), std::stod(c_bed[10]), 5e-3);
    EXPECT_NEAR(std::stod(c_bed[10]), std::stod(c_pgen[10]), 1e-5);
    EXPECT_NEAR(std::stod(c_bed[11]), std::stod(c_pgen[11]), 1e-5);

    // Verify null calibration across non-causal SNPs (median chi-sq near ~0.455, bounded [0.15, 1.1])
    std::vector<double> null_p;
    for (std::size_t row_idx = 50; row_idx < 550; ++row_idx) {
        const auto cols = gcta_test::SplitTabsOrSpaces(l_bed[row_idx]);
        const double pval = std::stod(cols[12]);
        if (std::isfinite(pval)) null_p.push_back(pval);
    }
    std::sort(null_p.begin(), null_p.end());
    const double median_p = null_p[null_p.size() / 2];
    EXPECT_GT(median_p, 0.25);
    EXPECT_LT(median_p, 0.75);

    // 2. --joint-covar (--fastGWA-mlm-binary --joint-covar)
    const std::string out_jmdl = tmp.file("bin_joint_mdl");
    const std::string out_jload = tmp.file("bin_joint_load");
    ASSERT_EQ(gcta_test::RunGctaCli({
        "--bfile", ds.prefix, "--grm-sparse", ds.prefix, "--pheno", ds.bphen_path,
        "--qcovar", ds.qcov_path, "--fastGWA-mlm-binary", "--joint-covar",
        "--model-only", "--seed", "12345", "--threads", "2", "--out", out_jmdl
    }).exit_code, 0);
    ASSERT_EQ(gcta_test::RunGctaCli({
        "--bfile", ds.prefix, "--load-model", out_jmdl + ".fastGWA",
        "--threads", "2", "--out", out_jload
    }).exit_code, 0);
    const auto j_lines = gcta_test::ReadLines(out_jload + ".fastGWA");
    EXPECT_EQ(j_lines.size(), l_bed.size());

    // 3. Gene-based rare-variant burden test (--load-model --burden --set-list)
    const std::string out_burden = tmp.file("bin_burden");
    const auto r_burd = gcta_test::RunGctaCli({
        "--bfile", ds.prefix, "--load-model", out_bmdl + ".fastGWA",
        "--burden", "--set-list", ds.set_path, "--threads", "2", "--out", out_burden
    });
    ASSERT_EQ(r_burd.exit_code, 0) << r_burd.output;
    const auto burd_lines = gcta_test::ReadLines(out_burden + ".fastGWA");
    EXPECT_GE(burd_lines.size(), 2u);

    // 4. Quantitative --fastGWA-mlm-exact (no GRAMMAR-Gamma approximation)
    const std::string out_exact = tmp.file("quant_exact");
    const auto r_exact = gcta_test::RunGctaCli({
        "--bfile", ds.prefix, "--grm-sparse", ds.prefix, "--pheno", ds.qphen_path,
        "--qcovar", ds.qcov_path, "--fastGWA-mlm-exact", "--threads", "2", "--out", out_exact
    });
    ASSERT_EQ(r_exact.exit_code, 0) << r_exact.output;
    const auto exact_lines = gcta_test::ReadLines(out_exact + ".fastGWA");
    ASSERT_GT(exact_lines.size(), 2000u);
    const auto c_exact = gcta_test::SplitTabsOrSpaces(exact_lines[1]);
    EXPECT_EQ(c_exact[1], "rs1");
    EXPECT_GT(std::stod(c_exact[7]), 0.3);
    EXPECT_LT(std::stod(c_exact[9]), 1e-4);
}

// Regression test for upstream infinite loop in FastFAM::covarGLM (src/FastFAM.cpp:3797):
// On an exact 50/50 case-control cohort with intercept-only X, the true MLE is beta = 0,
// which produced 0.0 / 0.0 = NaN in mChange, while curIter was never incremented.
TEST(FastFamBugRegressionTest, CovarGlmTerminatesAndConvergesAtExactFiftyFiftyPrevalence) {
    const int rc = gcta_test::RunIsolated([&]() {
        alarm(2); // Abort with SIGALRM if covarGLM enters an infinite loop
        const int n = 40;
        Eigen::VectorXd y(n);
        for (int i = 0; i < n; ++i) y[i] = static_cast<double>(i % 2);
        Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, 1);
        Eigen::VectorXd beta = Eigen::VectorXd::Zero(1);
        ASSERT_TRUE(FastFAM::covarGLM(y, X, beta, /*maxIter=*/50, /*thresh=*/1e-6));
        EXPECT_NEAR(beta[0], 0.0, 1e-10);
    });
    EXPECT_EQ(rc, 0) << "covarGLM hung or failed on 50/50 prevalence (status=" << rc << ")";
}
