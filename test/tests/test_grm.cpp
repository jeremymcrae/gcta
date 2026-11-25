#include <gtest/gtest.h>
#include "test_helpers.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <zlib.h>

namespace {

std::vector<std::vector<int>> BuildSyntheticCohort(const std::string& prefix, int N = 16, int M = 24, bool with_missing = true) {
    std::mt19937 rng(20261124);
    std::vector<gcta_test::SampleSpec> samples(N);
    for (int i = 0; i < N; ++i) {
        samples[i] = {"F" + std::to_string(i + 1), "I" + std::to_string(i + 1), (i % 2) + 1, 1.0};
    }

    std::vector<gcta_test::SnpSpec> snps(M);
    std::vector<std::vector<int>> geno(M, std::vector<int>(N, 0));
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    for (int m = 0; m < M; ++m) {
        snps[m] = {std::to_string((m * 2) / M + 1), "rs" + std::to_string(m + 1),
                   static_cast<uint32_t>((m + 1) * 1000), "A", "G"};
        const double p = 0.2 + 0.6 * (static_cast<double>(m % 5) / 5.0);
        int sum_g = 0, cnt = 0;
        for (int i = 0; i < N; ++i) {
            if (with_missing && ((m * 7 + i * 3) % 29 == 0)) {
                geno[m][i] = -1;
            } else {
                const int g = (u01(rng) < p ? 1 : 0) + (u01(rng) < p ? 1 : 0);
                geno[m][i] = g;
                sum_g += g;
                ++cnt;
            }
        }
        // Guarantee non-monomorphic SNP
        if (sum_g == 0 || sum_g == 2 * cnt) {
            geno[m][0] = 0;
            geno[m][1] = 2;
        }
    }

    gcta_test::WritePlinkBedDataset(prefix, samples, snps, geno);
    gcta_test::WritePlink2PgenDataset(prefix, samples, snps, geno);
    gcta_test::WriteBgenV12Dataset(prefix, samples, snps, geno);
    return geno;
}

// Exact closed-form GCTA autosomal GRM from integer genotypes in {0, 1, 2, -1}.
// When empirical_var == false (BED/PGEN), denominator is 2 * af * (1 - af);
// when empirical_var == true (BGEN without --update-freq), denominator is sample variance.
void ComputeExactGrm(const std::vector<std::vector<int>>& geno_m_by_n,
                     int N,
                     std::vector<double>& grm_lower,
                     std::vector<float>& n_lower,
                     bool empirical_var = false) {
    const std::size_t npairs = static_cast<std::size_t>(N) * (N + 1) / 2;
    grm_lower.assign(npairs, 0.0);
    std::vector<int> counts(npairs, 0);

    for (std::size_t m = 0; m < geno_m_by_n.size(); ++m) {
        double sum = 0.0, sum2 = 0.0;
        int n_valid = 0;
        for (int i = 0; i < N; ++i) {
            if (geno_m_by_n[m][i] >= 0) {
                const double g = static_cast<double>(geno_m_by_n[m][i]);
                sum += g;
                sum2 += g * g;
                ++n_valid;
            }
        }
        if (n_valid <= 1) continue;
        const double mean = sum / n_valid;
        const double af = sum / (2.0 * n_valid);
        const double denom = empirical_var
            ? ((sum2 - sum * mean) / (n_valid - 1))
            : (2.0 * af * (1.0 - af));
        if (denom <= 1e-12) continue;

        std::size_t k = 0;
        for (int i = 0; i < N; ++i) {
            const int gi = geno_m_by_n[m][i];
            for (int j = 0; j <= i; ++j, ++k) {
                const int gj = geno_m_by_n[m][j];
                if (gi >= 0 && gj >= 0) {
                    grm_lower[k] += ((gi - mean) * (gj - mean)) / denom;
                    counts[k] += 1;
                }
            }
        }
    }

    n_lower.resize(npairs);
    for (std::size_t k = 0; k < npairs; ++k) {
        n_lower[k] = static_cast<float>(counts[k]);
        if (counts[k] > 0) grm_lower[k] /= static_cast<double>(counts[k]);
    }
}

} // namespace

TEST(GrmV2Test, MatchesClosedFormFormulaWithMissingnessThreadsAndPartitions) {
    gcta_test::TempDir tmp("gcta_grm_exact_");
    const std::string prefix = tmp.file("exact");
    const int N = 12, M = 18;

    std::vector<gcta_test::SampleSpec> samples(N);
    for (int i = 0; i < N; ++i) {
        samples[i] = {"F" + std::to_string(i + 1), "I" + std::to_string(i + 1), (i % 2) + 1, 1.0};
    }
    std::vector<gcta_test::SnpSpec> snps(M);
    std::vector<std::vector<int>> geno(M, std::vector<int>(N, 0));
    for (int m = 0; m < M; ++m) {
        snps[m] = {"1", "rs" + std::to_string(m + 1), static_cast<uint32_t>((m + 1) * 500), "A", "G"};
        for (int i = 0; i < N; ++i) {
            if ((m + i) % 11 == 0) geno[m][i] = -1;
            else geno[m][i] = (m * 3 + i * 5) % 3;
        }
        geno[m][0] = 0;
        geno[m][1] = 2;
    }
    gcta_test::WritePlinkBedDataset(prefix, samples, snps, geno);

    std::vector<double> exp_grm;
    std::vector<float> exp_n;
    ComputeExactGrm(geno, N, exp_grm, exp_n);

    const std::string out1 = tmp.file("grm_t1");
    const std::string out4 = tmp.file("grm_t4");
    ASSERT_EQ(gcta_test::RunGctaCli({"--bfile", prefix, "--make-grm", "--threads", "1", "--out", out1}).exit_code, 0);
    ASSERT_EQ(gcta_test::RunGctaCli({"--bfile", prefix, "--make-grm", "--threads", "4", "--out", out4}).exit_code, 0);

    const auto g1 = gcta_test::ReadBinaryFloats(out1 + ".grm.bin");
    const auto n1 = gcta_test::ReadBinaryFloats(out1 + ".grm.N.bin");
    const auto g4 = gcta_test::ReadBinaryFloats(out4 + ".grm.bin");
    ASSERT_EQ(g1.size(), exp_grm.size());
    ASSERT_EQ(n1.size(), exp_n.size());
    ASSERT_EQ(g4.size(), g1.size());

    for (std::size_t k = 0; k < exp_grm.size(); ++k) {
        EXPECT_NEAR(n1[k], exp_n[k], 1e-5f);
        EXPECT_NEAR(g1[k], exp_grm[k], 1e-5);
        EXPECT_NEAR(g4[k], g1[k], 1e-6f);
    }

    // Partitioned --make-grm-part 3 1..3 concatenated must match full --make-grm
    const std::string outp = tmp.file("grm_part");
    std::vector<float> g_concat, n_concat;
    for (int p = 1; p <= 3; ++p) {
        const auto rp = gcta_test::RunGctaCli({
            "--bfile", prefix, "--make-grm-part", "3", std::to_string(p), "--threads", "2", "--out", outp
        });
        ASSERT_EQ(rp.exit_code, 0) << rp.output;
        const std::string suffix = ".part_3_" + std::to_string(p);
        const auto gp = gcta_test::ReadBinaryFloats(outp + suffix + ".grm.bin");
        const auto np = gcta_test::ReadBinaryFloats(outp + suffix + ".grm.N.bin");
        g_concat.insert(g_concat.end(), gp.begin(), gp.end());
        n_concat.insert(n_concat.end(), np.begin(), np.end());
    }
    ASSERT_EQ(g_concat.size(), g1.size());
    for (std::size_t k = 0; k < g1.size(); ++k) {
        EXPECT_NEAR(g_concat[k], g1[k], 1e-6f);
        EXPECT_NEAR(n_concat[k], n1[k], 1e-5f);
    }
}

TEST(GrmV2Test, BedPgenAndBgenProduceConsistentGrmAndDominanceAndChrXWork) {
    gcta_test::TempDir tmp("gcta_grm_modes_");
    const std::string prefix = tmp.file("modes");
    const auto geno = BuildSyntheticCohort(prefix, 14, 20, /*with_missing=*/false);

    std::vector<double> expected_bgen_grm;
    std::vector<float> expected_bgen_n;
    ComputeExactGrm(geno, 14, expected_bgen_grm, expected_bgen_n, /*empirical_var=*/true);

    const std::string out_bed = tmp.file("g_bed");
    const std::string out_pgen = tmp.file("g_pgen");
    const std::string out_bgen = tmp.file("g_bgen");
    const std::string out_dom = tmp.file("g_dom");

    ASSERT_EQ(gcta_test::RunGctaCli({"--bfile", prefix, "--make-grm", "--out", out_bed}).exit_code, 0);
    ASSERT_EQ(gcta_test::RunGctaCli({"--pfile", prefix, "--make-grm", "--out", out_pgen}).exit_code, 0);
    ASSERT_EQ(gcta_test::RunGctaCli({
        "--bgen", prefix + ".bgen", "--sample", prefix + ".sample", "--make-grm", "--out", out_bgen
    }).exit_code, 0);
    ASSERT_EQ(gcta_test::RunGctaCli({"--bfile", prefix, "--make-grm-d", "--out", out_dom}).exit_code, 0);

    const auto gb = gcta_test::ReadBinaryFloats(out_bed + ".grm.bin");
    const auto gp = gcta_test::ReadBinaryFloats(out_pgen + ".grm.bin");
    const auto gx = gcta_test::ReadBinaryFloats(out_bgen + ".grm.bin");
    const auto gd = gcta_test::ReadBinaryFloats(out_dom + ".d.grm.bin");
    ASSERT_EQ(gb.size(), gp.size());
    ASSERT_EQ(gb.size(), gx.size());
    ASSERT_EQ(gb.size(), gd.size());

    for (std::size_t k = 0; k < gb.size(); ++k) {
        EXPECT_NEAR(gb[k], gp[k], 1e-5f);
        EXPECT_NEAR(gx[k], static_cast<float>(expected_bgen_grm[k]), 2e-4f);
        EXPECT_TRUE(std::isfinite(gd[k]));
    }

    // X-chromosome GRM (--make-grm-xchr): haploid males {0, 2}, diploid females {0, 1, 2}
    const std::string xpref = tmp.file("xchr");
    std::vector<gcta_test::SampleSpec> xsamples = {
        {"F1", "I1", 1, 1.0}, {"F2", "I2", 1, 1.0}, {"F3", "I3", 1, 1.0},
        {"F4", "I4", 2, 1.0}, {"F5", "I5", 2, 1.0}, {"F6", "I6", 2, 1.0},
    };
    std::vector<gcta_test::SnpSpec> xsnps = {
        {"23", "rsX1", 1000, "A", "G"},
        {"23", "rsX2", 2000, "C", "T"},
        {"23", "rsX3", 3000, "G", "A"},
    };
    std::vector<std::vector<int>> xgeno = {
        {0, 2, 0, 0, 1, 2},
        {2, 0, 2, 2, 1, 0},
        {0, 2, 2, 1, 0, 2},
    };
    gcta_test::WritePlinkBedDataset(xpref, xsamples, xsnps, xgeno);
    const std::string out_xchr = tmp.file("g_xchr");
    const auto rx = gcta_test::RunGctaCli({"--bfile", xpref, "--make-grm-xchr", "--out", out_xchr});
    ASSERT_EQ(rx.exit_code, 0) << rx.output;
    const auto g_xchr = gcta_test::ReadBinaryFloats(out_xchr + ".grm.bin");
    ASSERT_EQ(g_xchr.size(), 21u);
    for (float v : g_xchr) EXPECT_TRUE(std::isfinite(v));
}

TEST(GrmPostProcessAndV1Test, SparseBkCutoffUnifySubtractMgrmGzPcaAndRemlWork) {
    gcta_test::TempDir tmp("gcta_grm_post_");

    // Create a cohort of 40 individuals (including 5 identical-twin/duplicate pairs for high relatedness)
    const int N = 40, M = 80;
    std::mt19937 rng(99);
    std::vector<gcta_test::SampleSpec> samples(N);
    for (int i = 0; i < N; ++i) {
        samples[i] = {"F" + std::to_string(i + 1), "I" + std::to_string(i + 1), (i % 2) + 1, 0.0};
    }

    std::vector<gcta_test::SnpSpec> snps1(M / 2), snps2(M / 2), snps_all(M);
    std::vector<std::vector<int>> geno1(M / 2, std::vector<int>(N)), geno2(M / 2, std::vector<int>(N)), geno_all(M);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::vector<double> genetic_val(N, 0.0);

    for (int m = 0; m < M; ++m) {
        snps_all[m] = {"1", "rs" + std::to_string(m + 1), static_cast<uint32_t>((m + 1) * 1000), "A", "G"};
        std::vector<int> row(N);
        for (int i = 0; i < N - 5; ++i) {
            row[i] = static_cast<int>(rng() % 3);
        }
        // Last 5 samples copy samples 0..4 to create strong relatedness > 0.8
        for (int k = 0; k < 5; ++k) {
            row[N - 5 + k] = row[k];
        }
        row[0] = 0;
        row[1] = 2;
        row[N - 5] = 0;
        row[N - 4] = 2;

        const double beta_m = (m < 15) ? 0.5 : 0.0;
        for (int i = 0; i < N; ++i) genetic_val[i] += beta_m * row[i];

        geno_all[m] = row;
        if (m < M / 2) {
            snps1[m] = snps_all[m];
            geno1[m] = row;
        } else {
            snps2[m - M / 2] = snps_all[m];
            geno2[m - M / 2] = row;
        }
    }
    for (int i = 0; i < N; ++i) {
        samples[i].pheno = genetic_val[i] + 0.3 * norm(rng);
    }

    const std::string p_all = tmp.file("all");
    const std::string p_1 = tmp.file("part1");
    const std::string p_2 = tmp.file("part2");
    gcta_test::WritePlinkBedDataset(p_all, samples, snps_all, geno_all);
    gcta_test::WritePlinkBedDataset(p_1, samples, snps1, geno1);
    gcta_test::WritePlinkBedDataset(p_2, samples, snps2, geno2);

    const std::string g_all = tmp.file("grm_all");
    const std::string g_1 = tmp.file("grm_1");
    const std::string g_2 = tmp.file("grm_2");
    ASSERT_EQ(gcta_test::RunGctaCli({"--bfile", p_all, "--make-grm", "--out", g_all}).exit_code, 0);
    ASSERT_EQ(gcta_test::RunGctaCli({"--bfile", p_1, "--make-grm", "--out", g_1}).exit_code, 0);
    ASSERT_EQ(gcta_test::RunGctaCli({"--bfile", p_2, "--make-grm", "--out", g_2}).exit_code, 0);

    // 1. --make-bK-sparse
    const std::string g_sp = tmp.file("grm_sp");
    ASSERT_EQ(gcta_test::RunGctaCli({"--grm", g_all, "--make-bK-sparse", "0.25", "--out", g_sp}).exit_code, 0);
    const auto sp_lines = gcta_test::ReadLines(g_sp + ".grm.sp");
    // Must contain all 40 diagonal entries plus the 5 duplicate-pair off-diagonals
    EXPECT_GE(sp_lines.size(), 45u);

    // 2. --grm-cutoff 0.5 prunes at least 5 of the duplicate individuals
    const std::string g_cut = tmp.file("grm_cut");
    ASSERT_EQ(gcta_test::RunGctaCli({"--grm", g_all, "--grm-cutoff", "0.5", "--make-grm", "--out", g_cut}).exit_code, 0);
    const auto cut_ids = gcta_test::ReadLines(g_cut + ".grm.id");
    EXPECT_LE(cut_ids.size(), 35u);
    EXPECT_GE(cut_ids.size(), 25u);

    // 3. V1 --mgrm --make-grm merges grm_1 and grm_2 -> must match grm_all within 1e-5!
    const std::string mgrm_list = tmp.file("mgrm.txt");
    {
        std::ofstream fm(mgrm_list);
        fm << g_1 << "\n" << g_2 << "\n";
    }
    const std::string g_merged = tmp.file("grm_merged");
    const auto rm = gcta_test::RunGctaCli({"--mgrm", mgrm_list, "--make-grm", "--out", g_merged});
    ASSERT_EQ(rm.exit_code, 0) << rm.output;
    const auto bin_all = gcta_test::ReadBinaryFloats(g_all + ".grm.bin");
    const auto bin_mrg = gcta_test::ReadBinaryFloats(g_merged + ".grm.bin");
    ASSERT_EQ(bin_all.size(), bin_mrg.size());
    for (std::size_t k = 0; k < bin_all.size(); ++k) {
        EXPECT_NEAR(bin_all[k], bin_mrg[k], 2e-5f);
    }

    // 4. --subtract-grm: subtracting grm_1 from [grm_all, grm_1] recovers grm_2!
    const std::string sub_list = tmp.file("sub_mgrm.txt");
    {
        std::ofstream fs(sub_list);
        fs << g_all << "\n" << g_1 << "\n";
    }
    const std::string g_sub = tmp.file("grm_sub");
    const auto rs = gcta_test::RunGctaCli({"--mgrm", sub_list, "--subtract-grm", "--out", g_sub});
    ASSERT_EQ(rs.exit_code, 0) << rs.output;
    const auto bin_2 = gcta_test::ReadBinaryFloats(g_2 + ".grm.bin");
    const auto bin_sub = gcta_test::ReadBinaryFloats(g_sub + ".grm.bin");
    ASSERT_EQ(bin_2.size(), bin_sub.size());
    for (std::size_t k = 0; k < bin_2.size(); ++k) {
        EXPECT_NEAR(bin_2[k], bin_sub[k], 5e-5f);
    }

    // 5. --unify-grm across two disjoint/overlapping subsets with permuted sample orders
    const std::string keep_a = tmp.file("keep_a.txt");
    const std::string keep_b = tmp.file("keep_b.txt");
    {
        std::ofstream fa(keep_a), fb(keep_b);
        for (int i = 0; i < 20; ++i) fa << samples[i].fid << "\t" << samples[i].iid << "\n";
        for (int i = 29; i >= 10; --i) fb << samples[i].fid << "\t" << samples[i].iid << "\n";
    }
    const std::string ga = tmp.file("ga");
    const std::string gb = tmp.file("gb");
    ASSERT_EQ(gcta_test::RunGctaCli({"--bfile", p_1, "--keep", keep_a, "--make-grm", "--out", ga}).exit_code, 0);
    ASSERT_EQ(gcta_test::RunGctaCli({"--bfile", p_2, "--keep", keep_b, "--make-grm", "--out", gb}).exit_code, 0);
    const std::string uni_list = tmp.file("uni.txt");
    {
        std::ofstream fu(uni_list);
        fu << ga << "\n" << gb << "\n";
    }
    const std::string g_uni = tmp.path() + "//g_uni";
    const auto ru = gcta_test::RunGctaCli({"--mgrm", uni_list, "--unify-grm", "--out", g_uni});
    ASSERT_EQ(ru.exit_code, 0) << ru.output;
    const auto uni_ids_a = gcta_test::ReadLines(tmp.file("ga_g_uni.grm.id"));
    const auto uni_ids_b = gcta_test::ReadLines(tmp.file("gb_g_uni.grm.id"));
    EXPECT_EQ(uni_ids_a.size(), 10u);
    EXPECT_EQ(uni_ids_a, uni_ids_b);

    // 6. V1 --make-grm-gz, --pca 3, and --reml
    const std::string g_gz = tmp.file("grm_gz");
    ASSERT_EQ(gcta_test::RunGctaCli({"--grm", g_cut, "--make-grm-gz", "--out", g_gz}).exit_code, 0);
    gzFile gzf = gzopen((g_gz + ".grm.gz").c_str(), "rb");
    ASSERT_NE(gzf, nullptr);
    char gzbuf[256];
    EXPECT_NE(gzgets(gzf, gzbuf, sizeof(gzbuf)), nullptr);
    gzclose(gzf);

    const std::string out_pca = tmp.file("pca_out");
    const auto rpca = gcta_test::RunGctaCli({"--grm", g_cut, "--pca", "3", "--out", out_pca});
    ASSERT_EQ(rpca.exit_code, 0) << rpca.output;
    const auto eval_lines = gcta_test::ReadLines(out_pca + ".eigenval");
    ASSERT_GE(eval_lines.size(), 3u);
    EXPECT_GE(std::stod(eval_lines[0]), std::stod(eval_lines[1]));
    EXPECT_GT(std::stod(eval_lines[0]), 0.0);

    const std::string phen_file = tmp.file("reml.phen");
    {
        std::ofstream fp(phen_file);
        for (const auto& s : samples) {
            fp << s.fid << "\t" << s.iid << "\t" << s.pheno << "\n";
        }
    }
    const std::string out_reml = tmp.file("reml_out");
    const auto rreml = gcta_test::RunGctaCli({"--grm", g_cut, "--pheno", phen_file, "--reml", "--out", out_reml});
    ASSERT_EQ(rreml.exit_code, 0) << rreml.output;
    const auto hsq_lines = gcta_test::ReadLines(out_reml + ".hsq");
    EXPECT_GE(hsq_lines.size(), 5u);
}

// Regression test: flipping the reference allele of a subset of SNPs via --update-ref-allele
// only negates standardized genotypes (z -> -z), leaving z_i * z_j and the GRM invariant.
TEST(GrmBugRegressionTest, MakeGrmIsInvariantUnderUpdateRefAllele) {
    gcta_test::TempDir tmp("gcta_grm_ref_");
    const std::string prefix = tmp.file("refinv");
    BuildSyntheticCohort(prefix, 12, 16, /*with_missing=*/true);

    const std::string ref_file = tmp.file("flip_half.txt");
    {
        std::ofstream fr(ref_file);
        for (int m = 1; m <= 16; ++m) {
            fr << "rs" << m << "\t" << ((m % 2 == 1) ? "G" : "A") << "\n";
        }
    }

    const std::string out_orig = tmp.file("g_orig");
    const std::string out_flip = tmp.file("g_flip");
    ASSERT_EQ(gcta_test::RunGctaCli({"--bfile", prefix, "--make-grm", "--out", out_orig}).exit_code, 0);
    ASSERT_EQ(gcta_test::RunGctaCli({
        "--bfile", prefix, "--update-ref-allele", ref_file, "--make-grm", "--out", out_flip
    }).exit_code, 0);

    const auto g_orig = gcta_test::ReadBinaryFloats(out_orig + ".grm.bin");
    const auto g_flip = gcta_test::ReadBinaryFloats(out_flip + ".grm.bin");
    ASSERT_EQ(g_orig.size(), g_flip.size());
    for (std::size_t k = 0; k < g_orig.size(); ++k) {
        EXPECT_NEAR(g_orig[k], g_flip[k], 1e-5f) << "GRM changed at pair index " << k << " under --update-ref-allele!";
    }
}

// Regression test: --bgen --sample --keep --make-grm on data with missing genotypes
// must match exact non-missing pair counts (.grm.N.bin) and closed-form BGEN GRM (.grm.bin).
TEST(GrmBugRegressionTest, BgenMakeGrmWithKeepAndMissingMatchesBed) {
    gcta_test::TempDir tmp("gcta_grm_bgen_keep_");
    const std::string prefix = tmp.file("bk");
    const auto full_geno = BuildSyntheticCohort(prefix, 16, 20, /*with_missing=*/true);

    const std::vector<int> kept_idx = {1, 2, 4, 5, 7, 9, 11, 12, 14, 15};
    const std::string keep_file = tmp.file("keep.txt");
    {
        std::ofstream fk(keep_file);
        for (int i : kept_idx) {
            fk << "F" << (i + 1) << "\tI" << (i + 1) << "\n";
        }
    }

    std::vector<std::vector<int>> sub_geno(full_geno.size(), std::vector<int>(kept_idx.size()));
    for (std::size_t m = 0; m < full_geno.size(); ++m) {
        for (std::size_t k = 0; k < kept_idx.size(); ++k) {
            sub_geno[m][k] = full_geno[m][kept_idx[k]];
        }
    }
    std::vector<double> exp_bgen_grm;
    std::vector<float> exp_bgen_n;
    ComputeExactGrm(sub_geno, static_cast<int>(kept_idx.size()), exp_bgen_grm, exp_bgen_n, /*empirical_var=*/true);

    const std::string out_bed = tmp.file("bed_grm");
    const std::string out_bgen = tmp.file("bgen_grm");
    ASSERT_EQ(gcta_test::RunGctaCli({
        "--bfile", prefix, "--keep", keep_file, "--make-grm", "--threads", "2", "--out", out_bed
    }).exit_code, 0);
    ASSERT_EQ(gcta_test::RunGctaCli({
        "--bgen", prefix + ".bgen", "--sample", prefix + ".sample",
        "--keep", keep_file, "--make-grm", "--threads", "2", "--out", out_bgen
    }).exit_code, 0);

    const auto n_bed = gcta_test::ReadBinaryFloats(out_bed + ".grm.N.bin");
    const auto n_bgen = gcta_test::ReadBinaryFloats(out_bgen + ".grm.N.bin");
    const auto g_bgen = gcta_test::ReadBinaryFloats(out_bgen + ".grm.bin");
    ASSERT_EQ(n_bed.size(), n_bgen.size());
    ASSERT_EQ(g_bgen.size(), exp_bgen_grm.size());
    for (std::size_t k = 0; k < n_bed.size(); ++k) {
        EXPECT_NEAR(n_bed[k], n_bgen[k], 1e-5f) << "Non-missing SNP count mismatch at pair " << k;
        EXPECT_NEAR(g_bgen[k], static_cast<float>(exp_bgen_grm[k]), 3e-4f) << "GRM value mismatch at pair " << k;
    }
}
