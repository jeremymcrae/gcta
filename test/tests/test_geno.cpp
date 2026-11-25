#include <gtest/gtest.h>
#include "Geno.h"
#include "Marker.h"
#include "Pheno.h"
#include "PgenReader.h"
#include "Logger.h"
#include "test_helpers.h"
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>

TEST(PgenReaderTest, ReadsHardcallsFrequenciesAndSampleSubsetsFromBedAndPgen) {
    gcta_test::TempDir tmp("gcta_pgen_");
    const std::string prefix = tmp.file("mini");

    std::vector<gcta_test::SampleSpec> samples = {
        {"F1", "I1", 1, 1.0},
        {"F2", "I2", 2, 2.0},
        {"F3", "I3", 1, 1.0},
        {"F4", "I4", 2, 2.0},
        {"F5", "I5", 1, 1.0},
        {"F6", "I6", 2, 2.0},
    };
    std::vector<gcta_test::SnpSpec> snps = {
        {"1", "rs1", 1000, "A", "G"},
        {"1", "rs2", 2000, "C", "T"},
    };
    // Copies of A1 (or -1 missing)
    std::vector<std::vector<int>> geno = {
        {0, 1, 2, -1, 2, 1}, // non-missing: {0,1,2,2,1}, N=5, sum=6, af=6/10=0.6
        {2, 2, 0,  1, 0, 1}, // non-missing: {2,2,0,1,0,1}, N=6, sum=6, af=6/12=0.5
    };
    gcta_test::WritePlinkBedDataset(prefix, samples, snps, geno);
    gcta_test::WritePlink2PgenDataset(prefix, samples, snps, geno);

    for (const std::string& ext : {".bed", ".pgen"}) {
        PgenReader reader;
        uint32_t raw_n = 6, raw_m = 2;
        std::vector<uint32_t> all_idx = {0, 1, 2, 3, 4, 5};
        reader.Load(prefix + ext, &raw_n, &raw_m, all_idx);
        EXPECT_EQ(reader.GetRawSampleCt(), 6u);
        EXPECT_EQ(reader.GetVariantCt(), 2u);

        std::vector<uintptr_t> raw_buf(reader.GetGenoBufFullSizeUptr(), 0);
        reader.ReadRawFullHard(raw_buf.data(), 0);

        const int mask_sz = PgenReader::GetSubsetMaskSize(6);
        std::vector<uintptr_t> incl(mask_sz, 0), inter(mask_sz, 0);
        PgenReader::SetSampleSubsets(all_idx, 6, incl.data(), inter.data());

        SNPInfo info{};
        ASSERT_TRUE(PgenReader::CountHardFreqMissExt(raw_buf.data(), inter.data(), 6, 6, &info, false));
        EXPECT_EQ(info.N, 5u);
        EXPECT_NEAR(info.af, 0.6, 1e-12);
        reader.Close();
    }
}

TEST(GenoTest, UnpacksIdenticalGenotypesAcrossBedPgenAndBgenFormats) {
    gcta_test::TempDir tmp("gcta_geno_fmt_");
    const std::string prefix = tmp.file("fmt");

    std::vector<gcta_test::SampleSpec> samples = {
        {"F1", "I1", 1, 1.0},
        {"F2", "I2", 2, 2.0},
        {"F3", "I3", 1, 1.0},
        {"F4", "I4", 2, 2.0},
        {"F5", "I5", 1, 1.0},
        {"F6", "I6", 2, 2.0},
        {"F7", "I7", 1, 1.0},
        {"F8", "I8", 2, 2.0},
    };
    std::vector<gcta_test::SnpSpec> snps = {
        {"1", "rs1", 1000, "A", "G"},
        {"1", "rs2", 2000, "C", "T"},
        {"2", "rs3", 3000, "G", "A"},
    };
    std::vector<std::vector<int>> geno = {
        {0, 1, 2, 0, 1, 2, 1, 1}, // complete, af = 8/16 = 0.5
        {2, 2, 1, 0, 1, 1, 0, 2}, // complete, af = 9/16 = 0.5625
        {0, 0, 1, 0, 1, 0, 2, 0}, // complete, af = 4/16 = 0.25
    };
    gcta_test::WritePlinkBedDataset(prefix, samples, snps, geno);
    gcta_test::WritePlink2PgenDataset(prefix, samples, snps, geno);
    gcta_test::WriteBgenV12Dataset(prefix, samples, snps, geno);

    auto extract_all = [&](const std::map<std::string, std::vector<std::string>>& in_opts,
                           bool std_geno,
                           std::vector<std::vector<double>>& out_geno,
                           std::vector<double>& out_af) {
        const std::string dump = tmp.file("dump.bin");
        const int rc = gcta_test::RunIsolated([&]() {
            LOGGER.open(tmp.file("geno_fmt.log"));
            auto opts = in_opts;
            Pheno::registerOption(opts);
            Marker::registerOption(opts);
            Geno::registerOption(opts);
            Pheno pheno;
            Marker marker;
            Geno g(&pheno, &marker);

            std::ofstream out(dump, std::ios::binary);
            std::vector<uint32_t> ex = {0, 1, 2};
            g.loopDouble(ex, 2, true, /*bGenoCenter=*/true, /*bGenoStd=*/std_geno, true,
                         {[&](uintptr_t* buf, const std::vector<uint32_t>& idx_vec) {
                             for (std::size_t k = 0; k < idx_vec.size(); ++k) {
                                 GenoBufItem item;
                                 item.extractedMarkerIndex = idx_vec[k];
                                 g.getGenoDouble(buf, static_cast<int>(k), &item);
                                 ASSERT_TRUE(item.valid);
                                 out.write(reinterpret_cast<const char*>(&item.af), sizeof(double));
                                 const uint32_t n = static_cast<uint32_t>(item.geno.size());
                                 out.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
                                 out.write(reinterpret_cast<const char*>(item.geno.data()), n * sizeof(double));
                             }
                         }},
                         false);
            LOGGER.close();
        });
        ASSERT_EQ(rc, 0);
        unlink(tmp.file("geno_fmt.log").c_str());

        std::ifstream in(dump, std::ios::binary);
        out_geno.clear();
        out_af.clear();
        for (int m = 0; m < 3; ++m) {
            double af = 0.0;
            uint32_t n = 0;
            in.read(reinterpret_cast<char*>(&af), sizeof(double));
            in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
            std::vector<double> row(n);
            in.read(reinterpret_cast<char*>(row.data()), n * sizeof(double));
            out_af.push_back(af);
            out_geno.push_back(std::move(row));
        }
    };

    // Centered genotypes (g - 2*af) and allele frequencies must match identically across BED, PGEN, and BGEN
    std::vector<std::vector<double>> g_bed, g_pgen, g_bgen;
    std::vector<double> af_bed, af_pgen, af_bgen;
    extract_all({{"--bfile", {prefix}}}, /*std_geno=*/false, g_bed, af_bed);
    extract_all({{"--pfile", {prefix}}}, /*std_geno=*/false, g_pgen, af_pgen);
    extract_all({{"--bgen", {prefix + ".bgen"}}, {"--sample", {prefix + ".sample"}}}, /*std_geno=*/false, g_bgen, af_bgen);

    for (int m = 0; m < 3; ++m) {
        EXPECT_NEAR(af_bed[m], af_pgen[m], 1e-12);
        EXPECT_NEAR(af_bed[m], af_bgen[m], 1e-6);
        ASSERT_EQ(g_bed[m].size(), 8u);
        for (std::size_t i = 0; i < 8; ++i) {
            EXPECT_NEAR(g_bed[m][i], g_pgen[m][i], 1e-12);
            EXPECT_NEAR(g_bed[m][i], g_bgen[m][i], 1e-5);
        }
    }

    // Standardized HWE genotypes must also match identically between BED and PGEN
    std::vector<std::vector<double>> s_bed, s_pgen;
    std::vector<double> saf_bed, saf_pgen;
    extract_all({{"--bfile", {prefix}}}, /*std_geno=*/true, s_bed, saf_bed);
    extract_all({{"--pfile", {prefix}}}, /*std_geno=*/true, s_pgen, saf_pgen);
    for (int m = 0; m < 3; ++m) {
        for (std::size_t i = 0; i < 8; ++i) {
            EXPECT_NEAR(s_bed[m][i], s_pgen[m][i], 1e-12);
        }
    }
}

TEST(GenoTest, CliFreqAndMakeBedFilterMafAndSamplesCorrectly) {
    gcta_test::TempDir tmp("gcta_geno_cli_");
    const std::string prefix = tmp.file("in");
    const std::string keep_file = tmp.file("keep.txt");
    const std::string out_freq = tmp.file("freq_out");
    const std::string out_bed = tmp.file("sub_bed");

    std::vector<gcta_test::SampleSpec> samples;
    for (int i = 0; i < 10; ++i) {
        samples.push_back({"F" + std::to_string(i), "I" + std::to_string(i), (i % 2) + 1, 1.0});
    }
    std::vector<gcta_test::SnpSpec> snps = {
        {"1", "rs_common", 1000, "A", "G"},
        {"1", "rs_mono",   2000, "C", "T"},
    };
    std::vector<std::vector<int>> geno = {
        {0, 1, 2, 1, 0, 2, 1, 1, 0, 2}, // AF = 10/20 = 0.5
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // monomorphic AF = 0.0 -> filtered by --maf 0.05
    };
    gcta_test::WritePlinkBedDataset(prefix, samples, snps, geno);

    {
        std::ofstream fk(keep_file);
        for (int i = 0; i < 8; ++i) {
            fk << "F" << i << "\tI" << i << "\n";
        }
    }

    const auto r1 = gcta_test::RunGctaCli({"--bfile", prefix, "--freq", "--out", out_freq});
    ASSERT_EQ(r1.exit_code, 0) << r1.output;
    const auto freq_lines = gcta_test::ReadLines(out_freq + ".frq");
    ASSERT_EQ(freq_lines.size(), 3u); // header + 2 SNPs
    const auto row_common = gcta_test::SplitTabsOrSpaces(freq_lines[1]);
    EXPECT_NEAR(std::stod(row_common[5]), 0.5, 1e-6);

    const std::string out_rec = tmp.file("rec_out");
    const auto r2 = gcta_test::RunGctaCli({
        "--bfile", prefix, "--keep", keep_file, "--maf", "0.05", "--recodet", "--out", out_rec
    });
    ASSERT_EQ(r2.exit_code, 0) << r2.output;
    const auto xmat_lines = gcta_test::ReadLines(out_rec + ".xmat");
    // Header + 1 polymorphic SNP (rs_mono filtered out by --maf 0.05)
    EXPECT_EQ(xmat_lines.size(), 2u);
}

// Regression test for upstream null-pointer segfault in Geno::processMain ("make_bed"):
// Commenting out init_keep()/init_AsyncBuffer() in Geno::Geno() left asyncBuffer and
// keep_mask NULL when running --bfile ... --make-bed.
TEST(GenoBugRegressionTest, MakeBedFromBfileFiltersMafAndSamplesWithoutSegfault) {
    gcta_test::TempDir tmp("gcta_make_bed_bug_");
    const std::string prefix = tmp.file("in");
    const std::string keep_file = tmp.file("keep.txt");
    const std::string out_bed = tmp.file("sub_bed");

    std::vector<gcta_test::SampleSpec> samples;
    for (int i = 0; i < 10; ++i) {
        samples.push_back({"F" + std::to_string(i), "I" + std::to_string(i), (i % 2) + 1, 1.0});
    }
    std::vector<gcta_test::SnpSpec> snps = {
        {"1", "rs_common", 1000, "A", "G"},
        {"1", "rs_mono",   2000, "C", "T"},
    };
    std::vector<std::vector<int>> geno = {
        {0, 1, 2, 1, 0, 2, 1, 1, 0, 2},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    };
    gcta_test::WritePlinkBedDataset(prefix, samples, snps, geno);
    {
        std::ofstream fk(keep_file);
        for (int i = 0; i < 8; ++i) fk << "F" << i << "\tI" << i << "\n";
    }

    const auto r = gcta_test::RunGctaCli({
        "--bfile", prefix, "--keep", keep_file, "--maf", "0.05", "--make-bed", "--out", out_bed
    });
    ASSERT_EQ(r.exit_code, 0) << r.output;
    EXPECT_EQ(gcta_test::ReadLines(out_bed + ".fam").size(), 8u);
    EXPECT_EQ(gcta_test::ReadLines(out_bed + ".bim").size(), 1u);
}

// Regression test for upstream bug in Geno::getGenoDouble_bed (src/Geno.cpp):
// When --update-ref-allele flips the reference allele (isEffRev == true), additive
// genotypes must swap aa0 (0.0) and aa2 (2.0) rather than setting both to 2.0,
// so raw dosages become (2 - g) and centered dosages negate (-g_centered).
TEST(GenoBugRegressionTest, UpdateRefAlleleFlipsAdditiveDosagesCleanly) {
    gcta_test::TempDir tmp("gcta_ref_flip_");
    const std::string prefix = tmp.file("flip");
    const std::string ref_file = tmp.file("ref.txt");

    std::vector<gcta_test::SampleSpec> samples = {
        {"F1", "I1", 1, 1.0},
        {"F2", "I2", 2, 2.0},
        {"F3", "I3", 1, 1.0},
        {"F4", "I4", 2, 2.0},
        {"F5", "I5", 1, 1.0},
        {"F6", "I6", 2, 2.0},
    };
    std::vector<gcta_test::SnpSpec> snps = {{"1", "rs_flip", 1000, "A", "G"}};
    // Sample 0 has 0 copies of A (2 copies of G); Sample 2 has 2 copies of A (0 copies of G)
    std::vector<std::vector<int>> geno = {{0, 1, 2, 0, 1, 0}};
    gcta_test::WritePlinkBedDataset(prefix, samples, snps, geno);

    {
        std::ofstream fr(ref_file);
        fr << "rs_flip\tG\n";
    }

    const int rc = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("flip.log"));
        std::map<std::string, std::vector<std::string>> opts = {
            {"--bfile", {prefix}},
            {"--update-ref-allele", {ref_file}}
        };
        Pheno::registerOption(opts);
        Marker::registerOption(opts);
        Geno::registerOption(opts);
        Pheno pheno;
        Marker marker;
        ASSERT_TRUE(marker.isEffecRev(0));

        Geno g(&pheno, &marker);
        std::vector<uint32_t> ex = {0};
        // Uncentered, unstandardized raw dosages after flipping to allele G:
        // Original copies of A: {0, 1, 2, 0, 1, 0} -> copies of G must be {2, 1, 0, 2, 1, 2}
        g.loopDouble(ex, 1, true, false, false, false,
                     {[&](uintptr_t* buf, const std::vector<uint32_t>& idx_vec) {
                         GenoBufItem item;
                         item.extractedMarkerIndex = idx_vec[0];
                         g.getGenoDouble(buf, 0, &item);
                         ASSERT_TRUE(item.valid);
                         ASSERT_EQ(item.geno.size(), 6u);
                         EXPECT_NEAR(item.geno[0], 2.0, 1e-12);
                         EXPECT_NEAR(item.geno[1], 1.0, 1e-12);
                         EXPECT_NEAR(item.geno[2], 0.0, 1e-12); // Upstream bug set aa2=2.0 here!
                         EXPECT_NEAR(item.geno[3], 2.0, 1e-12);
                         EXPECT_NEAR(item.geno[4], 1.0, 1e-12);
                         EXPECT_NEAR(item.geno[5], 2.0, 1e-12);
                     }},
                     false);
        LOGGER.close();
    });
    EXPECT_EQ(rc, 0);
}

// Regression test for two upstream bugs in Geno::getGenoDouble_bgen (src/Geno.cpp):
// 1. Missing ploidy under --keep pushed the raw BGEN sample index (sindex) rather than
//    the kept sample index (j) into miss_index, corrupting missingness bit positions.
// 2. gbuf->missing.resize(missPtrSize, 0) failed to clear stale bits when reusing a
//    GenoBufItem across SNPs, leaking SNP 0's missing bit into complete SNP 1.
TEST(GenoBugRegressionTest, BgenMissingnessUsesKeptSampleIndexAndClearsAcrossSnps) {
    gcta_test::TempDir tmp("gcta_bgen_miss_");
    const std::string prefix = tmp.file("bmiss");
    const std::string keep_file = tmp.file("keep.txt");

    std::vector<gcta_test::SampleSpec> samples;
    for (int i = 0; i < 10; ++i) {
        samples.push_back({"F" + std::to_string(i), "I" + std::to_string(i), (i % 2) + 1, 1.0});
    }
    std::vector<gcta_test::SnpSpec> snps = {
        {"1", "rs_miss0", 1000, "A", "G"},
        {"1", "rs_comp1", 2000, "C", "T"},
    };
    // Raw sample 8 is missing on SNP 0; SNP 1 has NO missing calls at all.
    std::vector<std::vector<int>> geno = {
        {0, 1, 2, 0, 1, 2, 0, 1, -1, 2},
        {2, 1, 0, 2, 1, 0, 1, 2,  1, 0},
    };
    gcta_test::WriteBgenV12Dataset(prefix, samples, snps, geno);

    // Keep 6 samples: raw indices {1, 3, 4, 6, 8, 9} -> raw sample 8 is kept sample index 4!
    {
        std::ofstream fk(keep_file);
        for (int idx : {1, 3, 4, 6, 8, 9}) {
            fk << "F" << idx << "\tI" << idx << "\n";
        }
    }

    const int rc = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("bmiss.log"));
        std::map<std::string, std::vector<std::string>> opts = {
            {"--bgen", {prefix + ".bgen"}},
            {"--sample", {prefix + ".sample"}},
            {"--keep", {keep_file}}
        };
        Pheno::registerOption(opts);
        Marker::registerOption(opts);
        Geno::registerOption(opts);
        Pheno pheno;
        Marker marker;
        Geno g(&pheno, &marker);

        std::vector<uint32_t> ex = {0, 1};
        GenoBufItem reused_item; // Deliberately reused across SNPs like GRM::calculate_GRM_blas
        g.loopDouble(ex, 2, true, true, true, true,
                     {[&](uintptr_t* buf, const std::vector<uint32_t>& idx_vec) {
                         ASSERT_EQ(idx_vec.size(), 2u);

                         // SNP 0: kept sample 4 (raw sample 8) is missing -> bit 4 must be 1, all other bits 0
                         reused_item.extractedMarkerIndex = idx_vec[0];
                         g.getGenoDouble(buf, 0, &reused_item);
                         ASSERT_TRUE(reused_item.valid);
                         ASSERT_FALSE(reused_item.missing.empty());
                         EXPECT_EQ(reused_item.missing[0], static_cast<uintptr_t>(1ULL << 4))
                             << "Expected missing bit at kept index 4 (mask=16), got mask=" << reused_item.missing[0];

                         // SNP 1: 0 missing calls -> reused_item.missing[0] must be cleared to 0!
                         reused_item.extractedMarkerIndex = idx_vec[1];
                         g.getGenoDouble(buf, 1, &reused_item);
                         ASSERT_TRUE(reused_item.valid);
                         ASSERT_FALSE(reused_item.missing.empty());
                         EXPECT_EQ(reused_item.missing[0], static_cast<uintptr_t>(0))
                             << "Stale missing bits leaked into complete SNP 1!";
                     }},
                     false);
        LOGGER.close();
    });
    EXPECT_EQ(rc, 0);
}
