#include <gtest/gtest.h>
#include "Marker.h"
#include "Logger.h"
#include "test_helpers.h"
#include <fstream>
#include <map>
#include <string>
#include <vector>

TEST(MarkerTest, LoadsCommittedBimAndMapsChromosomes) {
    gcta_test::TempDir tmp("gcta_marker_");
    const int rc = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("marker.log"));
        std::map<std::string, std::vector<std::string>> opts = {
            {"--bim", {CUR_SRC_DIR + "/data/test.bim"}}
        };
        Marker::registerOption(opts);
        Marker marker;
        EXPECT_EQ(marker.count_raw(), 1000u);
        EXPECT_EQ(marker.count_extract(), 1000u);

        bool ok = false;
        EXPECT_EQ(marker.mapCHR("1", ok), 1);
        EXPECT_TRUE(ok);
        EXPECT_EQ(marker.mapCHR("22", ok), 22);
        EXPECT_TRUE(ok);
        EXPECT_EQ(marker.mapCHR("X", ok), 23);
        EXPECT_TRUE(ok);
        EXPECT_EQ(marker.mapCHR("Y", ok), 24);
        EXPECT_TRUE(ok);

        const auto m0 = gcta_test::SplitTabsOrSpaces(marker.getMarkerStrExtract(0));
        const auto m1 = gcta_test::SplitTabsOrSpaces(marker.getMarkerStrExtract(1));
        ASSERT_GE(m0.size(), 2u);
        ASSERT_GE(m1.size(), 2u);
        marker.extract_marker({m0[1], m1[1]}, true);
        EXPECT_EQ(marker.count_extract(), 2u);
        LOGGER.close();
    });
    EXPECT_EQ(rc, 0);

    // Extracting zero matching SNPs must exit cleanly via Logger::e
    const int rc_empty = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("marker_empty.log"));
        std::map<std::string, std::vector<std::string>> opts = {
            {"--bim", {CUR_SRC_DIR + "/data/test.bim"}}
        };
        Marker::registerOption(opts);
        Marker marker;
        marker.extract_marker({"nonexistent_snp"}, true);
    });
    EXPECT_EQ(rc_empty, EXIT_FAILURE);
}

TEST(MarkerTest, SupportsExtractExcludeChrFilterAlleleFlipGeneSetsPvarAndBgen) {
    gcta_test::TempDir tmp("gcta_marker_ext_");
    const std::string prefix = tmp.file("panel");
    const std::string extract_path = tmp.file("extract.snplist");
    const std::string ref_allele_path = tmp.file("ref_allele.txt");
    const std::string gene_path = tmp.file("genes.set");

    std::vector<gcta_test::SampleSpec> samples = {{"F1", "I1", 1, 1.0}, {"F2", "I2", 2, 2.0}};
    std::vector<gcta_test::SnpSpec> snps = {
        {"1",  "rs1", 1000, "A", "G"},
        {"1",  "rs2", 2000, "C", "T"},
        {"2",  "rs3", 1500, "G", "A"},
        {"23", "rsX", 3000, "T", "C"},
    };
    std::vector<std::vector<int>> geno = {{0, 2}, {1, 1}, {2, 0}, {0, 2}};
    gcta_test::WritePlinkBedDataset(prefix, samples, snps, geno);
    gcta_test::WritePlink2PgenDataset(prefix, samples, snps, geno);
    gcta_test::WriteBgenV12Dataset(prefix, samples, snps, geno);

    {
        std::ofstream fe(extract_path);
        fe << "rs1\nrs2\nrs3\n";
    }
    {
        // Set G (originally A2 on rs1) as reference/effect allele -> triggers A_rev[0] == true
        std::ofstream fr(ref_allele_path);
        fr << "rs1\tG\nrs2\tC\nrs3\tG\n";
    }
    {
        std::ofstream fg(gene_path);
        fg << "gene\tchr\tstart\tend\tstrand\n";
        fg << "GENE1\t1\t500\t2500\t+\n";
        fg << "GENE2\t2\t1000\t2000\t+\n";
    }

    // 1. Test --bfile + --extract + --update-ref-allele + read_gene + windowing
    const int rc1 = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("m1.log"));
        std::map<std::string, std::vector<std::string>> opts = {
            {"--bfile", {prefix}},
            {"--extract", {extract_path}},
            {"--update-ref-allele", {ref_allele_path}}
        };
        Marker::registerOption(opts);
        Marker marker;
        EXPECT_EQ(marker.count_raw(), 4u);
        EXPECT_EQ(marker.count_extract(), 3u);
        EXPECT_TRUE(marker.isEffecRev(0));  // rs1 flipped to G
        EXPECT_FALSE(marker.isEffecRev(1)); // rs2 kept as C

        const auto genes = marker.read_gene(gene_path);
        ASSERT_EQ(genes.size(), 2u);
        EXPECT_EQ(genes[0].second.size(), 2u); // rs1, rs2 on chr 1
        EXPECT_EQ(genes[1].second.size(), 1u); // rs3 on chr 2

        bool chr_ends = false, is_x = false;
        const auto win = marker.getNextSizeIndex(0, 10, chr_ends, is_x);
        // Stops at chromosome 1 boundary -> 2 markers (rs1, rs2)
        EXPECT_EQ(win.size(), 2u);
        EXPECT_TRUE(chr_ends);
        EXPECT_FALSE(is_x);
        LOGGER.close();
    });
    EXPECT_EQ(rc1, 0);

    // 2. Test PLINK2 .pvar + --autosome
    const int rc2 = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("m2.log"));
        std::map<std::string, std::vector<std::string>> opts = {
            {"--pfile", {prefix}},
            {"--autosome", {}}
        };
        Marker::registerOption(opts);
        Marker marker;
        EXPECT_EQ(marker.count_raw(), 4u);
        EXPECT_EQ(marker.count_extract(), 3u); // excludes chr 23 (X)
        LOGGER.close();
    });
    EXPECT_EQ(rc2, 0);

    // 3. Test BGEN v1.2 + .bgi index loading + --chrx
    const int rc3 = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("m3.log"));
        std::map<std::string, std::vector<std::string>> opts = {
            {"--bgen", {prefix + ".bgen"}},
            {"--chrx", {}}
        };
        Marker::registerOption(opts);
        Marker marker;
        EXPECT_EQ(marker.count_extract(), 1u);
        EXPECT_NE(marker.getMarkerStrExtract(0).find("rsX"), std::string::npos);
        LOGGER.close();
    });
    EXPECT_EQ(rc3, 0);
}
