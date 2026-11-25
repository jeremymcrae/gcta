#include <gtest/gtest.h>
#include "Pheno.h"
#include "Logger.h"
#include "test_helpers.h"
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>

TEST(PhenoTest, LoadsCommittedFamAndPhenFixtures) {
    gcta_test::TempDir tmp("gcta_pheno_");
    const int rc = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("pheno.log"));
        std::map<std::string, std::vector<std::string>> opts = {
            {"--fam", {CUR_SRC_DIR + "/data/test.fam"}},
            {"--pheno", {CUR_SRC_DIR + "/data/test.phen"}}
        };
        Pheno::registerOption(opts);
        Pheno pheno;
        EXPECT_EQ(pheno.count_raw(), 3925u);
        EXPECT_EQ(pheno.count_keep(), 3925u);

        const auto first_ids = pheno.get_id(0, 2, "\t");
        ASSERT_EQ(first_ids.size(), 3u);
        EXPECT_FALSE(first_ids[0].empty());

        std::vector<std::string> p_ids;
        std::vector<double> p_vals;
        pheno.get_pheno(p_ids, p_vals);
        EXPECT_EQ(p_ids.size(), p_vals.size());
        EXPECT_GT(p_vals.size(), 3000u);

        std::vector<uint64_t> mask_words((pheno.count_raw() + 63) / 64, 0);
        pheno.getMaskBit(mask_words.data());
        EXPECT_EQ(mask_words[0], ~0ULL);
        LOGGER.close();
    });
    EXPECT_EQ(rc, 0);
}

TEST(PhenoTest, KeepRemoveMultiPhenoUpdateSexAndPsamSampleFormats) {
    gcta_test::TempDir tmp("gcta_pheno_sub_");
    const std::string prefix = tmp.file("cohort");
    const std::string phen_path = tmp.file("multi.phen");
    const std::string keep_path = tmp.file("keep.txt");
    const std::string rm_path = tmp.file("rm.txt");
    const std::string sex_path = tmp.file("sex.txt");

    std::vector<gcta_test::SampleSpec> samples = {
        {"F1", "I1", 1, 1.0},
        {"F2", "I2", 2, 2.0},
        {"F3", "I3", 0, 1.0}, // unknown sex initially
        {"F4", "I4", 2, 2.0},
        {"F5", "I5", 1, 1.0},
    };
    std::vector<gcta_test::SnpSpec> snps = {{"1", "rs1", 1000, "A", "G"}};
    gcta_test::WritePlinkBedDataset(prefix, samples, snps, {{0, 1, 2, 1, 0}});
    gcta_test::WritePlink2PgenDataset(prefix, samples, snps, {{0, 1, 2, 1, 0}});
    gcta_test::WriteBgenV12Dataset(prefix, samples, snps, {{0, 1, 2, 1, 0}});

    {
        std::ofstream fp(phen_path);
        fp << "F1\tI1\t10.0\t100.5\n";
        fp << "F2\tI2\t20.0\t-9\n";
        fp << "F3\tI3\t30.0\t300.5\n";
        fp << "F4\tI4\t40.0\t400.5\n";
        fp << "F5\tI5\t50.0\t500.5\n";
    }
    {
        std::ofstream fk(keep_path);
        fk << "F1\tI1\nF3\tI3\nF4\tI4\nF5\tI5\n";
    }
    {
        std::ofstream fr(rm_path);
        fr << "F5\tI5\n";
    }
    {
        std::ofstream fs(sex_path);
        fs << "F1\tI1\t1\n";
        fs << "F3\tI3\t1\n"; // update F3 from unknown (0) -> male (1)
        fs << "F4\tI4\t2\n";
    }

    // 1. Test --bfile + --pheno + --mpheno 2 + --keep + --remove + --update-sex
    const int rc1 = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("p1.log"));
        std::map<std::string, std::vector<std::string>> opts = {
            {"--bfile", {prefix}},
            {"--pheno", {phen_path}},
            {"--mpheno", {"2"}},
            {"--keep", {keep_path}},
            {"--remove", {rm_path}},
            {"--update-sex", {sex_path}},
            {"--filter-sex", {}}
        };
        Pheno::registerOption(opts);
        Pheno pheno;
        EXPECT_EQ(pheno.count_raw(), 5u);
        // Kept {F1, F3, F4, F5} minus removed {F5} = {F1, F3, F4} (all have valid sex after --update-sex)
        EXPECT_EQ(pheno.count_keep(), 3u);
        EXPECT_EQ(pheno.count_male(), 2u); // F1 and updated F3

        std::vector<std::string> ids;
        std::vector<double> vals;
        pheno.get_pheno(ids, vals);
        ASSERT_EQ(vals.size(), 3u);
        EXPECT_NEAR(vals[0], 100.5, 1e-12);
        EXPECT_NEAR(vals[1], 300.5, 1e-12);
        EXPECT_NEAR(vals[2], 400.5, 1e-12);
        LOGGER.close();
    });
    EXPECT_EQ(rc1, 0);

    // 2. Test PLINK2 .psam loading
    const int rc2 = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("p2.log"));
        std::map<std::string, std::vector<std::string>> opts = {
            {"--pfile", {prefix}},
            {"--filter-sex", {}}
        };
        Pheno::registerOption(opts);
        Pheno pheno;
        EXPECT_EQ(pheno.count_raw(), 5u);
        // F3 has sex=0 in .psam, so --filter-sex retains 4 individuals
        EXPECT_EQ(pheno.count_keep(), 4u);
        LOGGER.close();
    });
    EXPECT_EQ(rc2, 0);

    // 3. Test Oxford .sample loading
    const int rc3 = gcta_test::RunIsolated([&]() {
        LOGGER.open(tmp.file("p3.log"));
        std::map<std::string, std::vector<std::string>> opts = {
            {"--sample", {prefix + ".sample"}}
        };
        Pheno::registerOption(opts);
        Pheno pheno;
        EXPECT_EQ(pheno.count_raw(), 5u);
        EXPECT_EQ(pheno.count_keep(), 5u);
        LOGGER.close();
    });
    EXPECT_EQ(rc3, 0);
}
